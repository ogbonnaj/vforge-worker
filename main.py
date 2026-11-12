"""VFORGE server worker (step 6).

Run:   python backend/worker/main.py            (real providers)
       python backend/worker/main.py --fake     (test: tiny triangle models, no GPU quota)
       python backend/worker/main.py --check    (test Supabase + show each Space's API, uses no quota)
       python backend/worker/main.py --once     (handle what is waiting, then exit)

       python backend/worker/main.py --max-minutes 30 --remote-state   (cloud: stop when queue empty / all resting / time up)
       python backend/worker/main.py --peek     (cloud: is there work? writes work=yes/no to $GITHUB_OUTPUT)

Run ONLY ONE worker at a time. Needs internet all the time.
Secrets: SUPABASE_SERVICE_KEY (env var) or backend/worker/worker_secrets.json. Never put them in the app.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from providers import (ORDER, FakeProvider, ProviderError, QuotaError, build_providers)  # noqa: E402
from supa import Supa, SupaError  # noqa: E402

MAX_ATTEMPTS = int(os.environ.get("VFORGE_MAX_ATTEMPTS", "20"))   # full passes that failed (quota waits do not count)
MODELS_PER_JOB = int(os.environ.get("VFORGE_MODELS_PER_JOB", "2"))
BACKOFF_BASE = float(os.environ.get("VFORGE_BACKOFF", "10"))       # seconds; doubles up to 5 minutes
CALL_TIMEOUT = float(os.environ.get("VFORGE_CALL_TIMEOUT", "600"))  # seconds one Space call may take before it counts as failed
STRIKES_BEFORE_PAUSE = 3                                          # a provider failing 3x in a row rests 10 min
LOG_FILE = os.path.join(HERE, "worker.log")
STATE_FILE = os.path.join(HERE, "worker_state.json")


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 500_000:
            os.replace(LOG_FILE, LOG_FILE + ".old")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_config():
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    secrets = {}
    p = os.path.join(HERE, "worker_secrets.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            secrets = json.load(f)
    url = os.environ.get("SUPABASE_URL") or secrets.get("supabase_url", "")
    if not url:
        try:
            with open(os.path.join(root, "config", "supabase.json"), encoding="utf-8") as f:
                url = json.load(f)["supabase_url"]
        except (OSError, ValueError, KeyError):
            sys.exit("Missing SUPABASE_URL (env var / secret)")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or secrets.get("service_key", "")
    if not key:
        sys.exit("Missing SUPABASE_SERVICE_KEY (env var) or service_key in backend/worker/worker_secrets.json")
    hf = os.environ.get("HF_TOKEN") or secrets.get("hf_token", "")
    return url, key, hf


def call_with_timeout(fn, seconds):
    """Run fn in a helper thread; if it hangs longer than `seconds`, give up on it (counts as a failure)."""
    box = {}

    def run():
        try:
            box["v"] = fn()
        except BaseException as e:      # noqa: BLE001 - re-raised below
            box["e"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise ProviderError("timed out after %d s" % seconds)
    if "e" in box:
        raise box["e"]
    return box.get("v")


class Worker:
    def __init__(self, supa, providers, sleeper=time.sleep, clock=time.time, remote_state=False):
        self.supa, self.providers = supa, providers
        self.sleep, self.clock = sleeper, clock
        self.remote_state = remote_state     # True in the cloud: rest times live in Supabase, not on the machine
        self.idle = False                    # last step() found no queued jobs at all
        self.resting = False                 # last step() found every provider resting
        self.until = {}      # provider -> time when it may be used again
        self.strikes = {}    # provider -> consecutive failures
        self.retry_at = {}   # job id -> time of next try
        self._last_wait_msg = ""
        self._load_state()

    # ---- remembered quota waits survive a restart ----
    def _load_state(self):
        if self.remote_state:
            try:
                self.until = {k: float(v) for k, v in self.supa.get_state().get("until", {}).items()}
            except (SupaError, ValueError):
                self.until = {}
            return
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                self.until = {k: float(v) for k, v in json.load(f).get("until", {}).items()}
        except (OSError, ValueError):
            self.until = {}

    def _save_state(self):
        if self.remote_state:
            try:
                self.supa.put_state({"until": self.until})
            except SupaError as e:
                log("could not save rest times: %s" % e)
            return
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({"until": self.until}, f)
        except OSError:
            pass

    def available(self, quality):
        order = ORDER.get(quality, ORDER["high"])
        now = self.clock()
        return [n for n in order if n in self.providers and self.until.get(n, 0) <= now]

    def next_free_time(self):
        times = [t for n, t in self.until.items() if n in self.providers]
        return min(times) if times else self.clock() + 60

    # ---- one model: try providers in order, switching on quota/failure ----
    def make_one(self, job, image, seed, workdir):
        """Returns (file, provider_name, None) | (None, None, 'quota') | (None, None, error text)."""
        last_err = ""
        for name in self.available(job.get("quality", "high")):
            prov = self.providers[name]
            try:
                log("job %s: trying %s (model seed %d)" % (job["id"][:8], name, seed))
                path = call_with_timeout(lambda: prov.generate(image, seed, workdir), CALL_TIMEOUT)
                self.strikes[name] = 0
                return path, name, None
            except QuotaError as e:
                self.until[name] = self.clock() + e.wait_seconds
                self._save_state()
                log("%s quota used up, resting %d min; switching provider" % (name, e.wait_seconds // 60))
            except ProviderError as e:
                last_err = "%s: %s" % (name, e)
                self.strikes[name] = self.strikes.get(name, 0) + 1
                log("%s failed: %s" % (name, e))
                if self.strikes[name] >= STRIKES_BEFORE_PAUSE:
                    self.until[name] = self.clock() + 600
                    self.strikes[name] = 0
                    self._save_state()
        if last_err:
            return None, None, last_err
        return None, None, "quota"   # nothing was available to try

    def process(self, job):
        jid, workdir = job["id"], tempfile.mkdtemp(prefix="vforge_")
        try:
            if job.get("positive_prompt") or job.get("negative_prompt"):
                log("job %s: prompts received (image-to-3D Spaces take only the image, prompts not used)" % jid[:8])
            image = os.path.join(workdir, "input" + os.path.splitext(job["input_path"])[1])
            try:
                self.supa.download(job["input_path"], image)
            except SupaError as e:
                if e.status in (400, 404):
                    return self.supa.patch(jid, {"status": "failed", "error": "input image not found on server"})
                raise
            outputs, last_err, quota_stop = [], "", False
            for n in range(1, MODELS_PER_JOB + 1):
                f, prov, err = self.make_one(job, image, seed=1000 + n * 7919, workdir=workdir)
                if f:
                    ext = os.path.splitext(f)[1].lower() or ".glb"
                    remote = "projects/%s/output/%s_%d%s" % (job["project_id"], job["asset_id"], n, ext)
                    self.supa.upload(remote, f, "model/gltf-binary" if ext == ".glb" else "application/octet-stream")
                    outputs.append(remote)
                    self.supa.patch(jid, {"provider": prov})
                elif err == "quota":
                    quota_stop = True
                    break
                else:
                    last_err = err
                    break
            if outputs:
                note = None
                if len(outputs) < MODELS_PER_JOB:
                    note = "only %d of %d models made (%s)" % (
                        len(outputs), MODELS_PER_JOB, "quota used up" if quota_stop else last_err)
                self.supa.patch(jid, {"status": "done", "outputs": outputs, "error": note})
                log("job %s: done with %d model(s)" % (jid[:8], len(outputs)))
            elif quota_stop:
                self.supa.patch(jid, {"status": "queued"})      # waits, not counted as a failed try
                self.retry_at[jid] = self.next_free_time()
                self._note_wait()
            else:
                attempts = int(job.get("attempts") or 0) + 1
                if attempts >= MAX_ATTEMPTS:
                    self.supa.patch(jid, {"status": "failed", "attempts": attempts, "error": last_err[:300]})
                    log("job %s: FAILED after %d attempts: %s" % (jid[:8], attempts, last_err))
                else:
                    self.supa.patch(jid, {"status": "queued", "attempts": attempts, "error": last_err[:300]})
                    self.retry_at[jid] = self.clock() + min(300, BACKOFF_BASE * 2 ** min(attempts, 5))
                    log("job %s: attempt %d/%d failed, retrying later" % (jid[:8], attempts, MAX_ATTEMPTS))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _note_wait(self):
        msg = "all providers resting; waiting until " + time.strftime("%H:%M", time.localtime(self.next_free_time()))
        if msg != self._last_wait_msg:
            log(msg)
            self._last_wait_msg = msg

    def step(self):
        """Do one unit of work. Returns seconds to sleep before the next step."""
        self.idle = self.resting = False
        try:
            jobs = self.supa.queued_jobs()
        except SupaError as e:
            log("cannot reach Supabase: %s" % e)
            return 30
        now = self.clock()
        ready = [j for j in jobs if self.retry_at.get(j["id"], 0) <= now]
        if not jobs:
            self.idle = True
        if not ready:
            waits = [t - now for t in self.retry_at.values() if t > now]
            return max(2, min(waits)) if waits and jobs else 5
        job = ready[0]
        if not self.available(job.get("quality", "high")):
            # Both quality orders contain every provider, so this means all are resting.
            self._note_wait()
            self.resting = True
            return max(30, min(600, self.next_free_time() - now))
        try:
            claimed = self.supa.claim(job["id"])
            if not claimed:
                return 1
            self.process(claimed)
        except SupaError as e:
            log("Supabase error: %s" % e)
            return 30
        except Exception as e:     # never let one bad job stop the worker
            log("unexpected error: %r" % e)
            try:
                self.supa.patch(job["id"], {"status": "queued", "error": ("worker error: %s" % e)[:300]})
            except SupaError:
                pass
            self.retry_at[job["id"]] = self.clock() + 60
            return 5
        return 0


def check(supa, hf):
    print("Supabase: ", end="")
    try:
        n = len(supa.queued_jobs(100))
        print("OK, %d queued job(s)" % n)
    except SupaError as e:
        print("FAILED -", e)
    for name, p in build_providers(hf).items():
        print("\n==== %s (%s) ====" % (name, p.space))
        try:
            p.client().view_api()
        except Exception as e:
            print("cannot connect:", str(e)[:300])


def peek(supa):
    """Cheap check for the cloud schedule (standard library only). Writes work=yes|no to $GITHUB_OUTPUT."""
    work = False
    try:
        queued, busy = supa.queued_jobs(1), supa.count_processing()
        until = supa.get_state().get("until", {})
        now = time.time()
        resting = all(float(until.get(n, 0)) > now for n in ORDER["high"])
        work = bool(busy) or (bool(queued) and not resting)
        print("queued: %s, stuck processing: %s, all providers resting: %s -> work=%s" % (
            len(queued), len(busy), resting, "yes" if work else "no"))
    except SupaError as e:
        print("peek: cannot check Supabase: %s" % e)
        if e.status in (401, 403):
            print("The Supabase key is wrong or not allowed. Fix the SUPABASE_SERVICE_KEY secret.")
            return 1
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write("work=%s\n" % ("yes" if work else "no"))
    return 0


def main():
    args = sys.argv[1:]
    url, key, hf = load_config()
    supa = Supa(url, key)
    if "--peek" in args:
        return peek(supa)
    if "--check" in args:
        return check(supa, hf)
    providers = {"fake": FakeProvider()} if "--fake" in args else build_providers(hf)
    if "--fake" in args:
        for q in ORDER:
            ORDER[q] = ["fake"]
    deadline = None
    if "--max-minutes" in args:
        deadline = time.time() + float(args[args.index("--max-minutes") + 1]) * 60
    w = Worker(supa, providers, remote_state=("--remote-state" in args and "--fake" not in args))
    try:
        n = supa.reset_processing(MAX_ATTEMPTS)
        if n:
            log("%d job(s) left 'processing' by an earlier run were put back in the queue" % n)
    except SupaError as e:
        log("start-up check failed: %s" % e)
    log("worker started (%s)" % ("FAKE mode" if "--fake" in args else ", ".join(providers)))
    while True:
        s = w.step()
        if "--once" in args and s >= 2:
            break
        if deadline:
            if w.idle:
                log("queue is empty, stopping")
                break
            if w.resting:
                log("all providers resting, stopping (next run resumes)")
                break
            if time.time() + s >= deadline:
                log("time limit reached, stopping (next run continues)")
                break
        if s:
            time.sleep(s)


if __name__ == "__main__":
    sys.exit(main() or 0)
