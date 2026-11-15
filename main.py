"""VFORGE server worker (step 6).

Run:   python backend/worker/main.py            (real providers)
       python backend/worker/main.py --fake     (test: tiny triangle models, no GPU quota)
       python backend/worker/main.py --check    (test Supabase + show each Space's API, uses no quota)
       python backend/worker/main.py --once     (handle what is waiting, then exit)

       python backend/worker/main.py --max-minutes 30 --remote-state   (cloud: stop when queue empty / all resting / time up)
       python backend/worker/main.py --peek     (cloud: is there work? writes work=yes/no to $GITHUB_OUTPUT)

       python backend/worker/main.py --texture [JOB_ID] [--model N]   (texture ONE finished white model; blank id = latest done job)
       python backend/worker/main.py --texture --fake-texture         (test of the texture step, no GPU)
       python backend/worker/main.py --texture-folder [--fake-texture] (texture what is waiting in bucket folder texture/in/)

Texturing uses its OWN Hugging Face token: HF_TEXTURE_TOKEN (or "hf_texture_token" in worker_secrets.json).
The shape token (HF_TOKEN) is never used for texturing, so shape quota is not touched.

Run ONLY ONE worker at a time. Needs internet all the time.
Secrets: SUPABASE_SERVICE_KEY (env var) or backend/worker/worker_secrets.json. Never put them in the app.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from providers import (ORDER, TEXTURE_ORDER, FakeProvider, FakeTexture, ProviderError, QuotaError,  # noqa: E402
                       IMG_EXT, _is_textured, _valid_model, api_summary, build_candidates, build_providers,
                       build_texture_providers)
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


def load_texture_token():
    """Token of the SEPARATE Hugging Face account used only for texturing. Blank = unauthenticated (small shared quota)."""
    secrets = {}
    p = os.path.join(HERE, "worker_secrets.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                secrets = json.load(f)
        except (OSError, ValueError):
            secrets = {}
    return os.environ.get("HF_TEXTURE_TOKEN") or secrets.get("hf_texture_token", "")


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
    def __init__(self, supa, providers, sleeper=time.sleep, clock=time.time, remote_state=False, tex_providers=None):
        self.supa, self.providers = supa, providers
        self.tex_providers = tex_providers or {}
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
                    if self.tex_providers and ext == ".glb":
                        tf, terr = _texture_with(self, self.tex_providers, list(TEXTURE_ORDER), f, image,
                                                 1000 + n * 7919, workdir,
                                                 job.get("positive_prompt") or "", job.get("negative_prompt") or "", jid[:8])
                        if tf:
                            f = tf
                        else:
                            log("job %s: texturing failed (%s), uploading white model" % (jid[:8], terr))
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


def check(supa, hf, hf_tex=""):
    print("Supabase: ", end="")
    try:
        n = len(supa.queued_jobs(100))
        print("OK, %d queued job(s)" % n)
    except SupaError as e:
        print("FAILED -", e)
    print("texture token: %s" % ("set" if hf_tex else "NOT set (texturing would use the small shared anonymous quota)"))
    for name, p in list(build_providers(hf).items()) + list(build_texture_providers(hf_tex).items()):
        print("\n==== %s (%s) ====" % (name, p.space))
        try:
            p.client().view_api()
        except Exception as e:
            print("cannot connect:", str(e)[:300])
    print("\n######## CANDIDATE Spaces (only looked at, nothing is run, no GPU used) ########")
    for name, p in build_candidates(hf).items():
        print("\n== %s (%s) ==" % (name, p.space))
        try:
            print("\n".join(api_summary(p.client())))
        except Exception as e:
            print("cannot connect:", str(e)[:200])


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
    if not work:
        try:      # items waiting for texturing (explicitly sent by you, so they are always worth a run)
            until = supa.get_state().get("until", {})
            tex_free = any(float(until.get(n, 0)) <= time.time() for n in TEXTURE_ORDER)
            waiting = texture_folders(supa)
            if waiting and tex_free:
                work = True
            print("texture items waiting: %d, a texture Space free: %s" % (len(waiting), tex_free))
        except SupaError:
            pass
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write("work=%s\n" % ("yes" if work else "no"))
    return 0


def _arg_after(args, flag, default=None):
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            return args[i + 1]
    return default


def _texture_with(w, providers, order, mesh, image, seed, workdir, prompt, negative, label):
    """Try the texture Spaces in order. Returns (path or None, last error text). Quota waits are remembered."""
    last = ""
    for name in order:
        if name not in providers:
            continue
        if w.until.get(name, 0) > w.clock():
            log("%s is resting until %s, skipping" % (name, time.strftime("%H:%M", time.localtime(w.until[name]))))
            last = last or "%s is resting (GPU quota)" % name
            continue
        prov = providers[name]
        try:
            log("%s: texturing with %s" % (label, name))
            out = call_with_timeout(lambda: prov.texture(mesh, image, seed, workdir, prompt, negative), CALL_TIMEOUT)
            if not _valid_model(out) or not _is_textured(out, getattr(prov, "allow_vertex_colors", False)):
                raise ProviderError("returned a model with NO texture")
            return out, ""
        except QuotaError as e:
            w.until[name] = w.clock() + e.wait_seconds
            w._save_state()
            last = "%s quota used up" % name
            log("%s quota used up, resting %d min; trying the next texture Space" % (name, e.wait_seconds // 60))
        except ProviderError as e:
            last = "%s: %s" % (name, e)
            log("%s failed: %s" % (name, e))
    return None, last


def texture_job(supa, hf, args):
    """Texture ONE finished white model that is STILL on the server (the app deletes models after download,
    so this only works before the app has downloaded them; for downloaded models use --texture-folder).
    --texture [JOB_ID]   blank = the most recently finished job;  --model N = which model of the job (default 1)
    Saves <white model name>_tex.glb next to the white model. Returns 0 on success, 1 if not."""
    job_id = (_arg_after(args, "--texture") or "").strip()
    try:
        model_n = int(_arg_after(args, "--model", "1"))
    except ValueError:
        model_n = 1
    fake = "--fake-texture" in args
    if job_id and not re.fullmatch(r"[0-9a-fA-F-]{36}", job_id):
        log("job id must be the FULL id (36 characters) from the Supabase jobs table, or blank for the latest job")
        return 1
    try:
        job = supa.get_job(job_id) if job_id else supa.latest_done_job()
    except SupaError as e:
        log("cannot read the job: %s" % e)
        return 1
    if not job:
        log("no such job found (finished jobs disappear once the app has downloaded them)" if job_id
            else "no finished job found on the server - use --texture-folder for models already downloaded")
        return 1
    outs = job.get("outputs") or []
    if isinstance(outs, str):
        try:
            outs = json.loads(outs)
        except ValueError:
            outs = []
    white = [o for o in outs if "_tex" not in o]
    if not white:
        log("job %s has no finished white model yet" % job["id"][:8])
        return 1
    if not 1 <= model_n <= len(white):
        log("job %s has %d model(s); --model %d does not exist" % (job["id"][:8], len(white), model_n))
        return 1
    remote_white = white[model_n - 1]
    providers = {"fake_texture": FakeTexture()} if fake else build_texture_providers(hf)
    order = ["fake_texture"] if fake else list(TEXTURE_ORDER)
    w = Worker(supa, providers, remote_state=("--remote-state" in args and not fake))
    workdir = tempfile.mkdtemp(prefix="vforge_tex_")
    try:
        image = os.path.join(workdir, "input" + os.path.splitext(job["input_path"])[1])
        mesh = os.path.join(workdir, "white" + (os.path.splitext(remote_white)[1] or ".glb"))
        try:
            supa.download(job["input_path"], image)
            supa.download(remote_white, mesh)
        except SupaError as e:
            log("cannot download the image or the white model: %s" % e)
            return 1
        out, err = _texture_with(w, providers, order, mesh, image, 1000 + model_n * 7919, workdir,
                                 job.get("positive_prompt") or "", job.get("negative_prompt") or "",
                                 "job %s model %d" % (job["id"][:8], model_n))
        if not out:
            log("no texture Space worked, nothing was changed (%s). Try again later." % err)
            return 1
        remote_tex = os.path.splitext(remote_white)[0] + "_tex.glb"
        supa.upload(remote_tex, out, "model/gltf-binary")
        log("job %s: textured model saved as %s" % (job["id"][:8], remote_tex))
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


TEX_IN, TEX_OUT = "texture/in", "texture/out"


def texture_folders(supa):
    """Names of the item folders waiting in texture/in/ ."""
    return sorted(supa.list_dir(TEX_IN))


def texture_folder(supa, hf, args):
    """Texture what is waiting in the bucket. One folder per item:
         texture/in/<name>/model.glb      (white model, required)
         texture/in/<name>/image.png|jpg|jpeg|webp   (reference image, required)
         texture/in/<name>/prompt.txt     (optional: line 1 = positive prompt, line 2 = negative prompt)
       Result:  texture/out/<name>/model_tex.glb   (inputs are deleted after success).
       On failure the inputs stay and texture/out/<name>/error.txt says why. Returns 0 even if nothing to do."""
    fake = "--fake-texture" in args
    providers = {"fake_texture": FakeTexture()} if fake else build_texture_providers(hf)
    order = ["fake_texture"] if fake else list(TEXTURE_ORDER)
    try:
        names = texture_folders(supa)
    except SupaError as e:
        log("cannot list texture/in: %s" % e)
        return 1
    if not names:
        log("nothing waiting in texture/in")
        return 0
    w = Worker(supa, providers, remote_state=("--remote-state" in args and not fake))
    for name in names:
        base = "%s/%s" % (TEX_IN, name)
        try:
            files = supa.list_dir(base)
        except SupaError as e:
            log("%s: cannot list: %s" % (name, e))
            continue
        model = "model.glb" if "model.glb" in files else None
        image = next((f for f in files if f.lower().startswith("image.") and f.lower().endswith(IMG_EXT)), None)
        if not model or not image:
            log("%s: needs model.glb and image.png/jpg/webp (found: %s), skipping" % (name, ", ".join(files) or "nothing"))
            continue
        if all(w.until.get(n, 0) > w.clock() for n in order if n in providers):
            log("all texture Spaces are resting, leaving the rest for the next run")
            break
        workdir = tempfile.mkdtemp(prefix="vforge_texf_")
        try:
            lp_m, lp_i = os.path.join(workdir, "white.glb"), os.path.join(workdir, "input" + os.path.splitext(image)[1])
            try:
                supa.download(base + "/" + model, lp_m)
                supa.download(base + "/" + image, lp_i)
                pos = neg = ""
                if "prompt.txt" in files:
                    lp_p = os.path.join(workdir, "prompt.txt")
                    supa.download(base + "/prompt.txt", lp_p)
                    with open(lp_p, encoding="utf-8", errors="replace") as f:
                        lines = (f.read().splitlines() + ["", ""])[:2]
                    pos, neg = lines[0].strip(), lines[1].strip()
            except SupaError as e:
                log("%s: cannot download inputs: %s" % (name, e))
                continue
            out, err = _texture_with(w, providers, order, lp_m, lp_i, 1000 + 7919, workdir, pos, neg, name)
            err_path = "%s/%s/error.txt" % (TEX_OUT, name)
            if out:
                supa.upload("%s/%s/model_tex.glb" % (TEX_OUT, name), out, "model/gltf-binary")
                supa.delete(err_path)
                for f in files:
                    supa.delete(base + "/" + f)
                log("%s: textured model saved in %s/%s/model_tex.glb" % (name, TEX_OUT, name))
            else:
                ep = os.path.join(workdir, "error.txt")
                with open(ep, "w", encoding="utf-8") as f:
                    f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), err or "failed"))
                try:
                    supa.upload(err_path, ep, "text/plain")
                except SupaError:
                    pass
                log("%s: not textured (%s); inputs kept, will retry on a later run" % (name, err))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    return 0


def main():
    args = sys.argv[1:]
    url, key, hf = load_config()
    supa = Supa(url, key)
    if "--peek" in args:
        return peek(supa)
    hf_tex = load_texture_token()
    if "--check" in args:
        return check(supa, hf, hf_tex)
    if "--texture-folder" in args:
        return texture_folder(supa, hf_tex, args)
    if "--texture" in args:
        return texture_job(supa, hf_tex, args)
    providers = {"fake": FakeProvider()} if "--fake" in args else build_providers(hf)
    if "--fake" in args:
        for q in ORDER:
            ORDER[q] = ["fake"]
    deadline = None
    if "--max-minutes" in args:
        deadline = time.time() + float(args[args.index("--max-minutes") + 1]) * 60
    tex = {} if "--fake" in args or not hf_tex else build_texture_providers(hf_tex)
    w = Worker(supa, providers, remote_state=("--remote-state" in args and "--fake" not in args), tex_providers=tex)
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
