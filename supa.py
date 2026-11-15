"""Tiny Supabase client for the VFORGE worker. Standard library only (no pip needed)."""
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone


class SupaError(Exception):
    def __init__(self, msg, status=0):
        super().__init__(msg)
        self.status = status


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class Supa:
    def __init__(self, url, key, bucket="vforge", timeout=60):
        self.url = url.rstrip("/")
        self.key = key
        self.bucket = bucket
        self.timeout = timeout

    def _req(self, method, path, body=None, headers=None, raw=False):
        h = {"apikey": self.key}
        if self.key.startswith("eyJ"):      # old-style JWT service_role key; new sb_secret_ keys go on apikey only
            h["Authorization"] = "Bearer " + self.key
        if headers:
            h.update(headers)
        req = urllib.request.Request(self.url + path, data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode("utf-8", "replace")
            raise SupaError("HTTP %d: %s" % (e.code, msg), e.code)
        except (urllib.error.URLError, OSError) as e:
            raise SupaError("network: %s" % e, 0)
        return data if raw else (json.loads(data) if data else None)

    @staticmethod
    def _enc(path):
        return "/".join(urllib.parse.quote(p, safe="") for p in path.split("/"))

    # ---- jobs table ----
    def queued_jobs(self, limit=20):
        q = "status=eq.queued&order=created_at.asc&limit=%d" % limit
        return self._req("GET", "/rest/v1/jobs?" + q) or []

    def get_job(self, job_id):
        rows = self._req("GET", "/rest/v1/jobs?id=eq.%s&limit=1" % urllib.parse.quote(job_id, safe="-")) or []
        return rows[0] if rows else None

    def latest_done_job(self):
        rows = self._req("GET", "/rest/v1/jobs?status=eq.done&order=updated_at.desc&limit=1") or []
        return rows[0] if rows else None

    def count_processing(self):
        return self._req("GET", "/rest/v1/jobs?status=eq.processing&select=id") or []

    def claim(self, job_id):
        """queued -> processing. Returns the row, or None if someone else got it first."""
        rows = self._req(
            "PATCH", "/rest/v1/jobs?id=eq.%s&status=eq.queued" % job_id,
            json.dumps({"status": "processing", "updated_at": now_iso()}).encode(),
            {"Content-Type": "application/json", "Prefer": "return=representation"})
        return rows[0] if rows else None

    def patch(self, job_id, fields):
        fields = dict(fields)
        fields["updated_at"] = now_iso()
        self._req("PATCH", "/rest/v1/jobs?id=eq.%s" % job_id, json.dumps(fields).encode(),
                  {"Content-Type": "application/json"})

    def reset_processing(self, max_attempts=20):
        """At start-up: jobs a dead/killed worker left 'processing' go back to 'queued'.
        Each such rescue counts as one attempt, so a job that keeps killing the worker ends as 'failed'."""
        rows = self._req("GET", "/rest/v1/jobs?status=eq.processing&select=id,attempts") or []
        for r in rows:
            a = int(r.get("attempts") or 0) + 1
            if a >= max_attempts:
                self.patch(r["id"], {"status": "failed", "attempts": a,
                                     "error": "worker stopped or timed out too many times on this job"})
            else:
                self.patch(r["id"], {"status": "queued", "attempts": a})
        return len(rows)

    # ---- worker memory (quota rest times) kept in the bucket so it survives fresh cloud machines ----
    STATE_PATH = "worker/state.json"

    def get_state(self):
        try:
            raw = self._req("GET", "/storage/v1/object/authenticated/%s/%s" % (self.bucket, self._enc(self.STATE_PATH)),
                            raw=True)
        except SupaError as e:
            if e.status in (400, 404):
                return {}
            raise
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except ValueError:
            return {}

    def put_state(self, state):
        self._req("POST", "/storage/v1/object/%s/%s" % (self.bucket, self._enc(self.STATE_PATH)),
                  json.dumps(state).encode(), {"Content-Type": "application/json", "x-upsert": "true"})

    # ---- storage ----
    def download(self, obj_path, dest):
        data = self._req("GET", "/storage/v1/object/authenticated/%s/%s" % (self.bucket, self._enc(obj_path)),
                         raw=True)
        with open(dest, "wb") as f:
            f.write(data)

    def upload(self, obj_path, file_path, content_type="model/gltf-binary"):
        with open(file_path, "rb") as f:
            data = f.read()
        self._req("POST", "/storage/v1/object/%s/%s" % (self.bucket, self._enc(obj_path)), data,
                  {"Content-Type": content_type, "x-upsert": "true"})
