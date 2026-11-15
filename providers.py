"""Image-to-3D providers (free Hugging Face Spaces) for the VFORGE worker.

Every provider has generate(image_path, seed, workdir) -> path of a model file.
It raises QuotaError when the free GPU quota is used up (worker switches provider and
remembers when to come back) and ProviderError for any other failure.

TEXTURE RULE: when WANT_TEXTURE is True a model is only accepted if it is textured. A white/untextured result raises
ProviderError so the worker moves on to the next provider.

NOTE: Space APIs belong to other people and change. These adapters were written from the
Spaces' published APIs and could not be tested live while writing. Run
`python backend/worker/main.py --check` to see each Space's real API.
"""
import os
import re
import shutil
import struct
import json

GLB_EXT = (".glb", ".gltf", ".obj", ".ply")

# False = shape only (white mesh): cheap, about 60 s of free GPU per model.
# True  = shape + texture in one call: about 3 min of GPU, so only ~1 model per day on a free account.
WANT_TEXTURE = False


class ProviderError(Exception):
    pass


class QuotaError(ProviderError):
    def __init__(self, msg, wait_seconds=3600):
        super().__init__(msg)
        self.wait_seconds = wait_seconds


QUOTA_WORDS = ("quota", "zerogpu", "exceeded your", "gpu limit", "daily limit", "too many requests", "429")


def classify(exc):
    """Turn any exception from a Space into QuotaError or ProviderError."""
    text = str(exc)
    low = text.lower()
    if any(w in low for w in QUOTA_WORDS):
        wait = 3600
        m = re.search(r"(\d+):(\d{2}):(\d{2})", text)      # ZeroGPU says "Try again in 0:13:41"
        if m:
            wait = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + 30
        return QuotaError(text[:300], wait)
    return ProviderError(text[:300])


def _first_model_file(obj):
    """Find the first model file path inside whatever a Space returned."""
    if isinstance(obj, str):
        if obj.lower().endswith(GLB_EXT) and os.path.exists(obj):
            return obj
        return None
    if isinstance(obj, dict):
        for k in ("path", "value", "name"):
            if k in obj:
                r = _first_model_file(obj[k])
                if r:
                    return r
        return None
    if isinstance(obj, (list, tuple)):
        for o in obj:
            r = _first_model_file(o)
            if r:
                return r
    return None


def _valid_model(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) < 500:
        return False
    if path.lower().endswith(".glb"):
        with open(path, "rb") as f:
            return f.read(4) == b"glTF"
    return True


def _all_model_files(obj):
    """Every model file path inside whatever a Space returned (white AND textured ones)."""
    found = []
    if isinstance(obj, str):
        if obj.lower().endswith(GLB_EXT) and os.path.exists(obj):
            found.append(obj)
    elif isinstance(obj, dict):
        for k in ("path", "value", "name"):
            if k in obj:
                found += _all_model_files(obj[k])
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            found += _all_model_files(o)
    return found


def _glb_json(path):
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"glTF":
            return None
        clen, ctype = struct.unpack("<I4s", f.read(8))
        if ctype != b"JSON":
            return None
        return json.loads(f.read(clen).decode("utf-8"))


def _is_textured(path, allow_vertex_colors=False):
    """True if the GLB has a real texture image used by a material.
    With allow_vertex_colors, per-vertex colours (COLOR_0) also count."""
    if not path or not path.lower().endswith(".glb"):
        return False
    try:
        js = _glb_json(path)
    except Exception:
        return False
    if not js:
        return False
    if js.get("images") and js.get("textures"):
        for m in js.get("materials", []):
            if "baseColorTexture" in m.get("pbrMetallicRoughness", {}) or "emissiveTexture" in m:
                return True
    if allow_vertex_colors:
        for mesh in js.get("meshes", []):
            for prim in mesh.get("primitives", []):
                if "COLOR_0" in prim.get("attributes", {}):
                    return True
    return False


class SpaceProvider:
    name = "space"
    space = ""
    allow_vertex_colors = False      # True = per-vertex colour counts as textured

    def __init__(self, hf_token=""):
        self.hf_token = hf_token
        self._client = None

    def client(self):
        if self._client is None:
            try:
                from gradio_client import Client
            except ImportError:
                raise ProviderError("gradio_client is not installed (pip install gradio_client)")
            try:
                try:
                    self._client = Client(self.space, token=self.hf_token or None, verbose=False)
                except TypeError:
                    self._client = Client(self.space, hf_token=self.hf_token or None, verbose=False)
            except Exception as e:
                raise classify(e)
        return self._client

    def run(self, image_path, seed):
        raise NotImplementedError

    def generate(self, image_path, seed, workdir):
        try:
            result = self.run(image_path, seed)
            files = _all_model_files(result)
            if not files:
                raise ProviderError("Space returned no model file")
            src = None
            if WANT_TEXTURE:
                for f in files:                  # pick the first file that really has texture
                    if _is_textured(f, self.allow_vertex_colors):
                        src = f
                        break
                if not src:
                    raise ProviderError("%s returned a model with NO texture (white). Trying next provider." % self.name)
            else:
                src = files[0]                   # shape-only mode: any valid model is fine
            ext = os.path.splitext(src)[1].lower() or ".glb"
            dest = os.path.join(workdir, "%s_%d%s" % (self.name, seed, ext))
            shutil.copyfile(src, dest)
            if not _valid_model(dest):
                raise ProviderError("Space returned an empty or broken model file")
            return dest
        except ProviderError:
            raise
        except Exception as e:
            raise classify(e)


def _hy_api():
    return "/generation_all" if WANT_TEXTURE else "/shape_generation"


class Hunyuan3D(SpaceProvider):
    name = "hunyuan3d"
    space = "tencent/Hunyuan3D-2"

    def run(self, image_path, seed):
        from gradio_client import handle_file
        return self.client().predict(
            image=handle_file(image_path), steps=30, guidance_scale=5.0, seed=seed,
            octree_resolution=256, check_box_rembg=True, num_chunks=8000,
            randomize_seed=False, api_name=_hy_api())


class Hunyuan3D21(SpaceProvider):
    """Newer official Hunyuan3D-2.1 Space (shape + PBR texture). Same /generation_all API shape as 2.0:
    returns (white mesh, textured glb, html, stats, seed). The base class picks the textured file."""
    name = "hunyuan3d21"
    space = "tencent/Hunyuan3D-2.1"

    def run(self, image_path, seed):
        from gradio_client import handle_file
        return self.client().predict(
            image=handle_file(image_path), steps=30, guidance_scale=5.0, seed=seed,
            octree_resolution=256, check_box_rembg=True,
            randomize_seed=False, api_name=_hy_api())


class Trellis(SpaceProvider):
    name = "trellis"
    space = "JeffreyXiang/TRELLIS"

    def run(self, image_path, seed):
        from gradio_client import handle_file
        c = self.client()
        try:
            c.predict(api_name="/start_session")
        except Exception:
            pass                                    # not every version has it
        pre = c.predict(image=handle_file(image_path), api_name="/preprocess_image")
        c.predict(image=handle_file(pre), multiimages=[], is_multiimage=False, seed=seed,
                  ss_guidance_strength=7.5, ss_sampling_steps=12, slat_guidance_strength=3,
                  slat_sampling_steps=12, multiimage_algo="stochastic", api_name="/image_to_3d")
        return c.predict(mesh_simplify=0.95, texture_size=1024, api_name="/extract_glb")


class TripoSR(SpaceProvider):
    name = "triposr"
    space = "stabilityai/TripoSR"
    allow_vertex_colors = True       # TripoSR has no UV texture, only vertex colours

    def run(self, image_path, seed):
        from gradio_client import handle_file
        c = self.client()
        pre = c.predict(input_image=handle_file(image_path), do_remove_background=True,
                        foreground_ratio=0.85, api_name="/preprocess")
        res = c.predict(image=handle_file(pre), mc_resolution=256, api_name="/generate")
        # returns (obj, glb); prefer the glb
        if isinstance(res, (list, tuple)):
            for r in reversed(res):
                if isinstance(r, str) and r.lower().endswith(".glb"):
                    return r
        return res


def tiny_glb(path):
    """A valid minimal GLB (one triangle). Used by the 'fake' provider."""
    pos = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
          "buffers": [{"byteLength": len(pos)}],
          "bufferViews": [{"buffer": 0, "byteLength": len(pos), "target": 34962}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                         "min": [0, 0, 0], "max": [1, 1, 0]}]}
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    b = pos + b"\0" * (-len(pos) % 4)
    total = 12 + 8 + len(j) + 8 + len(b)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(j), b"JSON") + j)
        f.write(struct.pack("<I4s", len(b), b"BIN\0") + b)


class FakeProvider:
    """Test provider: makes a tiny triangle model, uses no GPU, no internet. Run main.py --fake."""
    name = "fake"
    space = "(none)"

    def generate(self, image_path, seed, workdir):
        dest = os.path.join(workdir, "fake_%d.glb" % seed)
        tiny_glb(dest)
        return dest


def build_providers(hf_token=""):
    return {"hunyuan3d21": Hunyuan3D21(hf_token), "hunyuan3d": Hunyuan3D(hf_token),
            "trellis": Trellis(hf_token), "triposr": TripoSR(hf_token)}


# Order tried per quality setting.
if WANT_TEXTURE:
    # hunyuan3d (2.0) is left out: its texture module is broken, so it spends GPU on the shape then fails.
    ORDER = {"high": ["hunyuan3d21", "trellis", "triposr"],
             "low": ["hunyuan3d21", "trellis", "triposr"]}
else:
    # shape only: 2.0 first (it worked for you), then 2.1. trellis/triposr are dead Spaces right now but fail instantly.
    ORDER = {"high": ["hunyuan3d", "hunyuan3d21", "trellis", "triposr"],
             "low": ["hunyuan3d", "hunyuan3d21", "trellis", "triposr"]}
    
