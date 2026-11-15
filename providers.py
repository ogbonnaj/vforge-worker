"""Image-to-3D providers (free Hugging Face Spaces) for the VFORGE worker.

Every provider has generate(image_path, seed, workdir) -> path of a model file.
It raises QuotaError when the free GPU quota is used up (worker switches provider and
remembers when to come back) and ProviderError for any other failure.

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


class SpaceProvider:
    name = "space"
    space = ""

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
            src = result if isinstance(result, str) and os.path.exists(result) else _first_model_file(result)
            if not src:
                raise ProviderError("Space returned no model file")
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


class Hunyuan3D(SpaceProvider):
    name = "hunyuan3d"
    space = "tencent/Hunyuan3D-2"

    def run(self, image_path, seed):
        from gradio_client import handle_file
        return self.client().predict(
            image=handle_file(image_path), steps=30, guidance_scale=5.0, seed=seed,
            octree_resolution=256, check_box_rembg=True, num_chunks=8000,
            randomize_seed=False, api_name="/shape_generation")


class Trellis(SpaceProvider):
    name = "trellis"
    space = "trellis-community/TRELLIS"      # was JeffreyXiang/TRELLIS (CONFIG_ERROR on 15 Nov 2026); community copy was running on 9 Nov

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


# ============================ TEXTURING (separate from shape generation) ============================
# Texture Spaces take a finished white model (+ the reference image) and return a textured GLB.
# They use their OWN Hugging Face token (HF_TEXTURE_TOKEN), so the shape providers' GPU quota is never touched.
import zlib

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def _glb_json(path):
    """Return the JSON part of a .glb as a dict, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"glTF":
                return None
            ln, kind = struct.unpack("<I4s", f.read(8))
            if kind != b"JSON":
                return None
            return json.loads(f.read(ln).decode("utf-8", "replace"))
    except (OSError, ValueError, struct.error):
        return None


def _is_textured(path, allow_vertex_colors=False):
    """True when the .glb really carries a texture image used by a material (or vertex colours if allowed)."""
    g = _glb_json(path)
    if not g:
        return False
    has_img = bool(g.get("images")) and bool(g.get("textures"))
    used = False
    for m in g.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        if "baseColorTexture" in pbr or "emissiveTexture" in m:
            used = True
    if has_img and used:
        return True
    if allow_vertex_colors:
        for me in g.get("meshes", []):
            for pr in me.get("primitives", []):
                if "COLOR_0" in pr.get("attributes", {}):
                    return True
    return False


def _paths_from(obj, exts):
    """All existing file paths with one of `exts` inside whatever a Space returned (keeps order)."""
    out = []
    if isinstance(obj, str):
        if obj.lower().endswith(exts) and os.path.exists(obj):
            out.append(obj)
    elif isinstance(obj, dict):
        for k in ("image", "path", "value", "name", "video"):
            if k in obj:
                out.extend(_paths_from(obj[k], exts))
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            out.extend(_paths_from(o, exts))
    return out


def _endpoints(client):
    """The Space's named endpoints as {name: [params]} read from its own API description."""
    info = client.view_api(return_format="dict", print_info=False)
    res = {}
    for name, ep in (info.get("named_endpoints") or {}).items():
        res[name] = (ep.get("parameters") if isinstance(ep, dict) else ep) or []
    return res


def _kind(p):
    """What a Space parameter is for: mesh, image, gallery, negative, prompt, seed, uvsize, random, other."""
    text = ("%s %s" % (p.get("parameter_name", ""), p.get("label", ""))).lower()
    comp = str(p.get("component", "")).lower()
    if "negative" in text:
        return "negative"
    if comp == "model3d" or "mesh" in text or "3d model" in text or "glb" in text:
        return "mesh"
    if comp == "gallery":
        return "gallery"
    if comp == "image" or "image" in text or "reference" in text:
        return "image"
    if "prompt" in text or "description" in text:
        return "prompt"
    if "random" in text:
        return "random"
    if "seed" in text:
        return "seed"
    if "uv" in text and "size" in text:
        return "uvsize"
    return "other"


def _kinds(params):
    return [_kind(p) for p in params]


def _build_kwargs(params, values):
    """Map our values {mesh,image,gallery,prompt,negative,seed,uvsize} onto a Space's parameters by name.
    Parameters we do not know are left out when they have a default; otherwise it is an error."""
    kw = {}
    for p in params:
        k = _kind(p)
        name = p.get("parameter_name") or p.get("label")
        if k in values and values[k] is not None:
            kw[name] = values[k]
        elif k == "random":
            kw[name] = False
        elif not p.get("parameter_has_default", False):
            raise ProviderError("this Space needs a value for '%s' that VFORGE does not know; run --check" % name)
    return kw


def _png_1x1(rgb=(200, 120, 60)):
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = b"\x00" + bytes(rgb)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def tiny_textured_glb(path):
    """A valid minimal GLB: one triangle with UVs and a 1x1 texture. Used by the fake texture provider."""
    pos = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    uv = struct.pack("<6f", 0, 0, 1, 0, 0, 1)
    png = _png_1x1()
    png += b"\0" * (-len(png) % 4)
    bin_ = pos + uv + png
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0}]}],
          "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
          "textures": [{"source": 0}], "images": [{"bufferView": 2, "mimeType": "image/png"}],
          "buffers": [{"byteLength": len(bin_)}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
                          {"buffer": 0, "byteOffset": 36, "byteLength": 24, "target": 34962},
                          {"buffer": 0, "byteOffset": 60, "byteLength": len(png)}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                         "min": [0, 0, 0], "max": [1, 1, 0]},
                        {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"}]}
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    total = 12 + 8 + len(j) + 8 + len(bin_)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(j), b"JSON") + j)
        f.write(struct.pack("<I4s", len(bin_), b"BIN\0") + bin_)


class TextureProvider(SpaceProvider):
    """Base for texture Spaces. texture() returns the path of a textured .glb or raises ProviderError/QuotaError."""
    allow_vertex_colors = False
    default_prompt = "high quality, detailed, realistic texture"

    def run_texture(self, mesh_path, image_path, seed, workdir, prompt, negative):
        raise NotImplementedError

    def texture(self, mesh_path, image_path, seed, workdir, prompt="", negative=""):
        try:
            res = self.run_texture(mesh_path, image_path, seed, workdir, prompt or self.default_prompt, negative or "")
            src = res if isinstance(res, str) and os.path.exists(res) else None
            if not src:
                found = _paths_from(res, (".glb",))
                src = found[0] if found else None
            if not src:
                raise ProviderError("Space returned no .glb file (it may return another format)")
            dest = os.path.join(workdir, "%s_%d_tex.glb" % (self.name, seed))
            shutil.copyfile(src, dest)
            return dest
        except ProviderError:
            raise
        except Exception as e:
            raise classify(e)


class MVAdapterImg2Texture(TextureProvider):
    """VAST-AI/MV-Adapter-Img2Texture: white mesh + reference image -> textured GLB.
    Two calls inside the Space (make 6 views, then bake them onto the mesh). Endpoints are found from the Space's
    own API description, not hard-coded, because the Space's author may rename them."""
    name = "mvadapter_img2tex"
    space = "VAST-AI/MV-Adapter-Img2Texture"
    UV_SIZE = 2048

    def run_texture(self, mesh_path, image_path, seed, workdir, prompt, negative):
        from gradio_client import handle_file
        c = self.client()
        eps = _endpoints(c)
        views_ep = bake_ep = None
        for n, params in eps.items():
            ks = _kinds(params)
            if any(w in n.lower() for w in ("example", "session", "random")):
                continue
            if "mesh" in ks and "gallery" in ks:
                bake_ep = bake_ep or n
            elif "mesh" in ks and "image" in ks:
                views_ep = views_ep or n
        if not views_ep or not bake_ep:
            raise ProviderError("MV-Adapter Space API is different from expected (views=%s, bake=%s); run --check"
                                % (views_ep, bake_ep))
        if "/start_session" in eps:
            try:
                c.predict(api_name="/start_session")
            except Exception:
                pass
        vals = {"mesh": handle_file(mesh_path), "image": handle_file(image_path), "prompt": prompt,
                "negative": negative or None, "seed": seed, "uvsize": self.UV_SIZE}
        views = c.predict(api_name=views_ep, **_build_kwargs(eps[views_ep], vals))
        pics = _paths_from(views, IMG_EXT)
        if not pics:
            raise ProviderError("the Space made no view images")
        gal = [handle_file(p) for p in pics]
        gp = [p for p in eps[bake_ep] if _kind(p) == "gallery"][0]
        if "caption" in json.dumps(gp.get("python_type", "")).lower():
            gal = [{"image": g, "caption": None} for g in gal]
        vals2 = dict(vals, gallery=gal)
        return c.predict(api_name=bake_ep, **_build_kwargs(eps[bake_ep], vals2))


class EmbodiedGenTexture(TextureProvider):
    """HorizonRobotics/EmbodiedGen-Texture-Gen: white mesh + text prompt (+ reference image) -> textured GLB.
    Real flow seen in the Space's own API list (4 calls): make condition images from the mesh, make texture images
    from the prompt, pick one image, bake it onto the mesh."""
    name = "embodiedgen_tex"
    space = "HorizonRobotics/EmbodiedGen-Texture-Gen"
    STEPS = 30          # Space default is 50; fewer steps = less of the small daily GPU quota
    TEXTURE_SIZE = 2048

    def run_texture(self, mesh_path, image_path, seed, workdir, prompt, negative):
        from gradio_client import handle_file
        c = self.client()
        eps = _endpoints(c)
        need = ("/generate_condition", "/generate_texture_mvimages", "/get_selected_image", "/backproject_texture_v2")
        missing = [n for n in need if n not in eps]
        if missing:
            raise ProviderError("EmbodiedGen Space API changed (missing %s); run --check" % ", ".join(missing))
        if "/start_session" in eps:
            try:
                c.predict(api_name="/start_session")
            except Exception:
                pass
        c.predict(mesh_path=handle_file(mesh_path), api_name="/generate_condition")
        imgs = c.predict(prompt=prompt, num_inference_steps=self.STEPS, seed=seed,
                         ip_img_path=handle_file(image_path), api_name="/generate_texture_mvimages")
        cands = [(_paths_from(x, IMG_EXT) or [None])[0] for x in list(imgs or [])[:3]]
        if len(cands) < 3 or None in cands:
            raise ProviderError("the Space made no candidate texture images")
        sel = c.predict(choice="sample1", sample1=handle_file(cands[0]), sample2=handle_file(cands[1]),
                        sample3=handle_file(cands[2]), api_name="/get_selected_image")
        chosen = _paths_from(sel, IMG_EXT)
        if not chosen:
            raise ProviderError("the Space returned no selected image")
        return c.predict(mesh_path=handle_file(mesh_path), input_image=handle_file(chosen[0]),
                         texture_size=self.TEXTURE_SIZE, api_name="/backproject_texture_v2")


class TripoSGTexture(TextureProvider):
    """VAST-AI/TripoSG /run_texture(image, mesh_path, seed) -> textured model. Seen live in the check log (15 Nov 2026).
    Parameters are mapped by name from the Space's own API description, like the other texture adapters."""
    name = "triposg_tex"
    space = "VAST-AI/TripoSG"

    def run_texture(self, mesh_path, image_path, seed, workdir, prompt, negative):
        from gradio_client import handle_file
        c = self.client()
        eps = _endpoints(c)
        if "/run_texture" not in eps:
            raise ProviderError("TripoSG Space API changed (no /run_texture); run --check")
        if "/start_session" in eps:
            try:
                c.predict(api_name="/start_session")
            except Exception:
                pass
        vals = {"mesh": handle_file(mesh_path), "image": handle_file(image_path), "seed": seed}
        return c.predict(api_name="/run_texture", **_build_kwargs(eps["/run_texture"], vals))


class FakeTexture:
    """Test texture provider: a tiny textured triangle, no GPU, no internet. Run main.py --texture --fake-texture."""
    name = "fake_texture"
    space = "(none)"
    allow_vertex_colors = False

    def texture(self, mesh_path, image_path, seed, workdir, prompt="", negative=""):
        dest = os.path.join(workdir, "fake_%d_tex.glb" % seed)
        tiny_textured_glb(dest)
        return dest


TEXTURE_ORDER = ["embodiedgen_tex", "triposg_tex", "mvadapter_img2tex"]   # MV-Adapter Space was in RUNTIME_ERROR on 15 Nov 2026


def build_texture_providers(hf_token=""):
    """hf_token must be the TEXTURING account's token, never the shape account's."""
    return {"mvadapter_img2tex": MVAdapterImg2Texture(hf_token), "embodiedgen_tex": EmbodiedGenTexture(hf_token),
            "triposg_tex": TripoSGTexture(hf_token)}


# ---- candidate replacement Spaces: only LOOKED AT by main.py --check (no GPU used, never used for jobs) ----
CANDIDATE_SPACES = {
    "stable_fast_3d": "stabilityai/stable-fast-3d",
    "instantmesh": "TencentARC/InstantMesh",
    "hunyuan3d_21": "tencent/Hunyuan3D-2.1",
    "triposg": "VAST-AI/TripoSG",
    "sparc3d": "ilcve21/Sparc3D",
    "hi3dgen": "Stable-X/Hi3DGen",
}


class CandidateSpace(SpaceProvider):
    def __init__(self, name, space, hf_token=""):
        super().__init__(hf_token)
        self.name, self.space = name, space


def build_candidates(hf_token=""):
    return {n: CandidateSpace(n, sp, hf_token) for n, sp in CANDIDATE_SPACES.items()}


def api_summary(client):
    """Short one-line-per-endpoint API summary: name(param, param*) -> returns.  * = required."""
    info = client.view_api(return_format="dict", print_info=False)
    lines = []
    for name, ep in (info.get("named_endpoints") or {}).items():
        params = ep.get("parameters") if isinstance(ep, dict) else ep
        rets = ep.get("returns", []) if isinstance(ep, dict) else []
        ps = ", ".join("%s%s:%s" % (p.get("parameter_name") or p.get("label"),
                                   "" if p.get("parameter_has_default") else "*",
                                   str(p.get("component", "?")).lower()) for p in (params or []))
        rs = ", ".join(str(r.get("component", "?")).lower() for r in rets)
        if not params and not rets:
            continue                      # empty UI helper endpoints (lambda...) are noise
        lines.append("  %s(%s) -> %s" % (name, ps, rs))
    return lines


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
    return {"hunyuan3d": Hunyuan3D(hf_token), "trellis": Trellis(hf_token), "triposr": TripoSR(hf_token)}


# Order tried per quality setting. High-value images start with the best model.
ORDER = {"high": ["hunyuan3d", "trellis", "triposr"], "low": ["triposr", "trellis", "hunyuan3d"]}
