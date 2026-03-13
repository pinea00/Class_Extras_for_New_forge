#!/usr/bin/env python3
# Classic Extras Extension for Forge Classic
# Upscaling: models/ESRGAN + models/RealESRGAN + DAT
# Face Restoration: GFPGAN + CodeFormer

import os
import gc
import numpy as np
import gradio as gr
from PIL import Image

import torch
from modules import scripts, shared, devices, errors, modelloader
from modules.shared import cmd_opts, opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model
from modules import scripts_postprocessing

try:
    from modules.shared import hf_endpoint
except ImportError:
    hf_endpoint = "https://huggingface.co"

# ── Paths ──────────────────────────────────────────────────────────────────────

def get_models_path():
    try:
        from modules.paths import models_path
        return models_path
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

def _mp():
    return get_models_path()

ESRGAN_DIR    = os.path.join(_mp(), "ESRGAN")
REALESRGAN_DIR = os.path.join(_mp(), "RealESRGAN")
DAT_DIR       = os.path.join(_mp(), "DAT")
GFPGAN_DIR    = os.path.join(_mp(), "GFPGAN")
CODEFORMER_DIR = os.path.join(_mp(), "Codeformer")

for _d in [ESRGAN_DIR, REALESRGAN_DIR, DAT_DIR, GFPGAN_DIR, CODEFORMER_DIR]:
    os.makedirs(_d, exist_ok=True)


# ── DAT Model Integration ──────────────────────────────────────────────────────

class UpscalerDAT(Upscaler):
    def __init__(self, user_path):
        self.name = "DAT"
        self.user_path = user_path
        self.scalers = []
        super().__init__()

        for file in self.find_models(ext_filter=[".pt", ".pth"]):
            name = modelloader.friendly_name(file)
            scaler_data = UpscalerData(name, file, upscaler=self, scale=None)
            self.scalers.append(scaler_data)

        for model in get_dat_models(self):
            self.scalers.append(model)

    def do_upscale(self, img, path):
        try:
            local_path = self.load_model(path)
        except Exception:
            errors.report(f"Unable to load DAT model {path}", exc_info=True)
            return img

        model_descriptor = modelloader.load_spandrel_model(
            local_path,
            device=self.device,
            prefer_half=(not getattr(cmd_opts, "no_half", False) and not getattr(cmd_opts, "upcast_sampling", False)),
            expected_architecture="DAT",
        )
        tile_size = getattr(opts, "DAT_tile", 192)
        tile_overlap = getattr(opts, "DAT_tile_overlap", 8)
        return upscale_with_model(
            model_descriptor,
            img,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )

    def load_model(self, path):
        """Returns a local file path ready for loading."""
        for scaler in self.scalers:
            data_path = getattr(scaler, "data_path", None) or getattr(scaler, "path", None)
            if data_path != path:
                continue

            if os.path.isfile(data_path):
                return data_path

            if data_path.startswith("http"):
                download_dir = DAT_DIR
                local = modelloader.load_file_from_url(
                    data_path,
                    model_dir=download_dir,
                )
                if os.path.exists(local) and os.path.getsize(local) < 200:
                    local = modelloader.load_file_from_url(
                        data_path,
                        model_dir=download_dir,
                    )
                if not os.path.exists(local):
                    raise FileNotFoundError(f"DAT download failed: {local}")
                return local

        raise ValueError(f"Unable to find model info: {path}")

def get_dat_models(scaler):
    return [
        UpscalerData(
            name="DAT x2",
            path=f"{hf_endpoint}/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x2.pth",
            scale=2,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT x3",
            path=f"{hf_endpoint}/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x3.pth",
            scale=3,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT x4",
            path=f"{hf_endpoint}/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x4.pth",
            scale=4,
            upscaler=scaler,
        ),
    ]


# ── Upscaler registry ──────────────────────────────────────────────────────────

_upscaler_registry = {}

def _register_forge_upscalers():
    try:
        for scaler in shared.sd_upscalers:
            name = getattr(scaler, "name", None)
            if name and name not in ("None", ""):
                _upscaler_registry[name] = {
                    "type": "forge",
                    "scaler_obj": scaler,
                }
    except Exception as e:
        print(f"[Classic Extras] Forge upscalers error: {e}")

def _register_dir(directory, prefix=""):
    exts = (".pth", ".pt", ".safetensors")
    if not os.path.isdir(directory):
        return
    for f in os.listdir(directory):
        if f.lower().endswith(exts):
            stem = os.path.splitext(f)[0]
            name = f"{prefix}{stem}" if prefix else stem
            path = os.path.join(directory, f)
            if name not in _upscaler_registry:
                _upscaler_registry[name] = {"type": "spandrel", "path": path}

def _register_dat_upscalers():
    try:
        dat_instance = UpscalerDAT(user_path=DAT_DIR)
        for scaler in dat_instance.scalers:
            name = f"DAT/{scaler.name}"
            _upscaler_registry[name] = {
                "type": "dat",
                "scaler_obj": scaler,
                "upscaler_instance": dat_instance,
            }
    except Exception as e:
        print(f"[Classic Extras] DAT init failed: {e}")
        import traceback; traceback.print_exc()

def build_upscaler_registry():
    _upscaler_registry.clear()
    _register_forge_upscalers()
    _register_dir(ESRGAN_DIR,     prefix="")             
    _register_dir(REALESRGAN_DIR, prefix="RealESRGAN/")  
    _register_dat_upscalers()

def get_upscaler_names():
    if not _upscaler_registry:
        build_upscaler_registry()
    return ["None"] + sorted(_upscaler_registry.keys())


# ── Run upscaling ──────────────────────────────────────────────────────────────

def _run_forge_scaler(image, scaler_obj, upscale_by):
    upscaler = getattr(scaler_obj, "upscaler", None)
    data_path = getattr(scaler_obj, "data_path", None)

    if upscaler is not None and data_path is not None:
        result = upscaler.do_upscale(image, data_path)
    else:
        result = _run_spandrel(image, data_path, upscale_by)

    native_scale = getattr(scaler_obj, "scale", None)
    if result is not None and native_scale and native_scale != upscale_by:
        w, h = image.size
        result = result.resize((int(w * upscale_by), int(h * upscale_by)), Image.LANCZOS)

    return result

def _run_spandrel(image, model_path, upscale_by):
    try:
        model_descriptor = modelloader.load_spandrel_model(
            model_path,
            device=devices.device,
            prefer_half=(devices.device.type != "cpu"),
        )
        return upscale_with_model(model_descriptor, image, tile_size=192, tile_overlap=8)
    except Exception as e:
        print(f"[Classic Extras] Spandrel load failed for {model_path}: {e}")
        w, h = image.size
        return image.resize((int(w * upscale_by), int(h * upscale_by)), Image.LANCZOS)

def _run_dat(image, scaler_obj, upscaler_instance, upscale_by):
    result = upscaler_instance.do_upscale(image, scaler_obj.data_path)
    native = getattr(scaler_obj, 'scale', None)
    if native and upscale_by != native:
        w, h = image.size
        result = result.resize((int(w * upscale_by), int(h * upscale_by)), Image.LANCZOS)
    return result

def run_upscaler(image: Image.Image, upscaler_name: str, upscale_by: float) -> Image.Image:
    if upscaler_name == "None" or upscale_by <= 1.0:
        return image

    if not _upscaler_registry:
        build_upscaler_registry()

    info = _upscaler_registry.get(upscaler_name)
    if info is None:
        print(f"[Classic Extras] '{upscaler_name}' not in registry.")
        return image

    try:
        t = info["type"]
        if t == "forge":
            return _run_forge_scaler(image, info["scaler_obj"], upscale_by)
        elif t == "spandrel":
            return _run_spandrel(image, info["path"], upscale_by)
        elif t == "dat":
            return _run_dat(image, info["scaler_obj"],
                            info["upscaler_instance"], upscale_by)
        else:
            print(f"[Classic Extras] Unknown type: {t}")
            return image
    except Exception as e:
        print(f"[Classic Extras] Upscaling failed ({upscaler_name}): {e}")
        import traceback; traceback.print_exc()
        w, h = image.size
        return image.resize((int(w * upscale_by), int(h * upscale_by)), Image.LANCZOS)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Face Restoration (ONNX-based, no basicsr dependency) ──────────────────────

def _get_onnx_dir():
    ext_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(ext_root, "sd-webui-facefusion-dev", "models")

ONNX_FACE_MODELS = {
    "GFPGAN 1.4 (ONNX)":      "gfpgan_1.4.onnx",
    "CodeFormer (ONNX)":       "codeformer.onnx",
    "RestoreFormer (ONNX)":    "restoreformer.onnx",
}

def scan_face_models():
    models = ["None"]

    onnx_dir = _get_onnx_dir()
    for label, fname in ONNX_FACE_MODELS.items():
        if os.path.exists(os.path.join(onnx_dir, fname)):
            models.append(label)

    try:
        for restorer in shared.face_restorers:
            name = restorer.name()
            if name and name not in ("None", "") and name not in models:
                models.append(name)
    except Exception:
        pass

    return models

def run_face_restore(image: Image.Image, model_name: str, strength: float) -> Image.Image:
    if model_name == "None" or strength <= 0:
        return image

    img_np = np.array(image.convert("RGB"))

    if model_name in ONNX_FACE_MODELS:
        fname = ONNX_FACE_MODELS[model_name]
        onnx_path = os.path.join(_get_onnx_dir(), fname)
        return _run_onnx_face_restore(image, img_np, onnx_path, model_name, strength)

    try:
        for restorer in shared.face_restorers:
            if restorer.name() == model_name:
                result_np = restorer.restore(img_np)
                if result_np is None:
                    break
                result_np = np.clip(result_np, 0, 255).astype(np.uint8)
                if strength < 1.0:
                    result_np = (img_np * (1 - strength) + result_np * strength).astype(np.uint8)
                return Image.fromarray(result_np)
    except Exception as e:
        print(f"[Classic Extras] Built-in restorer failed ({model_name}): {e}")

    print(f"[Classic Extras] Face model '{model_name}' not found.")
    return image


# ── ONNX face restoration pipeline ────────────────────────────────────────────

_FFHQ_512_TEMPLATE = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19360],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
], dtype=np.float32)

_retinaface_session = None

def _get_retinaface_session():
    global _retinaface_session
    if _retinaface_session is not None:
        return _retinaface_session
    import onnxruntime
    ext_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(ext_root, "sd-webui-facefusion-dev", "models", "retinaface_10g.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"retinaface_10g.onnx not found at: {model_path}")
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    _retinaface_session = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
    return _retinaface_session

def _detect_faces_retinaface(frame_bgr: np.ndarray):
    import cv2
    session = _get_retinaface_session()
    h, w = frame_bgr.shape[:2]
    det_size = 640
    scale_h, scale_w = h / det_size, w / det_size
    resized = np.zeros((det_size, det_size, 3), dtype=np.float32)
    rh, rw = min(h, det_size), min(w, det_size)
    resized[:rh, :rw] = cv2.resize(frame_bgr, (rw, rh)).astype(np.float32)
    inp = ((resized - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    detections = session.run(None, {session.get_inputs()[0].name: inp})

    kps_list = []
    feature_strides = [8, 16, 32]
    feature_map_channel = 3
    anchor_total = 2

    for idx, stride in enumerate(feature_strides):
        scores = detections[idx]
        keep = np.where(scores >= 0.5)[0]
        if not keep.any():
            continue
        sh, sw = det_size // stride, det_size // stride
        y, x = np.mgrid[:sh, :sw][::-1]
        anchors = np.stack((y, x), axis=-1).reshape(-1, 2)
        anchors = np.stack([anchors] * anchor_total, axis=1).reshape(-1, 2)
        anchors = (anchors * stride).astype(np.float32)

        kps_raw = detections[idx + feature_map_channel * 2] * stride
        for i in keep:
            kps = anchors[i:i+1] + kps_raw[i].reshape(5, 2)
            kps *= [scale_w, scale_h]
            kps_list.append(kps)

    return kps_list

def _warp_face(frame_bgr: np.ndarray, kps: np.ndarray, size=(512, 512)):
    import cv2
    normed = _FFHQ_512_TEMPLATE * size[1] / size[0]
    affine_matrix = cv2.estimateAffinePartial2D(kps, normed, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    crop = cv2.warpAffine(frame_bgr, affine_matrix, (size[1], size[1]), borderMode=cv2.BORDER_REPLICATE)
    return crop, affine_matrix

def _paste_back(frame_bgr: np.ndarray, crop: np.ndarray, mask: np.ndarray, affine_matrix: np.ndarray):
    import cv2
    inv_matrix = cv2.invertAffineTransform(affine_matrix)
    frame_size = frame_bgr.shape[:2][::-1]
    inv_mask = cv2.warpAffine(mask, inv_matrix, frame_size).clip(0, 1)
    inv_crop = cv2.warpAffine(crop, inv_matrix, frame_size, borderMode=cv2.BORDER_REPLICATE)
    result = frame_bgr.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = inv_mask * inv_crop[:, :, c] + (1 - inv_mask) * frame_bgr[:, :, c]
    return result.astype(np.uint8)

def _make_box_mask(size=(512, 512), blur=0.3):
    import cv2
    blur_amount = int(size[0] * 0.5 * blur)
    blur_area = max(blur_amount // 2, 1)
    mask = np.ones(size, dtype=np.float32)
    mask[:blur_area, :] = 0
    mask[-blur_area:, :] = 0
    mask[:, :blur_area] = 0
    mask[:, -blur_area:] = 0
    if blur_amount > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
    return mask

def _prepare_crop(crop_bgr: np.ndarray) -> np.ndarray:
    inp = crop_bgr[:, :, ::-1] / 255.0
    inp = (inp - 0.5) / 0.5
    inp = np.expand_dims(inp.transpose(2, 0, 1), axis=0).astype(np.float32)
    return inp

def _normalize_crop(out: np.ndarray) -> np.ndarray:
    out = np.clip(out, -1, 1)
    out = (out + 1) / 2
    out = out.transpose(1, 2, 0)
    out = (out * 255.0).round().astype(np.uint8)
    out = out[:, :, ::-1]
    return out

def _run_onnx_face_restore(image: Image.Image, img_np: np.ndarray,
                            onnx_path: str, model_name: str, strength: float) -> Image.Image:
    import cv2
    import onnxruntime

    frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    try:
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = onnxruntime.InferenceSession(
            onnx_path, sess_opts=session_options, providers=["CPUExecutionProvider"])
    except TypeError:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    try:
        kps_list = _detect_faces_retinaface(frame_bgr)
        if not kps_list:
            kps_list = None

        result_frame = frame_bgr.copy()

        if kps_list:
            mask = _make_box_mask((512, 512), blur=0.3)
            for kps in kps_list:
                crop, affine_matrix = _warp_face(frame_bgr, kps, size=(512, 512))
                inp = _prepare_crop(crop)

                feeds = {}
                for model_input in session.get_inputs():
                    if model_input.name == "input":
                        feeds["input"] = inp
                    elif model_input.name == "weight":
                        feeds["weight"] = np.array([strength], dtype=np.double)

                out = session.run(None, feeds)[0][0]
                restored_crop = _normalize_crop(out)

                paste_frame = _paste_back(result_frame, restored_crop, mask, affine_matrix)

                result_frame = cv2.addWeighted(
                    result_frame.astype(np.float32), 1 - strength,
                    paste_frame.astype(np.float32), strength, 0
                ).astype(np.uint8)
        else:
            h, w = frame_bgr.shape[:2]
            resized = cv2.resize(frame_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            inp = _prepare_crop(resized)
            feeds = {"input": inp}
            for model_input in session.get_inputs():
                if model_input.name == "weight":
                    feeds["weight"] = np.array([strength], dtype=np.double)
            out = session.run(None, feeds)[0][0]
            restored = _normalize_crop(out)
            restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_LANCZOS4)
            result_frame = cv2.addWeighted(
                frame_bgr.astype(np.float32), 1 - strength,
                restored.astype(np.float32), strength, 0
            ).astype(np.uint8)

        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

    except Exception as e:
        print(f"[Classic Extras] ONNX face restore failed ({model_name}): {e}")
        import traceback; traceback.print_exc()
        return image
    finally:
        gc.collect()


# ── Postprocessing Script Integration ──────────────────────────────────────────

class ScriptPostprocessingClassicExtras(scripts_postprocessing.ScriptPostprocessing):
    name = "Classic Extras (Upscale & Face Restore)"
    order = 15000 

    def ui(self):
        build_upscaler_registry()
        
        with gr.Accordion("Classic Extras", open=False):
            gr.HTML("<b>🔍 Upscaling</b>")
            with gr.Row():
                upscaler_name = gr.Dropdown(
                    label="Upscaler",
                    choices=get_upscaler_names(),
                    value="None"
                )
            upscale_by = gr.Slider(
                label="Upscale by",
                minimum=1.0, maximum=8.0, step=0.05, value=4.0
            )

            gr.HTML("<br><b>👤 Face Restoration</b>")
            with gr.Row():
                face_model = gr.Dropdown(
                    label="Restore faces",
                    choices=scan_face_models(),
                    value="None"
                )
            face_strength = gr.Slider(
                label="Strength",
                minimum=0.0, maximum=1.0, step=0.05, value=0.8
            )
            restore_before = gr.Checkbox(
                label="Restore faces before upscaling", value=False
            )

        return {
            "upscaler_name": upscaler_name,
            "upscale_by": upscale_by,
            "face_model": face_model,
            "face_strength": face_strength,
            "restore_before": restore_before,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, upscaler_name, upscale_by, face_model, face_strength, restore_before):
        if upscaler_name == "None" and face_model == "None":
            return

        img = pp.image.convert("RGB") if pp.image.mode != "RGB" else pp.image.copy()

        if restore_before and face_model != "None":
            img = run_face_restore(img, face_model, face_strength)

        if upscaler_name != "None" and upscale_by > 1.0:
            img = run_upscaler(img, upscaler_name, upscale_by)

        if not restore_before and face_model != "None":
            img = run_face_restore(img, face_model, face_strength)

        pp.image = img
