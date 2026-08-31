"""ImageNet ResNet18 classifier with Integrated Gradients explanations."""
from __future__ import annotations

import base64
import io
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter
from torchvision.models import ResNet18_Weights, resnet18

app = FastAPI(title="FocusLens")
app.mount("/static", StaticFiles(directory="."), name="static")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
LABELS = WEIGHTS.meta["categories"]


@lru_cache(maxsize=1)
def get_model():
    model = resnet18(weights=WEIGHTS).to(DEVICE).eval()
    return model


def integrated_gradients(model: torch.nn.Module, input_tensor: torch.Tensor,
                         baseline_tensor: torch.Tensor, target: int,
                         n_steps: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Integrated Gradients directly with PyTorch autograd.

    IG(x) = (x - baseline) * integral_0^1 grad F(baseline + alpha*(x-baseline)) d alpha.
    The integral is approximated with a uniform Riemann sum, and the returned
    convergence delta is the completeness error for the selected logit.
    """
    delta = input_tensor - baseline_tensor
    total_grad = torch.zeros_like(input_tensor)
    for step in range(1, n_steps + 1):
        alpha = float(step) / n_steps
        interpolated = (baseline_tensor + alpha * delta).detach().requires_grad_(True)
        logits = model(interpolated)
        target_score = logits[:, target].sum()
        gradient = torch.autograd.grad(target_score, interpolated, retain_graph=False)[0]
        total_grad += gradient
    attributions = delta * (total_grad / n_steps)
    with torch.no_grad():
        input_score = model(input_tensor)[:, target]
        baseline_score = model(baseline_tensor)[:, target]
        convergence_delta = attributions.sum() - (input_score - baseline_score).sum()
    return attributions, convergence_delta


def preprocess(image: Image.Image) -> torch.Tensor:
    # Equivalent to the standard ImageNet resize + center crop preprocessing.
    image = image.resize((256, 256), Image.Resampling.BILINEAR)
    left = (image.width - 224) // 2
    top = (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0).to(DEVICE)


def data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def baseline_for(kind: str, original: Image.Image, input_tensor: torch.Tensor) -> torch.Tensor:
    if kind == "black":
        source = Image.new("RGB", (224, 224), (0, 0, 0))
        return preprocess(source.resize((256, 256)))
    if kind == "white":
        source = Image.new("RGB", (224, 224), (255, 255, 255))
        return preprocess(source.resize((256, 256)))
    if kind == "blurred":
        # Blur after the same crop so the reference remains spatially compatible.
        source = original.resize((256, 256), Image.Resampling.BILINEAR)
        source = source.crop((16, 16, 240, 240)).filter(ImageFilter.GaussianBlur(35))
        return preprocess(source.resize((256, 256)))
    # "neutral" represents ImageNet's normalized zero / dataset mean image.
    return torch.zeros_like(input_tensor)


def baseline_preview(kind: str, original: Image.Image) -> Image.Image:
    """Return the human-readable reference image used by each IG path."""
    if kind == "black":
        return Image.new("RGB", (224, 224), (0, 0, 0))
    if kind == "white":
        return Image.new("RGB", (224, 224), (255, 255, 255))
    if kind == "blurred":
        return original.resize((224, 224), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(35))
    # Normalized zero corresponds approximately to ImageNet's mean RGB color.
    return Image.new("RGB", (224, 224), (124, 116, 103))


def heatmap_overlay(image: Image.Image, attribution: torch.Tensor, scale: float) -> tuple[str, float]:
    base = image.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(base, dtype=np.float32)
    values = attribution.squeeze(0).detach().cpu().numpy()
    signed = values.sum(axis=0)
    magnitude = np.abs(values).mean(axis=0)
    # A signed map preserves both supporting (orange) and opposing (blue) evidence.
    signed = np.clip(signed / scale, -1, 1)
    # Suppress pixel-level speckle and increase the contrast of the regions.
    encoded = ((signed + 1.0) * 127.5).astype(np.uint8)
    signed = np.asarray(Image.fromarray(encoded, mode="L").filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32) / 127.5 - 1.0
    alpha = np.clip(np.abs(signed) * 2.0, 0, 1) * 0.82
    heat = np.zeros((224, 224, 3), dtype=np.float32)
    positive = signed >= 0
    heat[positive] = np.array([50, 255, 190])   # supporting evidence: cyan/green
    heat[~positive] = np.array([205, 65, 255])  # opposing evidence: violet
    arr = arr * 0.28
    result = (arr * (1 - alpha[:, :, None]) + heat * alpha[:, :, None]).astype(np.uint8)
    concentration = float(np.mean(magnitude >= np.quantile(magnitude, 0.9)) * 100)
    return data_url(Image.fromarray(result)), concentration


def signed_values(attribution: torch.Tensor) -> np.ndarray:
    """Collapse RGB attributions into the signed spatial quantity used in maps."""
    return attribution.squeeze(0).detach().cpu().numpy().sum(axis=0)


def difference_heatmap(attribution: torch.Tensor, scale: float) -> str:
    """Render attribution change on a neutral field so it cannot hide in the photo."""
    signed = np.clip(signed_values(attribution) / scale, -1, 1)
    # Pillow cannot blur mode F directly; encode signed values into a temporary
    # 8-bit image, blur it, then decode back to [-1, 1].
    encoded = ((signed + 1.0) * 127.5).astype(np.uint8)
    blurred = Image.fromarray(encoded, mode="L").filter(ImageFilter.GaussianBlur(2.5))
    signed = np.asarray(blurred, dtype=np.float32) / 127.5 - 1.0
    signed = np.clip(signed, -1, 1)
    alpha = np.clip(np.abs(signed) * 3.4, 0, 1)
    canvas = np.full((224, 224, 3), [8, 15, 48], dtype=np.float32)
    red, blue = np.array([50, 255, 190]), np.array([205, 65, 255])
    positive = signed >= 0
    canvas[positive] = canvas[positive] * (1 - alpha[positive, None]) + red * alpha[positive, None]
    canvas[~positive] = canvas[~positive] * (1 - alpha[~positive, None]) + blue * alpha[~positive, None]
    return data_url(Image.fromarray(canvas.astype(np.uint8)))


def evidence_only(image: Image.Image, attribution: torch.Tensor, scale: float) -> str:
    """Keep high-attribution regions bright and dim the rest of the input."""
    base = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) * 0.20
    raw = signed_values(attribution)
    signed = np.clip(raw / scale, -1, 1)
    encoded = ((signed + 1.0) * 127.5).astype(np.uint8)
    signed = np.asarray(Image.fromarray(encoded, mode="L").filter(ImageFilter.GaussianBlur(2.5)), dtype=np.float32) / 127.5 - 1.0
    alpha = np.clip(np.abs(signed) * 2.8, 0, 1) * 0.95
    heat = np.zeros((224, 224, 3), dtype=np.float32)
    positive = signed >= 0
    heat[positive] = np.array([50, 255, 190])
    heat[~positive] = np.array([205, 65, 255])
    result = (base * (1 - alpha[:, :, None]) + heat * alpha[:, :, None]).clip(0, 255).astype(np.uint8)
    return data_url(Image.fromarray(result))


def magnitude_heatmap(attribution: torch.Tensor) -> str:
    """Standalone magma-style positive attribution map, matching common IG plots."""
    magnitude = np.abs(attribution.squeeze(0).detach().cpu().numpy()).mean(axis=0)
    scale = float(np.quantile(magnitude, 0.90) + 1e-8)
    value = np.clip(magnitude / scale, 0, 1)
    encoded = (value * 255).astype(np.uint8)
    value = np.asarray(Image.fromarray(encoded, mode="L").filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32) / 255.0
    # Dark purple -> magenta -> orange -> yellow.
    stops = np.array([[12, 4, 38], [77, 12, 105], [190, 36, 108], [255, 125, 45], [255, 224, 92]], dtype=np.float32)
    positions = value * (len(stops) - 1)
    lo = np.floor(positions).astype(int).clip(0, len(stops) - 1)
    hi = np.ceil(positions).astype(int).clip(0, len(stops) - 1)
    mix = (positions - lo)[..., None]
    rgb = stops[lo] * (1 - mix) + stops[hi] * mix
    return data_url(Image.fromarray(rgb.astype(np.uint8)))


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/styles.css")
def stylesheet():
    return FileResponse("styles.css", media_type="text/css")


@app.get("/app.js")
def client_script():
    return FileResponse("app.js", media_type="application/javascript")


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    target_index: int | None = Form(None),
):
    try:
        original = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, "Please upload a valid image file.") from exc

    model = get_model()
    input_tensor = preprocess(original)
    with torch.no_grad():
        probabilities = F.softmax(model(input_tensor), dim=1)[0]
        values, indices = torch.topk(probabilities, 5)
    target = int(target_index) if target_index is not None else int(indices[0])
    if target < 0 or target >= len(LABELS):
        raise HTTPException(400, "Unknown ImageNet class index")

    raw_attributions = []
    for baseline in ("neutral", "black", "white", "blurred"):
        attributions, convergence_delta = integrated_gradients(
            model, input_tensor, baseline_for(baseline, original, input_tensor), target, n_steps=48
        )
        raw_attributions.append((baseline, attributions, convergence_delta))

    # One common scale prevents each panel's independent normalisation from
    # making distinct attribution magnitudes look deceptively alike.
    common_scale = float(np.quantile(np.abs(np.concatenate([
        signed_values(item[1]).ravel() for item in raw_attributions
    ])), 0.99) + 1e-8)
    common_magnitude_scale = float(np.quantile(np.concatenate([
        np.abs(item[1].detach().cpu().numpy()).mean(axis=1).ravel() for item in raw_attributions
    ]), 0.90) + 1e-8)
    neutral_attr = raw_attributions[0][1]
    explanations = []
    for baseline, attributions, convergence_delta in raw_attributions:
        overlay, concentration = heatmap_overlay(original, attributions, common_scale)
        heatmap = magnitude_heatmap(attributions)
        evidence = evidence_only(original, attributions, common_magnitude_scale)
        difference = attributions - neutral_attr
        # A lower percentile reveals meaningful moderate shifts instead of only
        # the few most extreme pixels.
        difference_scale = float(np.quantile(np.abs(signed_values(difference)), 0.80) + 1e-8)
        comparison_overlay = difference_heatmap(difference, difference_scale)
        explanations.append({
            "baseline": baseline, "overlay": overlay,
            "baseline_preview": data_url(baseline_preview(baseline, original)),
            "heatmap": heatmap, "evidence_only": evidence,
            "comparison_overlay": comparison_overlay,
            "mean_difference": round(float(torch.mean(torch.abs(difference)).item()), 5),
            "convergence_delta": round(float(abs(convergence_delta.item())), 5),
            "focus_concentration": round(concentration, 1),
        })
    return {
        "prediction": {"index": target, "label": LABELS[target], "confidence": round(float(probabilities[target]) * 100, 2)},
        "predictions": [
            {"index": int(index), "label": LABELS[int(index)], "confidence": round(float(value) * 100, 2)}
            for value, index in zip(values, indices)
        ],
        "explanations": explanations,
    }
