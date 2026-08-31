import base64
import io
import time
import json
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from PIL import Image
import numpy as np
import httpx
from schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, MetricsResponse, Detection
)
from model import load_model, get_default_model_name

app = FastAPI(
    title="YOLO Inference API",
    description="API REST para inferência com YOLOv8 no Raspberry Pi 5",
    version="1.0.0",
)

# ── Métricas simples em memória ─────────────────────────────
_metrics = {"total": 0, "success": 0, "total_ms": 0.0}

def log_event(event: str, level: str = "INFO", **kwargs):
    """Emite um evento estruturado em JSON para stdout."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level":     level,
        "event":     event,
        **kwargs,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)

def _decode_image(image_base64: str) -> np.ndarray:
    """Converte base64 → numpy array RGB."""
    raw = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)

def _load_image_from_request(request: PredictRequest) -> np.ndarray:
    """Lê a imagem a partir de Base64 ou URL pública sempre em RGB."""
    if not request.image_base64 and not request.image_url:
        raise HTTPException(
            status_code=422,
            detail="Forneça image_base64 ou image_url."
        )

    if request.image_base64:
        return _decode_image(request.image_base64)
    else:
        resp = httpx.get(request.image_url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return np.array(img)

def _run_inference(image_np: np.ndarray, model_name: str, confidence: float) -> PredictResponse:
    model = load_model(model_name)
    t0 = time.perf_counter()
    results = model(image_np, conf=confidence, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    detections = []
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            detections.append(Detection(
                label=model.names[cls_id],
                confidence=round(conf_val, 4),
                bbox=[round(float(c), 2) for c in coords],
            ))
    h, w = image_np.shape[:2]
    return PredictResponse(
        detections=detections,
        inference_ms=round(elapsed_ms, 2),
        model_used=model_name,
        image_width=w,
        image_height=h,
    )

# ── Endpoints ───────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_name = get_default_model_name()
    try:
        load_model(model_name)
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded, model_name=model_name)

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    request_id = str(uuid.uuid4())[:8]
    _metrics["total"] += 1

    log_event("predict_start",
              request_id=request_id,
              model=request.model_name,
              confidence=request.confidence)

    if not request.image_base64 and not request.image_url:
        log_event("predict_error", level="WARN",
                  request_id=request_id, reason="missing_input")
        raise HTTPException(status_code=422,
            detail="Forneça image_base64 ou image_url.")

    try:
        if request.image_base64:
            img = _decode_image(request.image_base64)
        else:
            resp = httpx.get(request.image_url, timeout=10)
            resp.raise_for_status()
            img = _decode_image(base64.b64encode(resp.content).decode())

        result = _run_inference(img, request.model_name, request.confidence)
        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms

        log_event("predict_complete",
                  request_id=request_id,
                  model=result.model_used,
                  detections=len(result.detections),
                  inference_ms=result.inference_ms,
                  image_size=f"{result.image_width}x{result.image_height}")
        return result

    except FileNotFoundError as e:
        log_event("predict_error", level="ERROR",
                  request_id=request_id, reason=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_event("predict_error", level="ERROR",
                  request_id=request_id, reason=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image", responses={200: {"content": {"image/jpeg": {}}}})
def predict_image(request: PredictRequest):
    """Executa a inferência e retorna a imagem anotada em JPEG com cores 100% calibradas em RGB."""
    _metrics["total"] += 1
    try:
        # 1. Carrega imagem em RGB
        img_rgb = _load_image_from_request(request)
        model = load_model(request.model_name)

        t0 = time.perf_counter()
        results = model(img_rgb, conf=request.confidence, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms
        # 2. plot() retorna o array RGB anotado
        annotated_array = results[0].plot()

        # 3. Salva diretamente via PIL (RGB nativo da web, sem conversão indevida do OpenCV)
        annotated_pil = Image.fromarray(annotated_array)
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG", quality=95)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    t_total = time.perf_counter()
    results = []
    for img_b64 in request.images_base64:
        img = _decode_image(img_b64)
        results.append(_run_inference(img, request.model_name, request.confidence))
    total_ms = (time.perf_counter() - t_total) * 1000
    return BatchPredictResponse(results=results, total_inference_ms=round(total_ms, 2))

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    avg = (_metrics["total_ms"] / _metrics["success"] if _metrics["success"] > 0 else 0.0)
    return MetricsResponse(
        total_requests=_metrics["total"],
        successful_requests=_metrics["success"],
        avg_inference_ms=round(avg, 2),
    )
