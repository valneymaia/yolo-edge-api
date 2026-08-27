import base64
import io
import time
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response, Query
from PIL import Image
import numpy as np
import httpx
import cv2


from schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, MetricsResponse, Detection
)
from model import load_model, get_default_model_name


app = FastAPI(
    title="YOLO Inference API",
    description="API REST para inferência com YOLOv8 e Câmera no Raspberry Pi 5",
    version="1.1.0",
)


_metrics = {"total": 0, "success": 0, "total_ms": 0.0}


def _decode_image(image_base64: str) -> np.ndarray:
    raw = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


def _load_image_from_request(request: PredictRequest) -> np.ndarray:
    if not request.image_base64 and not request.image_url:
        raise HTTPException(status_code=422, detail="Forneça image_base64 ou image_url.")


    if request.image_base64:
        return _decode_image(request.image_base64)
    else:
        resp = httpx.get(request.image_url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return np.array(img)


def _capture_frame_from_camera(device_id: int = 0) -> np.ndarray:
    """Captura frame via rpicam-still (Câmera CSI) ou OpenCV (Câmera USB)."""
    # 1. Tenta captura nativa CSI da Raspberry Pi (rpicam-still / libcamera-still)
    for cmd_tool in ["rpicam-still", "libcamera-still"]:
        try:
            cmd = [
                cmd_tool,
                "-t", "500",            # 500ms para ajuste de exposição e balanço de branco
                "-n",                   # Sem janela de preview
                "-o", "-",              # Saída direta em memória (stdout)
                "--width", "640",
                "--height", "480",
                "-e", "jpg"
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if result.returncode == 0 and len(result.stdout) > 0:
                img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
                return np.array(img)
        except Exception:
            pass


    # 2. Fallback para Câmeras USB padrão (V4L2)
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        try:
            for _ in range(3):
                cap.read()
            ret, frame_bgr = cap.read()
            if ret and frame_bgr is not None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()


    raise HTTPException(
        status_code=500,
        detail="Falha ao capturar imagem da câmera. Verifique a conexão do cabo flat."
    )


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


# ── Endpoints Originais ─────────────────────────────────────


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
    _metrics["total"] += 1
    try:
        img = _load_image_from_request(request)
        result = _run_inference(img, request.model_name, request.confidence)
        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image", responses={200: {"content": {"image/jpeg": {}}}})
def predict_image(request: PredictRequest):
    """Executa inferência em imagem enviada e retorna JPEG com caixas delimitadoras."""
    _metrics["total"] += 1
    try:
        img_rgb = _load_image_from_request(request)
        model = load_model(request.model_name)


        t0 = time.perf_counter()
        results = model(img_rgb, conf=request.confidence, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000


        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms


        annotated_array = results[0].plot()
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


# ── Novos Endpoints: Integração com a Câmera ────────────────


@app.post("/predict/camera", response_model=PredictResponse)
def predict_from_camera(
    device_id: int = Query(0, description="Índice do dispositivo (/dev/videoX)"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Limiar de confiança"),
    model_name: str = Query("yolov8n.pt", description="Modelo YOLO a ser utilizado")
):
    """Dispara a captura de uma foto pela câmera, infere e retorna as detecções em JSON."""
    _metrics["total"] += 1
    try:
        img_rgb = _capture_frame_from_camera(device_id=device_id)
        result = _run_inference(img_rgb, model_name, confidence)
        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/camera/image", responses={200: {"content": {"image/jpeg": {}}}})
def predict_from_camera_image(
    device_id: int = Query(0, description="Índice do dispositivo (/dev/videoX)"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Limiar de confiança"),
    model_name: str = Query("yolov8n.pt", description="Modelo YOLO a ser utilizado")
):
    """Dispara a câmera, infere e renderiza a foto real capturada com os bounding boxes na Swagger UI."""
    _metrics["total"] += 1
    try:
        img_rgb = _capture_frame_from_camera(device_id=device_id)
        model = load_model(model_name)


        t0 = time.perf_counter()
        results = model(img_rgb, conf=confidence, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000


        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms


        annotated_array = results[0].plot()
        annotated_pil = Image.fromarray(annotated_array)


        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG", quality=95)


        return Response(content=buffer.getvalue(), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoints Batch e Métricas ──────────────────────────────


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

