import uvicorn
import numpy as np
import cv2
import onnxruntime as ort
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Waste Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model
try:
    session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    session = None
    input_name = None

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def preprocess(img: np.ndarray) -> np.ndarray:
    """Preprocess image for YOLO model."""
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1))
    img_float = img_transposed.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_float, axis=0)
    return img_input


def postprocess(preds: np.ndarray, orig_shape: tuple, conf_thres: float = 0.25, 
                iou_thres: float = 0.45) -> tuple:
    """Post-process model predictions."""
    preds = preds.squeeze().T

    boxes = preds[:, :4]
    scores = preds[:, 4:]

    class_ids = scores.argmax(axis=1)
    confidences = scores.max(axis=1)

    mask = confidences > conf_thres
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return [], [], []

    # Convert to xyxy format
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # Scale to original image size
    h0, w0 = orig_shape
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] * w0 / 640
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] * h0 / 640

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        conf_thres,
        iou_thres
    )

    if len(indices) == 0:
        return [], [], []

    indices = np.array(indices).flatten()

    final_boxes = boxes_xyxy[indices].tolist()
    final_scores = confidences[indices].tolist()
    final_classes = class_ids[indices].tolist()

    return final_boxes, final_scores, final_classes


@app.on_event("startup")
async def warmup():
    """Warm up the model on startup."""
    if session is not None:
        try:
            dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
            session.run(None, {input_name: dummy_input})
            logger.info("✅ Model warmed up successfully")
        except Exception as e:
            logger.error(f"⚠️ Model warmup failed: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Waste Detection API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if session is not None else "unhealthy",
        "model_loaded": session is not None
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf_threshold: float = 0.25):
    """Detect objects in uploaded image."""
    if session is None:
        return JSONResponse(
            {"error": "Model not loaded"}, 
            status_code=503
        )
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            return JSONResponse(
                {"error": "Invalid file type. Please upload an image."}, 
                status_code=400
            )
        
        # Read and validate file size
        image_data = await file.read()
        if len(image_data) > MAX_FILE_SIZE:
            return JSONResponse(
                {"error": f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB"}, 
                status_code=413
            )
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                {"error": "Failed to decode image"}, 
                status_code=400
            )
        
        orig_h, orig_w = img.shape[:2]
        
        # Preprocess
        input_tensor = preprocess(img)
        
        # Run inference
        outputs = session.run(None, {input_name: input_tensor})
        preds = outputs[0]
        
        # Postprocess
        boxes, scores, classes = postprocess(
            preds, 
            (orig_h, orig_w),
            conf_thres=conf_threshold
        )
        
        logger.info(f"Detected {len(boxes)} objects with confidence >= {conf_threshold}")
        
        return JSONResponse({
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "count": len(boxes)
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            {"error": f"Processing error: {str(e)}"}, 
            status_code=500
        )
    finally:
        # Cleanup
        try:
            del image_data, nparr, img
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )