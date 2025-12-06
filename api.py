import uvicorn
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name

def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1)) # CHW
    img_float = img_transposed.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_float, axis=0)  # (1, 3, 640, 640)
    return img_input

def postprocess(preds, orig_shape, conf_thres=0.25, iou_thres=0.45):
    preds = preds.squeeze().T  # (8400, 14)

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

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    h0, w0 = orig_shape
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] * w0 / 640
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] * h0 / 640

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


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        orig_h, orig_w = img.shape[:2]

        input_tensor = preprocess(img)

       
        outputs = session.run(None, {input_name: input_tensor})
        preds = outputs[0]  # (1, 14, 8400)

        boxes, scores, classes = postprocess(preds, (orig_h, orig_w))

        return JSONResponse({
            "boxes": boxes,
            "scores": scores,
            "classes": classes
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
