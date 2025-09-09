import os
from flask import Flask, jsonify
from flask_restx import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage
# Disable OpenCV debug threads
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_IO_DISABLE_OPENEXR'] = '1'
os.environ['CV_LOADER_DEBUG'] = '0'

import cv2
cv2.setNumThreads(0)
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from collections import defaultdict

app = Flask(__name__)
api = Api(app, version="1.0", title="Freshness Detection API",
          description="Detect objects and their freshness using YOLO and CNN")

ns = api.namespace("detect", description="Detection operations")

# Configuration
YOLO_CONFIDENCE_THRESHOLD = 0.3
FRESHNESS_INPUT_SIZE = (150, 150)

# Human‑readable names for the 10 CNN output classes
class_names = {
    0: 'fresh apple',
    1: 'fresh banana',
    2: 'fresh cucumber',
    3: 'fresh orange',
    4: 'fresh potato',
    5: 'rotten apple',
    6: 'rotten banana',
    7: 'rotten cucumber',
    8: 'rotten orange',
    9: 'rotten potato'
}

# Map YOLO labels → indices in the freshness model
yolo_to_freshness = {
    'apple':      [0, 5],   # fresh apple, rotten apple
    'banana':     [1, 6],
    'cucumber':   [2, 7],
    'orange':     [3, 8],
    'potato':     [4, 9]
}

# Load models
try:
    model_yolo = YOLO('freshness detection/yolov8n_trained18.pt')
    model_freshness = load_model('freshness3.h5', compile=False)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model_yolo = None
    model_freshness = None

@app.route("/")
def healthcheck():
    return jsonify({"message": "API is running"}), 200

# Parser for file upload
upload_parser = reqparse.RequestParser()
upload_parser.add_argument(
    'file',
    type=FileStorage,
    location='files',
    required=True,
    help="Image file (JPG/PNG/WEBP)"
)

def preprocess_for_freshness(cropped_array):
    """Resize & scale a crop for the freshness model."""
    img = Image.fromarray(cropped_array)
    img = img.resize(FRESHNESS_INPUT_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@ns.route("/")
class FreshnessDetection(Resource):
    @api.expect(upload_parser)
    def post(self):
        if not model_yolo or not model_freshness:
            return {"error": "Models not loaded"}, 500

        args = upload_parser.parse_args()
        file = args.get('file')
        if not file or not isinstance(file, FileStorage):
            return {"error": "Invalid file upload"}, 400

        try:
            # Load & convert image
            img_pil = Image.open(file.stream).convert('RGB')
            img_np  = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # YOLO object detection
            results     = model_yolo(img_bgr)[0]
            boxes       = results.boxes.xyxy.cpu().numpy().astype(int)
            class_ids   = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()

            object_counts   = defaultdict(int)
            freshness_counts = defaultdict(int)

            # # 1) Run YOLO detections
            # for box, cls_id, conf in zip(boxes, class_ids, confidences):
            #     if conf < YOLO_CONFIDENCE_THRESHOLD:
            #         continue

            #     name = model_yolo.model.names[cls_id].lower()
            #     object_counts[name] += 1

            #     # Only run freshness on known produce
            #     if name in yolo_to_freshness:
            #         x1, y1, x2, y2 = box
            #         crop = img_np[y1:y2, x1:x2]
            #         if crop.size == 0:
            #             continue

            #         inp = preprocess_for_freshness(crop)
            #         preds = model_freshness.predict(inp, verbose=0)[0]

            #         # Pick only the two relevant freshness outputs
            #         indices = yolo_to_freshness[name]
            #         sub_preds = preds[indices]
            #         chosen = indices[np.argmax(sub_preds)]
            #         label = class_names[chosen]
            #         freshness_counts[label] += 1
            # 1) Run YOLO detections
            for box, cls_id, conf in zip(boxes, class_ids, confidences):
                if conf < YOLO_CONFIDENCE_THRESHOLD:
                    continue

                name = model_yolo.model.names[cls_id].lower()
                object_counts[name] += 1

                # Only run freshness if supported
                if name in yolo_to_freshness:
                    x1, y1, x2, y2 = box
                    crop = img_np[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    inp = preprocess_for_freshness(crop)
                    preds = model_freshness.predict(inp, verbose=0)[0]

                    # Pick only the two relevant freshness outputs
                    indices = yolo_to_freshness[name]
                    sub_preds = preds[indices]
                    chosen = indices[np.argmax(sub_preds)]
                    label = class_names[chosen]
                    freshness_counts[label] += 1
                else:
                    # New fruit not in freshness model
                    freshness_counts[f"{name} (freshness unknown)"] += 1


            # 2) Fallback: if YOLO detected nothing, classify the entire image
            if not object_counts:
                inp_full = preprocess_for_freshness(img_np)
                preds = model_freshness.predict(inp_full, verbose=0)[0]
                idx = int(np.argmax(preds))
                full_label = class_names[idx]       # e.g. "rotten orange"
                noun = full_label.split(' ', 1)[1]  # "orange"
                object_counts[noun] = 1
                freshness_counts[full_label] = 1

            # Format detected objects list
            detected_objects = [
                f"{cnt} {nm}{'s' if cnt > 1 else ''}"
                for nm, cnt in object_counts.items()
            ]

            # Format freshness status list
            freshness_status = [
                f"{cnt} {label}{'s' if cnt > 1 else ''}"
                for label, cnt in freshness_counts.items()
            ]

            return {
                "detected_objects": detected_objects,
                "freshness_status": freshness_status
            }, 200

        except Exception as e:
            return {"error": str(e)}, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
