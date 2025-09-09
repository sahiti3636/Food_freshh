import os
import argparse
import warnings
import json
from datetime import datetime
from pathlib import Path

# --- AI & Image Processing Imports ---
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import google.generativeai as genai
from dateutil import parser
import requests

# --- Suppress Warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
YOLO_CONFIDENCE_THRESHOLD = 0.3
FRESHNESS_INPUT_SIZE = (150, 150)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "freshness detection/yolov8n_trained18.pt")
FRESHNESS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "freshness3.h5")
PANTRY_FILE = os.path.join(os.path.dirname(__file__), "pantry.db")  # or pantry.json


# --- Model Definitions ---
class_names = {
    0: 'fresh apple', 1: 'fresh banana', 2: 'fresh cucumber', 3: 'fresh orange',
    4: 'fresh potato', 5: 'rotten apple', 6: 'rotten banana', 7: 'rotten cucumber',
    8: 'rotten orange', 9: 'rotten potato'
}

yolo_to_freshness = {
    'apple': [0, 5], 'banana': [1, 6], 'cucumber': [2, 7],
    'orange': [3, 8], 'potato': [4, 9]
}

# --- Global Models ---
model_yolo = None
model_freshness = None
model_ocr = None

# --- Model Loading ---
def load_models():
    global model_yolo, model_freshness, model_ocr

    try:
        model_yolo = YOLO(YOLO_MODEL_PATH, verbose=False)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")

    try:
        model_freshness = load_model(FRESHNESS_MODEL_PATH, compile=False)
        print("Freshness model loaded successfully.")
    except Exception as e:
        print(f"Error loading freshness model: {e}")

    try:
        if GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY":
            genai.configure(api_key=GOOGLE_API_KEY)
            model_ocr = genai.GenerativeModel("gemini-1.5-flash")
            print("Gemini model configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini: {e}")

# --- Pantry Management ---
def get_pantry_items():
    if not os.path.exists(PANTRY_FILE):
        return []
    with open(PANTRY_FILE, "r") as f:
        try:
            return json.load(f)["items"]
        except (json.JSONDecodeError, KeyError):
            return []

def save_to_pantry(new_entry):
    items = get_pantry_items()
    sno = max([item.get('sno', 0) for item in items], default=0) + 1
    new_entry['sno'] = sno
    items.append(new_entry)
    with open(PANTRY_FILE, "w") as f:
        json.dump({"items": items}, f, indent=2)

def list_pantry():
    items = get_pantry_items()
    if not items:
        print("Pantry is empty.")
        return
    for item in items:
        print(f"S.No: {item.get('sno')}, Product: {item.get('product')}, Type: {item.get('type')}, Expiry: {item.get('expiry')}, Rotten: {item.get('rotten')}")

# --- Helper Functions ---
def preprocess_for_freshness(crop_array):
    img = Image.fromarray(crop_array).resize(FRESHNESS_INPUT_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# --- Freshness Analysis ---
def analyze_freshness(img_np, img_bgr):
    if not model_yolo or not model_freshness:
        return {"error": "Freshness models not loaded."}

    results = model_yolo(img_bgr, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()

    freshness_results = []
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        if conf < YOLO_CONFIDENCE_THRESHOLD:
            continue
        name = model_yolo.model.names[cls_id].lower()
        if name in yolo_to_freshness:
            x1, y1, x2, y2 = box
            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0: continue

            inp = preprocess_for_freshness(crop)
            preds = model_freshness.predict(inp, verbose=0)[0]

            indices = yolo_to_freshness[name]
            chosen_index = indices[np.argmax(preds[indices])]
            label = class_names[chosen_index]
            freshness_results.append({"type": name, "rotten": "rotten" in label})

    if not freshness_results:
        inp_full = preprocess_for_freshness(img_np)
        preds = model_freshness.predict(inp_full, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = class_names.get(idx, "unknown item")
        noun = label.split(' ', 1)[-1] if ' ' in label else label
        freshness_results.append({"type": noun, "rotten": "rotten" in label})

    return {"freshness_status": freshness_results}

# --- Gemini Classification ---
def gemini_classify_and_expiry(img_pil):
    if not model_ocr:
        return {"error": "Gemini model not configured."}

    prompt = (
        "Look at this image. First, tell me if this is a packaged product (with a label) or a direct fruit/vegetable.\n"
        "If packaged, extract the product name and expiry/best before date.\n"
        "If fruit/vegetable, identify the type and tell if it is apple/orange/potato/cucumber/banana or something else.\n"
        "Respond in this exact format:\n"
        "Type: packaged/fruit\nName: <name>\nExpiry date: <date if packaged>\nFreshness: <fresh/rotten if fruit>"
    )
    try:
        response = model_ocr.generate_content([prompt, img_pil])
        return {"gemini_result": response.text.strip()}
    except Exception as e:
        return {"error": f"Gemini analysis failed: {e}"}

# --- Recipe Suggestions ---
def suggest_recipes():
    items = get_pantry_items()
    if not items:
        print("Pantry is empty.")
        return
    items_list = ", ".join([item["type"] for item in items])
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set.")
        return

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": f"I have these items: {items_list}. Suggest 3 recipes."}]
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        print(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Error fetching recipes: {e}")

# --- Main Processing ---
def main(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gemini_result = gemini_classify_and_expiry(img_pil)
    if "error" in gemini_result:
        print(f"Error: {gemini_result['error']}")
        return

    text = gemini_result.get("gemini_result", "").lower()
    lines = gemini_result["gemini_result"].splitlines()
    gemini_dict = {line.split(":", 1)[0].strip().lower(): line.split(":", 1)[1].strip() 
                   for line in lines if ":" in line}

    product_type = gemini_dict.get('type')
    product_name = gemini_dict.get('name')

    # --- Packaged Product ---
    if product_type == "packaged":
        print("Packaged product")
        from datetime import datetime

        expiry = gemini_dict.get('expiry date')
        expiry_date = None
        try:
            expiry_date = datetime.strptime(expiry, "%d.%m.%y").date()
            print(f"Parsed expiry date: {expiry_date}")
            if expiry_date < datetime.now().date():
                print("Alert: This product has expired.")
        except ValueError:
            print(f"⚠️ Could not parse expiry date reliably: {expiry}")
            expiry_date = expiry  # fallback


        save_to_pantry({
            "product": "packaged",
            "type": product_name,
            "expiry": str(expiry_date),  # ensure consistent format
            "rotten": None
        })


    # --- Fruit/Vegetable ---
    elif product_type == "fruit":
        print("Fruit/vegetable")
        if any(fruit in text for fruit in ["apple", "orange", "potato", "cucumber", "banana"]):
            freshness_result = analyze_freshness(img_np, img_bgr)
            if "error" in freshness_result:
                print(f"Error: {freshness_result['error']}")
                return
            first_result = freshness_result["freshness_status"][0]
            rotten = first_result.get("rotten", False)
            item_type = first_result.get("type", product_name)
            print(f"Type: {item_type}, Rotten: {rotten}")
            if not rotten:
                save_to_pantry({"product": "open", "type": item_type, "expiry": None, "rotten": rotten})
            else:
                print("Alert: This item is rotten.")
        else:
            # fallback: use Gemini info
            name, fresh = None, None
            for line in lines:
                if line.lower().startswith("name:"):
                    name = line.split(":", 1)[-1].strip()
                if line.lower().startswith("freshness:"):
                    fresh = line.split(":", 1)[-1].strip().lower()
            rotten = fresh == "rotten"
            if not rotten and name:
                save_to_pantry({"product": "open", "type": name, "expiry": None, "rotten": rotten})
            elif rotten:
                print("Alert: This item is rotten.")
    else:
        print("Could not classify the item.")

# --- CLI Entry ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze image for produce freshness & expiry")
    parser.add_argument("image_path", type=str, nargs="?", help="Path to image")
    parser.add_argument("--recipes", action="store_true", help="Suggest recipes")
    parser.add_argument("--list", action="store_true", help="List pantry items")
    args = parser.parse_args()

    load_models()

    if args.recipes:
        suggest_recipes()
    elif args.list:
        list_pantry()
    elif args.image_path:
        main(args.image_path)
    else:
        print("Please provide an image path, --recipes, or --list.")
