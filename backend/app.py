import os
import json
import requests
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# --- Config ---
UPLOAD_FOLDER = "uploads"
PANTRY_FILE = "pantry.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")  
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  

# Model paths
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "freshness_detection/yolov8n_trained18.pt")
FRESHNESS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "freshness3.h5")

# Flask setup
app = Flask(
    __name__, 
    template_folder="../frontend/templates",   # relative to backend folder
    static_folder="../frontend/static"         
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load Models ---
print("Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH, verbose=False)
freshness_model = load_model(FRESHNESS_MODEL_PATH, compile=False)

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

print("All models loaded")

# --- Class mappings ---
class_names = {
    0: 'fresh apple', 1: 'fresh banana', 2: 'fresh cucumber', 3: 'fresh orange',
    4: 'fresh potato', 5: 'rotten apple', 6: 'rotten banana', 7: 'rotten cucumber',
    8: 'rotten orange', 9: 'rotten potato'
}

yolo_to_freshness = {
    'apple': [0, 5], 'banana': [1, 6], 'cucumber': [2, 7],
    'orange': [3, 8], 'potato': [4, 9]
}

# --- Pantry Helpers ---
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_pantry():
    if not os.path.exists(PANTRY_FILE):
        return []
    with open(PANTRY_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return []

def save_pantry(pantry):
    with open(PANTRY_FILE, "w") as f:
        json.dump(pantry, f, indent=2)

# --- Freshness Logic ---
def preprocess_for_freshness(crop_array):
    img = Image.fromarray(crop_array).resize((150, 150))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def analyze_freshness(img_np, img_bgr):
    results = yolo_model(img_bgr, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()

    freshness_results = []
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        if conf < 0.3:
            continue
        name = yolo_model.model.names[cls_id].lower()
        if name in yolo_to_freshness:
            x1, y1, x2, y2 = box
            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            inp = preprocess_for_freshness(crop)
            preds = freshness_model.predict(inp, verbose=0)[0]
            indices = yolo_to_freshness[name]
            chosen_index = indices[np.argmax(preds[indices])]
            label = class_names[chosen_index]
            freshness_results.append({"type": name, "rotten": "rotten" in label})

    return freshness_results

# --- Gemini OCR + Expiry ---
import io
import base64

def gemini_classify_and_expiry(img_pil):
    # Convert image to bytes
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    prompt = (
        "Look at this image. First, tell me if this is a packaged product or fruit/vegetable.\n"
        "If packaged, extract product name and expiry date.\n"
        "If fruit/veg, identify the type and freshness (fresh/rotten).\n"
        "Respond in format:\n"
        "Type: packaged/fruit\nName: <name>\nExpiry date: <date if packaged>\nFreshness: <fresh/rotten>"
    )

    response = gemini_model.generate_content([
        {"mime_type": "image/jpeg", "data": img_bytes},  
        prompt
    ])
    return response.text.strip()

# --- Recipe Image Fetch ---
def get_recipe_image(recipe_title):
    """Fetch recipe image URL from Google Custom Search API"""
    try:
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': "AIzaSyCIQ_twaFj0IveSy7C7rvGfsRf1Q5fdk2g",
            'cx': '8218a656b43c24efb',
            'q': f"{recipe_title} recipe food",
            'searchType': 'image',
            'num': 1,
            'imgSize': 'medium',
            'imgType': 'photo',
            'safe': 'active'
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data and len(data['items']) > 0:
                return data['items'][0]['link']
    except Exception as e:
        print(f"Error fetching image for {recipe_title}: {e}")
    
    # Return placeholder image if API fails
    return f"/placeholder.svg?height=200&width=300"

# --- Routes ---
@app.route("/")
def index():
    return redirect(url_for("pantry"))

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/pantry")
def pantry():
    return render_template("pantry.html")

@app.route("/recipes")
def recipes():
    return render_template("recipes.html")

# --- API Endpoints ---
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # Open image
    img_pil = Image.open(path).convert("RGB")
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # --- Gemini classification ---
    gemini_result = gemini_classify_and_expiry(img_pil)
    lines = gemini_result.splitlines()
    gemini_dict = {line.split(":", 1)[0].strip().lower(): line.split(":", 1)[1].strip()
                   for line in lines if ":" in line}

    product_type = gemini_dict.get("type")
    product_name = gemini_dict.get("name")
    expiry = gemini_dict.get("expiry date")

    item = {
        "name": product_name,
        "type": product_type,
        "expiry_date": expiry,
        "freshness_rating": None,
        "warnings": [],
        "notices": []
    }

    # Freshness logic
    if product_type == "fruit":
        freshness = analyze_freshness(img_np, img_bgr)
        if freshness:
            item["type"] = freshness[0]["type"]
            item["freshness_rating"] = 10 if not freshness[0]["rotten"] else 2
            if freshness[0]["rotten"]:
                item["warnings"].append(f"{item['name']} is rotten! Please throw it out.")

    if product_type == "packaged" and expiry:
        try:
            exp_date = datetime.strptime(expiry, "%d.%m.%y")
            if exp_date < datetime.now():
                item["warnings"].append(f"{item['name']} expired on {expiry}. Throw it out!")
        except:
            item["notices"].append("Could not parse expiry date.")

    
    should_add = True

    if product_type == "fruit" and freshness and freshness[0]["rotten"]:
        should_add = False

    if product_type == "packaged" and "expired" in " ".join(item["warnings"]).lower():
        should_add = False

    if should_add:
        pantry = load_pantry()
        pantry.append(item)
        save_pantry(pantry)

    return jsonify(item)   

@app.route("/api/pantry", methods=["GET"])
def api_pantry():
    return jsonify(load_pantry())

@app.route("/api/recipes", methods=["GET"])
def api_recipes():
    pantry = load_pantry()
    ingredients = [item["name"] for item in pantry if item.get("name")]

    if not ingredients:
        return jsonify([])

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": f"Suggest 3 recipes using these items: {', '.join(ingredients)}. "
                           "Respond in JSON with keys: title, description, ingredients, instructions."
            }
        ]
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        text = data["choices"][0]["message"]["content"].strip()
        recipes = json.loads(text[text.find("["): text.rfind("]")+1])
        
        for recipe in recipes:
            recipe['image_url'] = get_recipe_image(recipe['title'])
        
        return jsonify(recipes)
    except Exception as e:
        print("⚠️ Groq error:", e)
        return jsonify([])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
