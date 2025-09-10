# Food Fresheness Detection and Virtual Pantry

An AI-powered **Virtual Pantry** that helps you:
- Upload grocery item photos to detect type, name, and expiry date using **YOLO** + **Freshness CNN**.  
- Analyze freshness with a custom **Keras model**.  
- Get smart recipe suggestions using **gpt-oss-120b**.  
- Manage your pantry with a clean **Flask + HTML/CSS frontend**.  

---

## Tech Stack
- **Backend**: Flask, FastAPI (API endpoints), TensorFlow, PyTorch, Ultralytics YOLO  
- **Frontend**: HTML, CSS
- **AI Models**:  
  - YOLOv8 (trained for grocery classification)  
  - Freshness detection CNN (`freshness3.h5`)  
  - Grok (`gpt-oss-120b`) for recipe generation  

---

## 📂 Project Structure
```
food-app/
│── backend/            # Flask backend
│   ├── app.py          # Main backend server
│   ├── requirements.txt
│   ├── freshness3.h5   # Freshness model (downloaded automatically if missing)
│   ├── yolov8n_trained18.pt  # YOLO model
│   ├── pantry.json     # Pantry storage
│── frontend/           # HTML/CSS templates
│   ├── templates/
│   │   ├── base.html
│   │   ├── upload.html
│   │   ├── pantry.html
│   │   ├── recipes.html
│   ├── static/         # CSS, images
│── run_local.sh            # Setup script (creates venv + installs deps)
│── README.md
```

---

## Setup & Run Locally

### Clone the repo
```bash
git clone https://github.com/yourusername/virtual-pantry-ai.git
cd virtual-pantry-ai
```

### Make.env
make a .env file in backend with all required API keys

### Run setup script
This will:
- Create a Python **3.9 virtual environment**
- Install all required dependencies
- Start the backend

```bash
chmod +x run_local.sh
./run_local.sh
```

