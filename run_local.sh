#!/bin/bash
set -e  # stop on first error

echo "🚀 Setting up Virtual Pantry project..."

# 1. Create venv with Python 3.9
if ! command -v python3.9 &> /dev/null
then
    echo "⚠️ Python 3.9 not found. Please install it first (brew install python@3.9 on macOS)."
    exit 1
fi

python3.9 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install dependencies with compatible versions
cat > backend/requirements.txt <<EOL
# --- Core AI & Image Processing ---
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
tensorflow-macos==2.13.0
keras==2.13.1
typing-extensions==4.5.0
ultralytics==8.1.30
opencv-python==4.8.1.78
Pillow==10.0.0
numpy>=1.23.5,<1.25
python-dotenv

# --- Google Gemini AI ---
google-generativeai>=0.5.0,<1.0.0

# --- HTTP requests & date parsing ---
requests==2.31.0
python-dateutil==2.8.2

# --- API Server ---
flask
fastapi==0.103.1
uvicorn==0.23.2
EOL


echo "📦 Installing dependencies..."
pip install -r backend/requirements.txt

# 4. Run backend (Flask serves frontend templates & static)
echo "✅ Setup complete. Starting backend..."
cd backend
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5002

