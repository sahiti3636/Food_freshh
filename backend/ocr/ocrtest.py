
import google.generativeai as genai
from PIL import Image

# 1. Configure Gemini with your API key
# 👉 Replace with your own key from https://ai.google.dev
genai.configure(api_key="AIzaSyAe8KatitkiF60XEHseZbhTuFJe3lltSiM")

# 2. Load the image
image_path = "ocrt.jpg"   # change path if needed
img = Image.open(image_path)

# 3. Create a Gemini model instance
model = genai.GenerativeModel("gemini-1.5-flash")

# 4. Ask Gemini to extract expiry date
prompt = "Extract only the expiry date (best before date) from this image."

response = model.generate_content([prompt, img])

# 5. Print the result
print("Expiry Date:", response.text.strip())