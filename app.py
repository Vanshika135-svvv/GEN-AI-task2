import os
import base64
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from io import BytesIO
from huggingface_hub import InferenceClient

load_dotenv()
app = Flask(__name__)

# Modern 2026 Inference Client
client = InferenceClient(
    provider="nscale",
    api_key=os.environ.get("HF_TOKEN"),
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        # Generate image
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
        )

        # Convert PIL to Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'image': img_str})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)