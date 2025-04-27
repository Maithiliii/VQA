import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering

# Initialize Flask app
app = Flask(__name__)

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Set up upload folder for images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image and question
@app.route('/ask_question', methods=['POST'])
def ask_question():
    # Get the uploaded image and question from the request
    image_file = request.files.get('image')
    question = request.form.get('question')

    if not image_file or not question:
        return jsonify({'error': 'Image or question missing'}), 400

    # Process image and question
    try:
        # Load image from the uploaded file
        image = Image.open(image_file).convert("RGB")
        
        # Preprocess the image and question for the model
        inputs = processor(image, question, return_tensors="pt").to(device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted answer
        predicted_idx = logits.argmax(-1).item()
        answer = model.config.id2label[predicted_idx]

        return jsonify({'answer': answer})
    
    except Exception as e:
        return jsonify({'error': f'Error processing the image or question: {str(e)}'}), 400

if __name__ == "__main__":
    app.run(debug=True)
