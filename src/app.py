from flask import Flask, request, jsonify
import os
import cv2
import base64
import numpy as np
from face_cin_extraction import extract_face as extract_face_from_cin
from face_taker import create_directory, initialize_camera, get_face_id, save_name
from face_trainer import get_images_and_labels
from recognize import initialize_camera as recog_initialize_camera, load_names
import logging
from config import PATHS, CAMERA, TRAINING, CONFIDENCE_THRESHOLD
from flask_cors import CORS
from config import CAMERA, FACE_DETECTION, PATHS, CONFIDENCE_THRESHOLD
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for the entire application
CORS(app, resources={r"/*": {"origins": "https://kyc.trimakus.com"}})

@app.route('/extract_face', methods=['POST'])
def extract_face_from_request():
    try:
        image = request.files.get('image')
        if not image:
            return jsonify({'error': 'No image provided'}), 400

        output_folder = PATHS['image_dir']
        create_directory(output_folder)

        face_id = get_face_id(output_folder)
        existing_files = [f for f in os.listdir(output_folder) if f.startswith(f"Users-{face_id}-")]
        count = len(existing_files)

        temp_image_path = os.path.join(output_folder, image.filename)
        image.save(temp_image_path)

        success = extract_face_from_cin(temp_image_path, output_folder, face_id, count)
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        if success:
            return jsonify({'message': 'Face extracted successfully', 'face_id': face_id}), 200
        else:
            return jsonify({'error': 'Failed to extract face'}), 500
    except Exception as e:
        logger.error(f"Error in extract_face_from_request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/start_capture', methods=['POST'])
def start_capture_api():
    try:
        face_name = request.form.get('name')
        if not face_name:
            return jsonify({'error': 'Name is required'}), 400

        output_folder = PATHS['image_dir']
        create_directory(output_folder)

        face_id = get_face_id(output_folder)
        save_name(face_id, face_name, PATHS['names_file'])

        images = request.files.getlist('images')  # Get all images from the batch
        count = 0

        for image in images:
            temp_image_path = os.path.join(output_folder, image.filename)
            image.save(temp_image_path)

            # Process the image (e.g., extract face)
            success = extract_face_from_cin(temp_image_path, output_folder, face_id, count)
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            if success:
                count += 1
            else:
                logger.warning(f"Failed to process image: {image.filename}")

        return jsonify({'message': f'{count} images captured successfully for {face_name}', 'face_id': face_id}), 200

    except Exception as e:
        logger.error(f"Error during image capture: {e}", exc_info=True)  # Log full traceback
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model_api():
    try:
        faces, ids = get_images_and_labels(PATHS['image_dir'])
        if not faces or not ids:
            return jsonify({'error': 'No training data found'}), 400

        # Initialize face recognizer with custom parameters
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )

        trainer_file = PATHS['trainer_file']
        
        # Check if an existing model is present for incremental training
        if os.path.exists(trainer_file):
            recognizer.read(trainer_file)
            logger.info("Updating existing model...")
            recognizer.update(faces, np.array(ids))
        else:
            logger.info("Training a new model...")
            recognizer.train(faces, np.array(ids))

        # Save the trained model
        recognizer.write(trainer_file)
        logger.info(f"Model trained/updated with {len(np.unique(ids))} unique faces")

        return jsonify({'message': 'Model trained successfully'}), 200

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return jsonify({'error': str(e)}), 500

# Load face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(PATHS['trainer_file']):
    recognizer.read(PATHS['trainer_file'])
else:
    logger.error("Trainer file not found. Please train the model first.")
    exit(1)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
if face_cascade.empty():
    logger.error("Error loading cascade classifier")
    exit(1)

# Load names
def load_names(filename):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                return json.loads(content) if content else {}
    except Exception as e:
        logger.error(f"Error loading names: {e}")
    return {}

names = load_names(PATHS['names_file'])

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION['scale_factor'],
            minNeighbors=FACE_DETECTION['min_neighbors'],
            minSize=FACE_DETECTION['min_size']
        )
        
        results = []
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence >= CONFIDENCE_THRESHOLD:
                name = names.get(str(id), "Unknown")
                message = f"Verified: {name}"
            else:
                name = "Unknown"
                message = "Not Verified"
            
            results.append({
                'name': name,
                'confidence': float(confidence) if confidence != "N/A" else "N/A",
                'message': message
            })
        
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error in recognize_face: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  
