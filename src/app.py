from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from face_cin_extraction import extract_face as extract_face_from_cin
from face_taker import create_directory, initialize_camera, get_face_id, save_name
from face_trainer import get_images_and_labels
from recognize import initialize_camera as recog_initialize_camera, load_names
import logging
from config import PATHS, CAMERA, TRAINING, CONFIDENCE_THRESHOLD
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for the entire application
CORS(app)

@app.route('/extract_face', methods=['POST'])
def extract_face_from_request():
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

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.write(PATHS['trainer_file'])

        return jsonify({'message': 'Model trained successfully'}), 200
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/recognize_face', methods=['POST'])
def recognize_face_api():
    try:
        cam = recog_initialize_camera(CAMERA['index'])
        if not cam:
            return jsonify({'error': 'Failed to initialize camera'}), 500

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(PATHS['trainer_file'])

        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        names = load_names(PATHS['names_file'])

        while True:
            ret, img = cam.read()
            if not ret:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                if confidence >= CONFIDENCE_THRESHOLD:
                    name = names.get(str(id), "Unknown")
                    cam.release()
                    return jsonify({'message': f'Face recognized: {name}', 'confidence': confidence}), 200
                else:
                    cam.release()
                    return jsonify({'message': 'Face not recognized', 'confidence': confidence}), 404

    except Exception as e:
        logger.error(f"Error during face recognition: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if 'cam' in locals():
            cam.release()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
