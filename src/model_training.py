# Suppress macOS warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
from PIL import Image
import os
import logging
from config import PATHS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(image_path: str):
    """
    Preprocess the image to normalize lighting and align faces consistently.
    """
    try:
        # Convert image to grayscale
        PIL_img = Image.open(image_path).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        # Optionally apply histogram equalization for normalization
        img_numpy = cv2.equalizeHist(img_numpy)

        return img_numpy
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def get_images_and_labels(path: str):
    """
    Load face images and corresponding labels from the given directory path.

    Parameters:
        path (str): Directory path containing face images.

    Returns:
        tuple: (face_samples, ids) Lists of face samples and corresponding labels.
    """
    try:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
        if not imagePaths:
            logger.warning("No images found in the directory.")
            return [], []

        faceSamples = []
        ids = []

        # Create face detector
        detector = cv2.CascadeClassifier(PATHS['cascade_file'])
        if detector.empty():
            raise ValueError("Error loading cascade classifier")

        for imagePath in imagePaths:
            try:
                # Validate filename structure
                filename = os.path.split(imagePath)[-1]
                parts = filename.split("-")
                if len(parts) < 3 or not parts[1].isdigit():
                    logger.warning(f"Invalid file format, skipping: {filename}")
                    continue

                # Extract ID
                id = int(parts[1])

                # Preprocess image and detect faces
                img_numpy = preprocess_image(imagePath)
                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(id)
            except Exception as e:
                logger.warning(f"Error processing file {imagePath}: {e}")
                continue

        return faceSamples, ids
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise


if __name__ == "__main__":
    try:
        logger.info("Starting face recognition training...")

        # Initialize face recognizer with custom parameters
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,   # Smaller radius captures finer details
            neighbors=8,  # Number of neighbors for LBP calculation
            grid_x=8,  # Grid size along x-axis
            grid_y=8   # Grid size along y-axis
        )

        # Check if a previously trained model exists
        trainer_file = PATHS['trainer_file']
        if os.path.exists(trainer_file):
            recognizer.read(trainer_file)
            logger.info("Loaded existing model for incremental training.")

        # Create face detector
        detector = cv2.CascadeClassifier(PATHS['cascade_file'])
        if detector.empty():
            raise ValueError("Error loading cascade classifier")

        # Get training data
        faces, ids = get_images_and_labels(PATHS['image_dir'])

        if not faces or not ids:
            raise ValueError("No training data found")

        # Train or update the model
        if os.path.exists(trainer_file):
            logger.info("Updating existing model...")
            recognizer.update(faces, np.array(ids))
        else:
            logger.info("Training new model...")
            recognizer.train(faces, np.array(ids))

        # Save the model
        recognizer.write(trainer_file)
        logger.info(f"Model trained/updated with {len(np.unique(ids))} unique faces")

        # Set a stricter confidence threshold
        CONFIDENCE_THRESHOLD = 80  # Lower value means stricter matching

        def predict_face(test_image_path):
            """
            Predict the face ID and confidence for a given test image.

            Parameters:
                test_image_path (str): Path to the test image.

            Returns:
                None: Logs the prediction result.
            """
            try:
                # Create face detector
                detector = cv2.CascadeClassifier(PATHS['cascade_file'])
                if detector.empty():
                    raise ValueError("Error loading cascade classifier")

                img = preprocess_image(test_image_path)
                faces = detector.detectMultiScale(img)

                for (x, y, w, h) in faces:
                    face = img[y:y+h, x:x+w]
                    label, confidence = recognizer.predict(face)

                    if confidence < CONFIDENCE_THRESHOLD:
                        logger.info(f"Predicted ID: {label} with confidence: {confidence}")
                    else:
                        logger.warning(f"Prediction confidence too low ({confidence}). Face not recognized.")
            except Exception as e:
                logger.error(f"Error during prediction: {e}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
