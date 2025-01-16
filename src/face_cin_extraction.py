import cv2
import os

def extract_face(image_path, output_folder, face_id, count, face_cascade_path=None):
    """
    Extracts a face from an image, converts it to grayscale, and saves it to the specified output path.
    
    Parameters:
        image_path (str): Path to the input image.
        output_folder (str): Path to save the extracted face.
        face_id (int): ID of the person whose face is being extracted.
        count (int): Image count for the given face_id (used for naming convention).
        face_cascade_path (str, optional): Path to the Haar cascade XML file for face detection.

    Returns:
        bool: True if a face is successfully detected and saved, False otherwise.
    """
    # Use default Haar cascade if none is provided
    if face_cascade_path is None:
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print(f"Error: Unable to load Haar cascade from {face_cascade_path}")
        return False

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return False

    # Convert the image to grayscale (Haar cascades require grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        print("No face detected in the image.")
        return False

    # Assume the first detected face is the target
    (x, y, w, h) = faces[0]

    # Crop the face from the image
    face = gray[y:y+h, x:x+w]  # Keep it in grayscale

    # Save the cropped face to the output path directly in the output folder (no subfolder)
    output_image = os.path.join(output_folder, f"Users-{face_id}-{count+1}.jpg")
    cv2.imwrite(output_image, face)
    print(f"Face extracted and saved to {output_image}")
    return True

# Example usage
if __name__ == "__main__":
    input_image = "cin_melik.jpg"  # Replace with the path to your image
    output_folder = "images/"  # Replace with your desired output folder

    # Face ID and count for the naming convention
    face_id = 1  # Replace with the appropriate face ID
    count = 0  # Count of the images for the given face_id

    # Extract the face
    success = extract_face(input_image, output_folder, face_id, count)
    if success:
        print("Face extraction completed successfully.")
    else:
        print("Face extraction failed.")
