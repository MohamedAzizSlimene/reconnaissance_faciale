import cv2

if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
    print("LBPHFaceRecognizer_create is available.")
else:
    print("LBPHFaceRecognizer_create is NOT available.")