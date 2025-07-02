import face_recognition
import cv2

def get_embedding(face_image):
    if face_image is None:
        return None

    if face_image.dtype != 'uint8':
        face_image = face_image.astype('uint8')

    rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb)
    if len(encodings) == 0:
        return None

    return encodings[0]
