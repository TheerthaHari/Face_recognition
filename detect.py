import face_recognition

def get_face_location(image):
    face_location= face_recognition.face_locations(image)
    return face_location[0] if face_location else None

