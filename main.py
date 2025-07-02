import cv2
import uuid
from detect import get_face_location
from typesense_reco import create_collection, store_faces, predict_using_typesense
from recognier import FaceRecognizer
from embedding import get_embedding

create_collection()


def register(image_path,name):
    image= cv2.imread(image_path)

    face_location= get_face_location(image)
    if face_location is None:
        print("Not able to detect face ")
        return
    
    top, right, bottom, left = face_location
    face = image[top:bottom, left:right]
    
    embedding=get_embedding(face)
    if embedding is None:
        print("not able to extract the embedding")
        return
    
    store_faces(str(uuid.uuid4()),name,embedding.tolist())
    print(f"stored face for :{name}")

def recognize(image_path):
    image = cv2.imread(image_path)

    face_location = get_face_location(image)
    if face_location is None:
        print("Could not detect face")
        return

    top, right, bottom, left = face_location
    face = image[top:bottom, left:right]

    embedding = get_embedding(face)
    if embedding is None:
        print("Not able to extract face embedding")
        return

    predicted_name, top_k_matches = predict_using_typesense(embedding.tolist(), k=3)

    print(f"\nPredicted: {predicted_name}")
    print("🔍 Top 3 matches:")
    for i, match in enumerate(top_k_matches, 1):
        print(f" {i}. {match['name']} (distance: {match['distance']:.2f})")


    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 1)
    cv2.putText(image, str(predicted_name), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    print(f"Predicted: {predicted_name}")

    cv2.imshow("Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


register("/home/theertha/Desktop/face/train/mohanlal.jpeg","MOHANLAL")
register("/home/theertha/Desktop/face/train/2025-06-23-171246.jpg","Sree")
register("/home/theertha/Desktop/face/train/mamoty.jpeg","Mammoty")
recognize("/home/theertha/Desktop/face/train/mohanlal2.jpeg")








