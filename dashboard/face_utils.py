from deepface import DeepFace
from .models import BlacklistPerson
import os

def recognize_face(frame_path):

    try:

        print("Checking:", frame_path)
        result = DeepFace.verify(
            img1_path="media/employees/dax.jpg",
            img2_path=frame_path,
            model_name="Facenet512",
            enforce_detection=False
        )

        print("RESULT =", result)
        print("Distance =", result["distance"])
        print("Verified =", result["verified"])

        distance = result["distance"]

        if distance < 0.50:
            return "Dax Panchal"

        return "Unknown"

        print("Distance =", distance)

    except Exception as e:

        print("ERROR =", e)
        return "Unknown"
    

def detect_emotion(frame_path):
    try:
        result = DeepFace.analyze(
            img_path=frame_path,
            actions=['emotion'],
            enforce_detection=False
        )

        return result[0]['dominant_emotion']

    except Exception as e:
        print("Emotion Error =", e)
        return "Unknown"
    


def check_blacklist(face_path):

    blacklist = BlacklistPerson.objects.all()

    for person in blacklist:

        try:
            print("FACE PATH =", face_path)
            print("BLACKLIST PATH =", person.image.path)

            if not os.path.exists(person.image.path):
                print("Blacklist file missing:", person.image.path)
                continue
            result = DeepFace.verify(
                img1_path=face_path,
                img2_path=str(person.image.path),
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=False
            )

            print("Blacklist Result =", result)
            print("Blacklist Distance =", result["distance"])
            
            if result["distance"] < 0.50:
                return person.name

        except Exception:
            
            import traceback
            print("BLACKLIST FULL ERROR")
            traceback.print_exc()

    return None