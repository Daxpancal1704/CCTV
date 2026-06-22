from deepface import DeepFace

def recognize_face(frame_path):

    try:

        result = DeepFace.verify(
            img1_path="media/employees/dax.jpg",
            img2_path=frame_path,
            model_name="Facenet512",
            enforce_detection=False
        )

        print("RESULT =", result)
        print("Distance =", result["distance"])
        print("Verified =", result["verified"])

        if result["verified"]:
            return "Dax Panchal"

        return "Unknown"

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