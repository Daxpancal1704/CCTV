from deepface import DeepFace

KNOWN_FACE = "media/employees/dax.jpg"

def recognize_face(frame_path):

    try:

        result = DeepFace.verify(
            img1_path=KNOWN_FACE,
            img2_path=frame_path,
            enforce_detection=False
        )

        if result["verified"]:
            return "Dax Panchal"

        return "Unknown"

    except:
        return "Unknown"