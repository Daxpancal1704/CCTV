# test_emotion.py

from deepface import DeepFace

result = DeepFace.analyze(
    img_path="media/current_face.jpg",
    actions=['emotion'],
    enforce_detection=False
)

print(result)