from deepface import DeepFace

result = DeepFace.verify(
    img1_path="media/employees/dax.jpg",
    img2_path="media/employees/dax.jpg",
    model_name="Facenet512",
    detector_backend="opencv",
    enforce_detection=True
)

print(result)