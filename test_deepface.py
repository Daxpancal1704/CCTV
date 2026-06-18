from deepface import DeepFace

print("Starting...")

result = DeepFace.verify(
    img1_path="media/employees/dax.jpg",
    img2_path="media/employees/dax.jpg",
    enforce_detection=False
)

print(result)
print("Finished")