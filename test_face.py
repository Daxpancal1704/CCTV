from deepface import DeepFace

result = DeepFace.verify(
    img1_path="media/employees/dax.jpg",
    img2_path="media/current_face.jpg",
    enforce_detection=False
)

print(result)
print("Verified :", result["verified"])
print("Distance :", result["distance"])
print("Threshold:", result["threshold"])