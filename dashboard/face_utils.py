from deepface import DeepFace
from .models import Employee


def recognize_face(frame_path):

    employees = Employee.objects.all()

    if not employees.exists():
        print("No employees found")
        return "Unknown"

    best_match_name = "Unknown"
    best_distance = 999

    for employee in employees:

        try:

            employee_image = employee.image.path

            result = DeepFace.verify(
                img1_path=employee_image,
                img2_path=frame_path,
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=False
            )

            distance = result.get(
                "distance",
                999
            )

            print("=" * 50)
            print("Employee :", employee.name)
            print("Distance :", distance)
            print("Threshold :", result.get("threshold"))
            print("Verified :", result.get("verified"))
            print("=" * 50)

            # Custom threshold
            if distance < 0.60:

                if distance < best_distance:

                    best_distance = distance
                    best_match_name = employee.name

        except Exception as e:

            print(
                "Recognition Error:",
                employee.name,
                str(e)
            )

    print(
        "Final Match:",
        best_match_name,
        "Distance:",
        best_distance
    )

    return best_match_name