from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse

import os
from datetime import datetime, timedelta, timezone
import cv2
from ultralytics import YOLO

from .face_utils import recognize_face
from .models import Snapshot, CameraLog, RecognitionLog

model = YOLO("yolov8n.pt")
model.to("cuda")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

camera1 = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

camera2 = cv2.VideoCapture(
    1,
    cv2.CAP_DSHOW
)

camera1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

camera1.set(cv2.CAP_PROP_FPS, 30)
camera2.set(cv2.CAP_PROP_FPS, 30)

print("Cam1 FPS:", camera1.get(cv2.CAP_PROP_FPS))
print("Cam2 FPS:", camera2.get(cv2.CAP_PROP_FPS))

camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

people_count = 0
face_count = 0
recognized_name = "Unknown"
entry_count = 0
exit_count = 0
occupancy = 0
# line_y = 250
previous_people = []



def home(request):

    width = int(camera1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(camera1.get(cv2.CAP_PROP_FPS))
    snapshots =Snapshot.objects.order_by('-created_at')[:10]
    logs = CameraLog.objects.order_by('-created_at')[:10]

    context = {
        "width": width,
        "height": height,
        "fps": fps,
        "snapshots": snapshots,
        "logs": logs
    }

    return render(request, "dashboard/index.html", context)



def generate_frames(camera, camera_name):

    global people_count
    global face_count
    global recognized_name

    frame_count = 0
    recognized_name = "Unknown"

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame_count += 1

        # ==========================
        # YOLO OBJECT DETECTION
        # ==========================

        yolo_results = model(frame, imgsz=640, conf=0.25)
        phone_count = 0
        object_labels = []

        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
            result = yolo_results[0]

            for xyxy, cls_id, conf in zip(
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf
            ):
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                class_id = int(cls_id.item())
                label = result.names[class_id]
                object_labels.append(label)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                if label.lower() in ["cell phone", "phone", "mobile phone"]:
                    phone_count += 1

        phone_text = f"Phones: {phone_count}"

        # ==========================
        # FACE DETECTION
        # ==========================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50)
        )

        face_count = len(faces)
        people_count = face_count

        # only reset when face disappears
        if len(faces) == 0:
            recognized_name = "Unknown"

        # ==========================
        # FACE RECOGNITION
        # ==========================

        if len(faces) > 0 and frame_count % 15 == 0:

            try:

                x, y, w, h = faces[0]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = x1 + max(0, w)
                y2 = y1 + max(0, h)

                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    print("Recognition skipped: empty face crop", x, y, w, h)
                else:
                    print("Recognition face crop size:", face_img.shape)
                    cv2.imwrite(
                        "media/current_face.jpg",
                        face_img
                    )

                    recognized_name = recognize_face(
                        "media/current_face.jpg"
                    )
                    print("recognize_face returned:", recognized_name)

                if recognized_name != "Unknown":

                    recent = RecognitionLog.objects.filter(
                        person_name=recognized_name,
                        created_at__gte=timezone.now() -
                        timedelta(seconds=30)
                    ).exists()

                    if not recent:

                        RecognitionLog.objects.create(
                            person_name=recognized_name,
                            status="Known",
                            camera_name=camera_name
                        )

                if recognized_name != "Unknown":

                    recent = RecognitionLog.objects.filter(
                        person_name=recognized_name,
                        created_at__gte=timezone.now() -
                        timedelta(seconds=30)
                    ).exists()

                    if not recent:

                        RecognitionLog.objects.create(
                            person_name=recognized_name,
                            status="Known",
                            camera_name=camera_name
                        )

            except Exception as e:

                print("Recognition Error:", e)
                recognized_name = "Unknown"

        # ==========================
        # DRAW FACE BOX
        # ==========================

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                recognized_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        # ==========================
        # STATS
        # ==========================

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Name: {recognized_name}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            phone_text,
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # ==========================
        # JPEG ENCODE
        # ==========================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 40]
        )

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )

def video_feed_cam1(request):

    return StreamingHttpResponse(
        generate_frames(camera1, "Laptop Camera"),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def video_feed_cam2(request):

    return StreamingHttpResponse(
        generate_frames(camera2,"USB Camera"),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def occupancy_api(request):

    return JsonResponse({
        "people": people_count,
        "entry": entry_count,
        "exit": exit_count,
        "occupancy": occupancy
    })


def people_count_api(request):

    return JsonResponse({
        "count": people_count
    })

def camera_status(request):

    if camera1.isOpened():
        status = "Online"
    else:
        status = "Offline"

    return JsonResponse({
        "status": status
    })

def capture_snapshot(request):

    success, frame = camera1.read()

    if success:

        filename = datetime.now().strftime(
            "%Y%m%d_%H%M%S.jpg"
        )

        path = os.path.join(
            "media",
            "snapshots",
            filename
        )

        cv2.imwrite(path, frame)

        Snapshot.objects.create(
            image=f"snapshots/{filename}",
            camera_name="Laptop Camera"
        )

        CameraLog.objects.create(
            event="Snapshot Captured"
        )

        return JsonResponse({

            "message":
            "Snapshot Saved"

        })

    return JsonResponse({

        "message":
        "Camera Error"

    })

def test_log(request):

    camera_name = (
        request.POST.get("camera_name") or
        request.GET.get("camera_name") or
        "Unknown Camera"
    )

    RecognitionLog.objects.create(
        person_name=recognized_name,
        status="Known",
        camera_name=camera_name
    )

    return JsonResponse({
        "message": "saved"
    })



def face_count_api(request):

    return JsonResponse({
        "count": face_count
    })        