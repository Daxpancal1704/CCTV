from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse

import os
from datetime import datetime
import cv2
from ultralytics import YOLO

from .face_utils import recognize_face
from .models import Snapshot, CameraLog, RecognitionLog

model = YOLO("yolov8n.pt")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

camera1 = cv2.VideoCapture(0)
camera2 = cv2.VideoCapture(1)

camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

people_count = 0
face_count = 0
recognized_name = "Unknown"

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


def generate_frames(camera):

    global people_count
    global face_count
    global recognized_name

    while True:

        
        success, frame = camera.read()


        if not success:
            break

        people_count = 0

        # YOLO

        results = model(
            frame,
            imgsz=640,
            verbose=False
        )

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.5:

                    people_count += 1

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Person {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

        # Face Detection

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

        if len(faces) > 0:

            cv2.imwrite(
                "media/current_frame.jpg",
                frame
            )

            try:

                recognized_name = recognize_face(
                    "media/current_frame.jpg"
                )

            except Exception:

                recognized_name = "Unknown"

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

        cv2.putText(
            frame,
            f"People: {people_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )

def video_feed_cam1(request):

    return StreamingHttpResponse(
        generate_frames(camera1),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def video_feed_cam2(request):

    return StreamingHttpResponse(
        generate_frames(camera2),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

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
            image=f"snapshots/{filename}"
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

    RecognitionLog.objects.create(
        person_name="Dax Panchal",
        status="Known"
    )

    return JsonResponse({
        "message": "saved"
    })



def face_count_api(request):

    return JsonResponse({
        "count": face_count
    })        