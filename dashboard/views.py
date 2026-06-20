from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import os
from datetime import datetime, timedelta, timezone
import cv2
import torch
from ultralytics import YOLO
from .face_utils import recognize_face
from .models import Snapshot, CameraLog, RecognitionLog
from .models import Visitorlogo
from .models import Alert

model = YOLO("yolov8n.pt")
if torch.cuda.is_available():
    model.to("cuda")
    print("GPU Enabled")
else:
    print("CPU Mode")

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

camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

camera1.set(cv2.CAP_PROP_FPS, 30)
camera2.set(cv2.CAP_PROP_FPS, 30)

print("Cam1 FPS:", camera1.get(cv2.CAP_PROP_FPS))
print("Cam2 FPS:", camera2.get(cv2.CAP_PROP_FPS))



people_count = 0
face_count = 0
recognized_name = "Unknown"
entry_count = 0
exit_count = 0
occupancy = 0
# line_y = 250
previous_people = []
last_unknown_time = None
last_recognized_time = None
last_recognized_name = None
last_occupancy_time = None
last_camera_status = {
    "camera1": True,
    "camera2": True,
}
last_camera_status_time = {
    "camera1": None,
    "camera2": None,
}
camera_last_seen = {
    "camera1": None,
    "camera2": None,
}



def home(request):

    width = int(camera1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(camera1.get(cv2.CAP_PROP_FPS))
    snapshots =Snapshot.objects.order_by('-created_at')[:10]
    logs = CameraLog.objects.order_by('-created_at')[:10]
    visitor_count=Visitorlogo.objects.count()
    visitors = Visitorlogo.objects.order_by('-detection_time')[:10]
    alerts = Alert.objects.order_by('-created_at')[:10]

    context = {
        "width": width,
        "height": height,
        "fps": fps,
        "snapshots": snapshots,
        "logs": logs,
        "visitor_count": visitor_count,
        "visitors": visitors,
        "alerts": alerts
    }

    return render(request, "dashboard/index.html", context)



def generate_frames(camera, camera_name, camera_key):

    global people_count
    global face_count
    global recognized_name
    global last_unknown_time
    global last_recognized_time
    global last_recognized_name
    global last_occupancy_time
    global camera_last_seen

    frame_count = 0
    recognized_name = "Unknown"

    while True:

        success, frame = camera.read()

        if not success:
            break

        camera_last_seen[camera_key] = datetime.now()
        frame_count += 1

        # ==========================
        # YOLO OBJECT DETECTION
        # ==========================

        yolo_results = model(frame, 
                             imgsz=320, 
                             conf=0.25,
                             verbose=False)
        phone_count = 0
        object_labels = []
        people_count = 0

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
                if label.lower() == "person":
                    people_count += 1
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
        # people_count = face_count

        # ==========================
        # OCCUPANCY ALERT
        # ==========================

        MAX_PEOPLE = 2

        if people_count > MAX_PEOPLE:

            now = datetime.now()

            if (
                last_occupancy_time is None
                or
                now - last_occupancy_time > timedelta(seconds=60)
            ):

                Alert.objects.create(

                    alert_type="Occupancy Alert",

                    message=f"{people_count} people detected in {camera_name}"

                )

                print(
                    f"Occupancy Alert: {people_count} people"
                )

                last_occupancy_time = now

        # only reset when face disappears
        if len(faces) == 0:
            recognized_name = "Unknown"

        # ==========================
        # FACE RECOGNITION
        # ==========================
        if len(faces) > 0 and frame_count % 15 == 0:

            print("Recognition Block Entered")

            try:

                x, y, w, h = faces[0]

                x1 = max(0, x)
                y1 = max(0, y)

                x2 = x1 + w
                y2 = y1 + h

                face_img = frame[y1:y2, x1:x2]

                face_img = cv2.resize(
                    face_img,
                    (224, 224)
                )

                if face_img.size > 0:

                    cv2.imwrite(
                        "media/current_face.jpg",
                        face_img
                    )

                    print("Recognition Started")

                    recognized_name = recognize_face(
                        "media/current_face.jpg"
                    )

                    print("Recognized =", recognized_name)

                    now = datetime.now()
                    should_save_visitor = False
                    visitor_name = recognized_name
                    alert_type = None
                    alert_message = None

                    if recognized_name == "Unknown":
                        alert_type = "Unknown Face"
                        alert_message = "Unknown person detected"
                        should_save_visitor = (
                            last_unknown_time is None or
                            now - last_unknown_time > timedelta(seconds=30)
                        )
                    else:
                        if (
                            last_recognized_name != recognized_name or
                            last_recognized_time is None or
                            now - last_recognized_time > timedelta(seconds=30)
                        ):
                            should_save_visitor = True

                    if should_save_visitor:
                        if alert_type is not None:
                            Alert.objects.create(
                                alert_type=alert_type,
                                message=alert_message
                            )
                            print("Unknown Visitor Detected")
                        else:
                            Alert.objects.create(
                                alert_type="Known Face",
                                message=f"{recognized_name} detected"
                            )
                            print(f"Known Visitor Detected: {recognized_name}")

                        filename = now.strftime(
                            "%Y%m%d_%H%M%S"
                        ) + ".jpg"

                        visitor_folder = os.path.join(
                            "media",
                            "visitors"
                        )

                        os.makedirs(
                            visitor_folder,
                            exist_ok=True
                        )

                        visitor_path = os.path.join(
                            visitor_folder,
                            filename
                        )

                        cv2.imwrite(
                            visitor_path,
                            frame
                        )

                        visitor = Visitorlogo.objects.create(
                            visitor_name=visitor_name,
                            Snapshot=f"visitors/{filename}"
                        )

                        print(
                            "Visitor Saved:",
                            visitor.id
                        )

                        if recognized_name == "Unknown":
                            last_unknown_time = now
                        else:
                            last_recognized_time = now
                            last_recognized_name = recognized_name

                        

            except Exception as e:

                print("Recognition Error:", e)

                recognized_name = "Unknown"
        # ==========================
        # DRAW FACE BOX
        # ==========================

        for (x, y, w, h) in faces:


            print(
                "Current Name =",
                recognized_name
            )

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
        generate_frames(camera1, "Laptop Camera", "camera1"),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def video_feed_cam2(request):

    return StreamingHttpResponse(
        generate_frames(camera2, "USB Camera", "camera2"),
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

def camera_is_active(camera, camera_key):
    try:
        if not camera.isOpened():
            return False

        last_seen = camera_last_seen.get(camera_key)
        if last_seen and datetime.now() - last_seen <= timedelta(seconds=10):
            return True

        success, frame = camera.read()
        if not success or frame is None:
            return False

        if hasattr(frame, 'size') and frame.size == 0:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()
        std_val = gray.std()

        if mean_val < 15 and std_val < 10:
            return False

        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        if abs(sobel).mean() < 5:
            return False

        return True
    except Exception:
        return False


def camera_status(request):

    global last_camera_status
    global last_camera_status_time

    camera_states = {
        "camera1": camera_is_active(camera1, "camera1"),
        "camera2": camera_is_active(camera2, "camera2"),
    }

    now = datetime.now().strftime("%I:%M %p")

    for camera_key, is_open in camera_states.items():
        readable_name = "Camera 1" if camera_key == "camera1" else "Camera 2"

        if not is_open and last_camera_status.get(camera_key, True):
            Alert.objects.create(
                alert_type="Camera Offline",
                message=f"{readable_name} Offline"
            )
            last_camera_status_time[camera_key] = now
        elif is_open and not last_camera_status.get(camera_key, True):
            Alert.objects.create(
                alert_type="Camera Online",
                message=f"{readable_name} Online"
            )
            last_camera_status_time[camera_key] = None

    for camera_key, is_open in camera_states.items():
        if not is_open and last_camera_status_time[camera_key] is None:
            last_camera_status_time[camera_key] = now

    last_camera_status.update(camera_states)

    return JsonResponse({
        "camera1": "Online" if camera_states["camera1"] else "Offline",
        "camera2": "Online" if camera_states["camera2"] else "Offline",
        "camera1_time": last_camera_status_time["camera1"],
        "camera2_time": last_camera_status_time["camera2"],
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


def alerts_api(request):
    alerts = Alert.objects.order_by('-created_at')[:20]
    alert_list = [
        {
            "alert_type": alert.alert_type,
            "message": alert.message,
            "created_at": alert.created_at.strftime("%B %d, %Y, %I:%M %p")
        }
        for alert in alerts
    ]
    return JsonResponse({
        "alerts": alert_list
    })        
