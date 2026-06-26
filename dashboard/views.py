from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import os
import time
import numpy as np
from datetime import date, datetime, timedelta
import cv2
import torch
from ultralytics import YOLO
from .face_utils import recognize_face
from .models import Snapshot, CameraLog, RecognitionLog
from .models import Visitorlogo
from .models import Alert
from .models import Attendance
from reportlab.pdfgen import canvas
from django.http import HttpResponse
from reportlab.lib.utils import ImageReader
from django.core.mail import send_mail
from django.conf import settings
from .face_utils import detect_emotion
from .face_utils import check_blacklist

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

camera_indices = {
    "camera1": 0,
    "camera2": 1,
}

camera1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera1:", camera1.isOpened())
print("Camera2:", camera2.isOpened())

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
emotion = "Unknown"
# last_occupancy_alert = None
last_camera_status = {
    "camera1": False,
    "camera2": False,
}
last_camera_status_time = {
    "camera1": None,
    "camera2": None,
}
camera_last_seen = {
    "camera1": None,
    "camera2": None,
}
camera_enabled = {
    "camera1": True,
    "camera2": True,
}
last_email_time = {}

# Mapping for assigning simple numeric IDs to recognized names
name_id_map = {}
next_person_id = 1
unknown_counter = 0
current_person_id = None

# MAX_PEOPLE = 5


def reopen_camera(camera, device_index):
    try:
        if not camera.isOpened():
            camera.open(device_index, cv2.CAP_DSHOW)
        return camera.isOpened()
    except Exception as e:
        print(f"Failed to reopen camera {device_index}: {e}")
        return False


def home(request):

    width = int(camera1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(camera1.get(cv2.CAP_PROP_FPS))
    snapshots =Snapshot.objects.order_by('-created_at')[:10]
    logs = CameraLog.objects.order_by('-created_at')[:10]
    visitor_count=Visitorlogo.objects.count()
    visitors = Visitorlogo.objects.order_by('-detection_time')[:10]
    alerts = Alert.objects.order_by('-created_at')[:10]
    attendance = Attendance.objects.order_by(
    '-entry_time'
)[:10]

    context = {
        "width": width,
        "height": height,
        "fps": fps,
        "snapshots": snapshots,
        "logs": logs,
        "visitor_count": visitor_count,
        "visitors": visitors,
        "alerts": alerts,
        "attendance": attendance,
    }

    return render(request, "dashboard/index.html", context)



def generate_frames(camera, camera_name, camera_key):

    global people_count
    global face_count
    # global recognized_name
    global last_unknown_time
    global last_recognized_time
    global last_recognized_name
    global last_occupancy_time
    global camera_last_seen
    # global emotion
    global name_id_map, next_person_id, unknown_counter
    
    frame_count = 0

    recognized_name = "Unknown"
    emotion = "Unknown"
    current_person_id = None
    
    while True:
        is_blacklisted = False

        if not camera_enabled.get(camera_key, True):
            if camera.isOpened():
                camera.release()
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Camera Stopped",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            ret, buffer = cv2.imencode(
                ".jpg",
                blank,
                [cv2.IMWRITE_JPEG_QUALITY, 40]
            )
            frame_bytes = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + frame_bytes +
                b'\r\n'
            )
            time.sleep(0.2)
            continue

        success, frame = camera.read()

        if camera_key == "camera2":
            print(
                "CAM2:",
                success,
                frame.shape if success else None
            )

        if not success or frame is None or (hasattr(frame, 'size') and frame.size == 0):
            print(f"{camera_name} read failure: success={success}, frame={None if frame is None else getattr(frame, 'shape', 'unknown')}")
            device_index = camera_indices.get(camera_key, 0)
            if reopen_camera(camera, device_index):
                print(f"Reopened {camera_name} on index {device_index}")
                continue
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
                    emotion = detect_emotion(
                        "media/current_face.jpg"
                    )

                    blacklisted_person = check_blacklist(
                        "media/current_face.jpg"
                    )
                    
                    print("BLACKLIST RESULT =", blacklisted_person)

                    if blacklisted_person:

                        is_blacklisted = True

                        alert = Alert.objects.create(
                            alert_type="Blacklisted Person",
                            message=f"{blacklisted_person} detected in {camera_name}"
                        )

                        print("ALERT ID =", alert.id)

                        print(f"BLACKLIST DETECTED: {blacklisted_person}")

                        if can_send_email(
                            f"blacklist_{blacklisted_person}"
                        ):

                            send_alert_email(
                                "🚨 BLACKLISTED PERSON DETECTED",
                                f"{blacklisted_person} detected in {camera_name}"
                            )
                        filename = datetime.now().strftime(
                            "blacklist_%Y%m%d_%H%M%S.jpg"
                        )

                        path = os.path.join(
                            "media",
                            "blacklist_events",
                            filename
                        )

                        os.makedirs(
                            "media/blacklist_events",
                            exist_ok=True
                        )

                        cv2.imwrite(
                            path,
                            frame
                        )
                    # assign or reuse a numeric id for the recognized person
                    if recognized_name != "Unknown":
                        if recognized_name not in name_id_map:
                            name_id_map[recognized_name] = next_person_id
                            next_person_id += 1
                        current_person_id = name_id_map[recognized_name]
                    else:
                        # unknowns get a running U<number>
                        unknown_counter += 1
                        current_person_id = f"U{unknown_counter}"

                    print("Emotion =", emotion)

                    today = date.today()

                    # if recognized_name != "Unknown":

                    #     Alert.objects.create(
                    #         alert_type="Known Face",
                    #         message=f"{recognized_name} detected in {camera_name}"
                    #     )

                    #     if can_send_email(f"known_{recognized_name}"):

                    #         send_alert_email(
                    #             "Known Person Detected",
                    #             f"{recognized_name} detected in {camera_name}"
                    #         )

                    already_marked = Attendance.objects.filter(
                            employee_name=recognized_name,
                            date=today
                        ).exists()

                    if not already_marked:

                            Attendance.objects.create(

                                employee_name=recognized_name,

                                status="Present"
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
                                alert_type="Unknown Face",
                                message=f"Unknown person detected in {camera_name}"
                            )

                            print("Unknown Visitor Detected")

                            if can_send_email("unknown_face"):

                                send_alert_email(
                                    "Unknown Face Detected",
                                    f"Unknown person detected in {camera_name}"
                                )

                        else:

                            Alert.objects.create(
                                alert_type="Known Face",
                                message=f"{recognized_name} detected in {camera_name}"
                            )

                            print(f"Known Visitor Detected: {recognized_name}")

                            if can_send_email(f"known_{recognized_name}"):

                                send_alert_email(
                                    "Known Person Detected",
                                    f"{recognized_name} detected in {camera_name}"
                                )

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
                        Snapshot=f"visitors/{filename}",
                        camera_name=camera_name
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
        if is_blacklisted:

            cv2.putText(
                frame,
                "BLACKLISTED PERSON",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
        for (x, y, w, h) in faces:

            if is_blacklisted:

                box_color = (0, 0, 255)
                text_color = (0, 0, 255)

            elif recognized_name != "Unknown":

                box_color = (0, 255, 0)
                text_color = (0, 255, 0)

            else:

                box_color = (0, 165, 255)
                text_color = (0, 165, 255)


            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

            # Display name and assigned ID
            id_text = f"ID: {current_person_id}" if current_person_id is not None else "ID: -"
            cv2.putText(
                frame,
                f"{recognized_name} | {id_text}",
                (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2
            )

            # Emotion
            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
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

        # cv2.putText(
        #     frame,
        #     f"Name: {recognized_name}",
        #     (10, 70),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (0, 255, 0),
        #     2
        # )

        # cv2.putText(
        #     frame,
        #     f"Emotion: {emotion}",
        #     (10, 150),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (255, 0, 255),
        #     2
        # )

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
        if not camera_enabled.get(camera_key, True):
            return False
        if not camera.isOpened():
            return False

        last_seen = camera_last_seen.get(camera_key)
        if last_seen and datetime.now() - last_seen <= timedelta(seconds=10):
            return True

        # Avoid additional reads from an already-streaming capture device.
        # Use the most recent frame timestamp from the generator to determine activity.
        return False
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

        readable_name = (
            "Camera 1"
            if camera_key == "camera1"
            else "Camera 2"
        )

        previous_status = last_camera_status.get(
            camera_key,
            False
        )

        print(
            f"{camera_key} | Previous={previous_status} | Current={is_open}"
        )

        # ==========================
        # OFFLINE ALERT
        # ==========================

        if previous_status and not is_open:

            print(f"{readable_name} OFFLINE")

            Alert.objects.create(
                alert_type="Camera Offline",
                message=f"{readable_name} Offline"
            )

            if can_send_email(
                f"{camera_key}_offline"
            ):

                send_alert_email(
                    "Camera Offline Alert",
                    f"{readable_name} is Offline."
                )

            last_camera_status_time[camera_key] = now

        # ==========================
        # ONLINE ALERT
        # ==========================

        elif not previous_status and is_open:

            print(f"{readable_name} ONLINE")

            Alert.objects.create(
                alert_type="Camera Online",
                message=f"{readable_name} Online"
            )

            if can_send_email(
                f"{camera_key}_online"
            ):

                send_alert_email(
                    "Camera Online Alert",
                    f"{readable_name} is Online again."
                )

            last_camera_status_time[camera_key] = None

        # Save latest state
        last_camera_status[camera_key] = is_open

    return JsonResponse({

        "camera1":
        "Online"
        if camera_states["camera1"]
        else "Offline",

        "camera2":
        "Online"
        if camera_states["camera2"]
        else "Offline",

        "camera1_time":
        last_camera_status_time["camera1"],

        "camera2_time":
        last_camera_status_time["camera2"]

    })
    
    
def control_camera(request):
    camera_id = request.GET.get("camera", "1")
    action = request.GET.get("action", "").lower()

    if camera_id == "1":
        camera = camera1
        camera_name = "Laptop Camera"
        camera_key = "camera1"
        device_index = camera_indices["camera1"]
    else:
        camera = camera2
        camera_name = "USB Camera"
        camera_key = "camera2"
        device_index = camera_indices["camera2"]

    if action == "start":
        camera_enabled[camera_key] = True
        if not camera.isOpened():
            success = camera.open(device_index, cv2.CAP_DSHOW)
        else:
            success = True
        if success:
            CameraLog.objects.create(event=f"{camera_name} Started")
            return JsonResponse({"message": f"{camera_name} started", "status": "started"})
        return JsonResponse({"message": f"Failed to start {camera_name}", "status": "error"}, status=500)

    if action == "stop":
        camera_enabled[camera_key] = False
        if camera.isOpened():
            camera.release()
        CameraLog.objects.create(event=f"{camera_name} Stopped")
        return JsonResponse({"message": f"{camera_name} stopped", "status": "stopped"})

    return JsonResponse({"message": "Invalid camera action", "status": "error"}, status=400)


def capture_snapshot(request):

    camera_id = request.GET.get(
        "camera",
        "1"
    )

    if camera_id == "1":

        camera = camera1
        camera_name = "Laptop Camera"

    else:

        camera = camera2
        camera_name = "USB Camera"

    success, frame = camera.read()

    if success:

        filename = datetime.now().strftime(
            "%Y%m%d_%H%M%S.jpg"
        )

        path = os.path.join(
            "media",
            "snapshots",
            filename
        )

        cv2.imwrite(
            path,
            frame
        )

        Snapshot.objects.create(
            image=f"snapshots/{filename}",
            camera_name=camera_name
        )

        CameraLog.objects.create(
            event=f"{camera_name} Snapshot Captured"
        )

        return JsonResponse({
            "message":
            f"{camera_name} Snapshot Saved"
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
     
def download_report(request):

    response = HttpResponse(
        content_type='application/pdf'
    )

    response[
        'Content-Disposition'
    ] = 'attachment; filename="Smart_CCTV_Report.pdf"'

    p = canvas.Canvas(response)

    # =========================
    # HEADER
    # =========================

    p.setFont(
        "Helvetica-Bold",
        20
    )

    p.drawString(
        130,
        800,
        "SMART CCTV REPORT"
    )

    report_id = datetime.now().strftime(
        "CCTV-%Y%m%d-%H%M"
    )

    p.setFont(
        "Helvetica",
        10
    )

    p.drawString(
        50,
        770,
        f"Report ID : {report_id}"
    )

    p.drawString(
        50,
        750,
        f"Generated : {datetime.now()}"
    )

    # =========================
    # SUMMARY
    # =========================

    total_visitors = RecognitionLog.objects.count()

    total_alerts = Alert.objects.count()

    total_snapshots = Snapshot.objects.count()

    known_faces = RecognitionLog.objects.filter(
        status="Known"
    ).count()

    unknown_faces = RecognitionLog.objects.filter(
        status="Unknown"
    ).count()

    p.setFont(
        "Helvetica-Bold",
        14
    )

    p.drawString(
        50,
        710,
        "System Statistics"
    )

    p.setFont(
        "Helvetica",
        10
    )

    p.drawString(
        50,
        690,
        f"Total Visitors : {total_visitors}"
    )

    p.drawString(
        50,
        675,
        f"Known Faces : {known_faces}"
    )

    p.drawString(
        50,
        660,
        f"Unknown Faces : {unknown_faces}"
    )

    p.drawString(
        50,
        645,
        f"Total Alerts : {total_alerts}"
    )

    p.drawString(
        50,
        630,
        f"Total Snapshots : {total_snapshots}"
    )

    # =========================
    # CAMERA STATUS
    # =========================

    p.setFont(
        "Helvetica-Bold",
        14
    )

    p.drawString(
        50,
        600,
        "Camera Status"
    )

    p.setFont(
        "Helvetica",
        10
    )

    p.drawString(
        50,
        580,
        "Camera 1 : Online"
    )

    p.drawString(
        50,
        565,
        "Camera 2 : Offline"
    )

    # =========================
    # CAMERA LOGS
    # =========================

    y = 530

    p.setFont(
        "Helvetica-Bold",
        14
    )

    p.drawString(
        50,
        y,
        "Camera Activity Logs"
    )

    y -= 20

    logs = CameraLog.objects.order_by(
        '-created_at'
    )[:5]

    p.setFont(
        "Helvetica",
        9
    )

    for log in logs:

        p.drawString(
            50,
            y,
            f"{log.event}"
        )

        y -= 15

    # =========================
    # VISITOR HISTORY
    # =========================

    y -= 20

    p.setFont(
        "Helvetica-Bold",
        14
    )

    p.drawString(
        50,
        y,
        "Visitor History"
    )

    y -= 20

    visitors = RecognitionLog.objects.order_by(
        '-detection_time'
    )[:5]

    p.setFont(
        "Helvetica",
        9
    )

    for visitor in visitors:

        p.drawString(
            50,
            y,
            f"{visitor.person_name} | {visitor.status}"
        )

        y -= 15

    # =========================
    # ATTENDANCE
    # =========================

    y -= 20

    p.setFont(
        "Helvetica-Bold",
        14
    )

    p.drawString(
        50,
        y,
        "Attendance Report"
    )

    y -= 20

    attendance = Attendance.objects.order_by(
        '-entry_time'
    )[:5]

    p.setFont(
        "Helvetica",
        9
    )

    for row in attendance:

        p.drawString(
            50,
            y,
            f"{row.employee_name} | {row.status}"
        )

        y -= 15

    # =========================
    # NEW PAGE
    # =========================

    p.showPage()

    p.setFont(
        "Helvetica-Bold",
        18
    )

    p.drawString(
        180,
        800,
        "Alert History"
    )

    y = 760

    alerts = Alert.objects.order_by(
        '-created_at'
    )[:15]

    p.setFont(
        "Helvetica",
        10
    )

    for alert in alerts:

        p.drawString(
            50,
            y,
            f"{alert.alert_type} | {alert.message}"
        )

        y -= 20

    # =========================
    # SNAPSHOTS PAGE
    # =========================

    p.showPage()

    p.setFont(
        "Helvetica-Bold",
        18
    )

    p.drawString(
        180,
        800,
        "Recent Snapshots"
    )

    snapshots = Snapshot.objects.order_by(
        '-created_at'
    )[:4]

    x = 50
    y = 600

    for snap in snapshots:

        try:

            p.drawImage(
                ImageReader(
                    snap.image.path
                ),
                x,
                y,
                width=200,
                height=150
            )

            x += 250

            if x > 300:

                x = 50
                y -= 200

        except:
            pass

    # =========================
    # FOOTER
    # =========================

    p.setFont(
        "Helvetica",
        8
    )

    p.drawString(
        180,
        20,
        "Generated By Smart CCTV Monitoring System"
    )

    p.save()

    return response




def send_alert_email(subject, message):

    send_mail(
        subject,
        message,
        settings.EMAIL_HOST_USER,
        ['daxp1704@gmail.com'],
        fail_silently=False
    )


def can_send_email(alert_type):

    now = datetime.now()

    if alert_type not in last_email_time:
        last_email_time[alert_type] = now
        return True

    diff = (now - last_email_time[alert_type]).seconds

    if diff > 60:
        last_email_time[alert_type] = now
        return True

    return False



def test_email(request):

    send_alert_email(
        "Smart CCTV Test",
        "Email system working successfully."
    )

    return JsonResponse({
        "message": "Email Sent"
    })



