from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import os
import time
import numpy as np
from datetime import date, datetime, timedelta
import cv2
import torch
from ultralytics import YOLO
from .face_utils import recognize_face
from .models import Snapshot, CameraLog, RecognitionLog, Employee
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
from .models import Camera, Employee, BlacklistPerson

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

active_cameras = {}
camera_indices = {}

failed_cameras_cooldown = {}

def get_camera_instance(camera_id):
    """Returns a cv2 VideoCapture instance for a given camera ID, initializing if needed."""
    global active_cameras, camera_enabled, camera_last_seen, failed_cameras_cooldown
    try:
        cam_obj = Camera.objects.get(id=camera_id)
    except Camera.DoesNotExist:
        return None
        
    cam_key = f"camera_{camera_id}"
    camera_enabled.setdefault(cam_key, cam_obj.is_enabled)
    
    if cam_key not in active_cameras or not active_cameras[cam_key].isOpened():
        now = datetime.now()
        # Prevent aggressive retries that can crash OpenCV DSHOW backend
        if cam_key in failed_cameras_cooldown:
            if now - failed_cameras_cooldown[cam_key] < timedelta(seconds=15):
                return None
                
        source = cam_obj.source_url
        if not source:
            source = cam_obj.camera_no - 1
        elif source.isdigit():
            source = int(source)
            
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if isinstance(source, int) else cv2.CAP_ANY)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                active_cameras[cam_key] = cap
                camera_indices[cam_key] = source
                if cam_key in failed_cameras_cooldown:
                    del failed_cameras_cooldown[cam_key]
            else:
                failed_cameras_cooldown[cam_key] = now
                return None
        except Exception as e:
            print(f"Error initializing camera {source}: {e}")
            failed_cameras_cooldown[cam_key] = now
            return None
            
    return active_cameras[cam_key]

people_count = 0
face_count = 0
recognized_name = "Unknown"
entry_count = 0
exit_count = 0
occupancy = 0
previous_people = []
last_unknown_time = None
last_recognized_time = None
last_recognized_name = None
last_occupancy_time = None
emotion = "Unknown"
last_camera_status = {}
last_camera_status_time = {}
camera_last_seen = {}
camera_enabled = {}
last_email_time = {}

name_id_map = {}
next_person_id = 1
unknown_counter = 0
current_person_id = None


def reopen_camera(camera, device_index):
    try:
        if not camera.isOpened():
            if isinstance(device_index, int):
                camera.open(device_index, cv2.CAP_DSHOW)
            else:
                camera.open(device_index, cv2.CAP_ANY)
        return camera.isOpened()
    except Exception as e:
        print(f"Failed to reopen camera {device_index}: {e}")
        return False


@login_required
def home(request):
    cameras = Camera.objects.all().order_by("camera_no")
    
    width, height, fps = 640, 480, 30
    
    snapshots = Snapshot.objects.order_by('-created_at')[:10]
    logs = CameraLog.objects.order_by('-created_at')[:10]
    visitor_count = Visitorlogo.objects.filter(detection_time__date=date.today()).count()
    visitors = Visitorlogo.objects.order_by('-detection_time')[:10]
    alerts = Alert.objects.order_by('-created_at')[:10]
    attendance = Attendance.objects.order_by('-entry_time')[:10]

    # Camera online/offline counts for doughnut chart
    online_count = 0
    for cam in cameras:
        cam_key = f"camera_{cam.id}"
        if camera_is_active(cam.id, cam_key):
            online_count += 1
            
    total_cams = cameras.count()
    offline_count = max(0, total_cams - online_count)

    # Known vs Unknown face counts for doughnut chart
    known_count = Visitorlogo.objects.exclude(visitor_name="Unknown").count()
    unknown_count = Visitorlogo.objects.filter(visitor_name="Unknown").count()

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
        "cameras": cameras,
        "online": online_count,
        "offline": offline_count,
        "known_count": known_count,
        "unknown_count": unknown_count,
    }

    return render(request, "dashboard/index.html", context)

@login_required
def live_monitoring(request):
    cameras = Camera.objects.filter(is_enabled=True).order_by("camera_no")
    width, height, fps = 640, 480, 30
    context = {
        "cameras": cameras,
        "width": width,
        "height": height,
        "fps": fps,
    }
    return render(request, "dashboard/live_monitoring.html", context)

@login_required
def cameras(request):
    cameras = Camera.objects.all().order_by("camera_no")
    for cam in cameras:
        cam_key = f"camera_{cam.id}"
        cam_instance = active_cameras.get(cam_key)
        cam.is_online = camera_is_active(cam_instance, cam_key) if cam_instance else False
        
    context = {"cameras": cameras}
    return render(request, "dashboard/cameras.html", context)

@login_required
def delete_camera(request, camera_id):
    if request.method == "POST":
        try:
            cam = Camera.objects.get(id=camera_id)
            cam_key = f"camera_{camera_id}"
            
            # release camera if open
            if cam_key in active_cameras:
                if active_cameras[cam_key].isOpened():
                    active_cameras[cam_key].release()
                del active_cameras[cam_key]
                
            cam.delete()
            return JsonResponse({"status": "success", "message": "Camera deleted successfully!"})
        except Camera.DoesNotExist:
            return JsonResponse({"status": "error", "message": "Camera not found"}, status=404)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    return JsonResponse({"status": "error", "message": "Invalid request method."}, status=400)

@login_required
def add_camera(request):
    if request.method == "POST":
        name = request.POST.get("name")
        url = request.POST.get("url", "")
        cam_type = request.POST.get("type", "Webcam")
        
        if not name:
            return JsonResponse({"status": "error", "message": "Camera name is required."}, status=400)
            
        try:
            # Determine next camera number
            last_cam = Camera.objects.order_by('-camera_no').first()
            next_no = (last_cam.camera_no + 1) if last_cam else 1
            
            Camera.objects.create(
                name=name,
                source_url=url,
                camera_type=cam_type,
                camera_no=next_no,
                is_enabled=True
            )
            return JsonResponse({"status": "success", "message": "Camera added successfully!"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
            
    return JsonResponse({"status": "error", "message": "Invalid request method."}, status=400)

@login_required
def ai_events(request):
    alerts = Alert.objects.order_by('-created_at')[:50]
    logs = CameraLog.objects.order_by('-created_at')[:50]
    context = {"alerts": alerts, "logs": logs}
    return render(request, "dashboard/ai_events.html", context)

@login_required
def attendance_view(request):
    attendance = Attendance.objects.all()
    
    status = request.GET.get('status')
    if status:
        attendance = attendance.filter(status__iexact=status)
        
    date_filter = request.GET.get('date')
    if date_filter:
        attendance = attendance.filter(date=date_filter)
        
    attendance = attendance.order_by('-date', '-entry_time')[:50]
    context = {
        "attendance": attendance,
        "current_status": status,
        "current_date": date_filter
    }
    return render(request, "dashboard/attendance.html", context)

@login_required
def export_attendance_csv(request):
    import csv
    from django.http import HttpResponse

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="attendance_records.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Employee / Person', 'Date', 'Entry Time', 'Exit Time', 'Status'])
    
    attendance = Attendance.objects.all()
    
    status = request.GET.get('status')
    if status:
        attendance = attendance.filter(status__iexact=status)
        
    date_filter = request.GET.get('date')
    if date_filter:
        attendance = attendance.filter(date=date_filter)
        
    attendance = attendance.order_by('-date', '-entry_time')
    
    for row in attendance:
        entry = row.entry_time.strftime("%H:%M:%S") if row.entry_time else "--"
        exit = row.exit_time.strftime("%H:%M:%S") if row.exit_time else "--"
        writer.writerow([row.employee_name, row.date, entry, exit, row.status])
        
    return response

@login_required
def visitors_view(request):
    q = request.GET.get('q', '')
    if q:
        visitors = Visitorlogo.objects.filter(visitor_name__icontains=q).order_by('-detection_time')[:50]
    else:
        visitors = Visitorlogo.objects.order_by('-detection_time')[:50]
    context = {"visitors": visitors}
    return render(request, "dashboard/visitors.html", context)

@login_required
def snapshots_view(request):
    snapshots = Snapshot.objects.order_by('-created_at')[:50]
    context = {"snapshots": snapshots}
    return render(request, "dashboard/snapshots.html", context)

@login_required
def logs_view(request):
    logs = CameraLog.objects.order_by('-created_at')[:50]
    context = {"logs": logs}
    return render(request, "dashboard/logs.html", context)

@login_required
def settings_view(request):
    return render(request, "dashboard/settings.html", {})

@login_required
def profile_view(request):
    if request.method == 'POST':
        user = request.user
        user.first_name = request.POST.get('first_name', user.first_name)
        user.last_name = request.POST.get('last_name', user.last_name)
        user.email = request.POST.get('email', user.email)
        user.save()
        from django.contrib import messages
        messages.success(request, 'Profile updated successfully!')
        from django.shortcuts import redirect
        return redirect('profile')
    return render(request, "dashboard/profile.html")

@login_required
def alerts_page(request):
    alerts = Alert.objects.order_by('-created_at')[:100]
    context = {"alerts": alerts}
    return render(request, "dashboard/alerts_page.html", context)

@login_required
def export_alerts_csv(request):
    import csv
    from django.http import HttpResponse

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="system_alerts.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Severity', 'Alert Type', 'Message', 'Timestamp'])
    
    alerts = Alert.objects.all().order_by('-created_at')
    
    for alert in alerts:
        # Determine severity based on logic in template
        severity = "Info"
        if alert.alert_type in ["Intrusion", "Blacklisted Person"]:
            severity = "Critical"
        elif alert.alert_type == "Unknown Face":
            severity = "Warning"
            
        timestamp = f" {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}" if alert.created_at else "--"
        writer.writerow([severity, alert.alert_type, alert.message, timestamp])
        
    return response

@login_required
def export_visitors_csv(request):
    import csv
    from django.http import HttpResponse

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="visitor_logs.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Visitor Name', 'Camera', 'Detection Time'])
    
    visitors = Visitorlogo.objects.all().order_by('-detection_time')
    
    for visitor in visitors:
        timestamp = f" {visitor.detection_time.strftime('%Y-%m-%d %H:%M:%S')}" if visitor.detection_time else "--"
        writer.writerow([visitor.visitor_name, visitor.camera_name, timestamp])
        
    return response

@login_required
def reports_view(request):
    return render(request, "dashboard/reports.html")

import base64
from django.core.files.base import ContentFile
from django.contrib import messages
from django.shortcuts import render, redirect

@login_required
def face_register(request):
    if request.method == "POST":
        name = request.POST.get("person_name")
        role = request.POST.get("person_role")
        gender = request.POST.get("person_gender")
        
        employee = Employee(name=name, role=role, gender=gender)
        
        # Helper function to process each angle
        def process_image(angle):
            base64_data = request.POST.get(f"base64_{angle}")
            file_data = request.FILES.get(f"image_{angle}")
            
            if base64_data:
                # format: "data:image/jpeg;base64,/9j/4AAQ..."
                fmt, imgstr = base64_data.split(';base64,') 
                ext = fmt.split('/')[-1]
                data = ContentFile(base64.b64decode(imgstr), name=f"{name}_{angle}.{ext}")
                setattr(employee, f"image_{angle}", data)
            elif file_data:
                setattr(employee, f"image_{angle}", file_data)

        process_image("front")
        process_image("right")
        process_image("left")
        
        employee.save()
        messages.success(request, f"Successfully registered {name}!")
        return redirect("face_register")

    registered_faces = Employee.objects.all().order_by('-id')
    return render(request, "dashboard/face_register.html", {"registered_faces": registered_faces})

@login_required
@require_http_methods(["POST"])
def delete_face(request, pk):
    try:
        employee = Employee.objects.get(id=pk)
        name = employee.name
        employee.delete()
        messages.success(request, f"Successfully deleted {name}.")
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
    return redirect("face_register")

@login_required
def wanted_persons(request):
    if request.method == "POST":
        name = request.POST.get("wanted_name")
        image = request.FILES.get("wanted_image")
        reason = request.POST.get("wanted_reason", "")

        if name and image:
            BlacklistPerson.objects.create(name=name, image=image)
            messages.success(request, f"'{name}' has been added to the watchlist. You will be alerted when detected on any CCTV.")
        else:
            messages.error(request, "Please provide both a name and an image.")

        return redirect("wanted_persons")

    persons = BlacklistPerson.objects.all().order_by('-added_at')
    return render(request, "dashboard/wanted_persons.html", {"wanted_persons": persons})

@login_required
@require_http_methods(["POST"])
def delete_wanted(request, pk):
    try:
        person = BlacklistPerson.objects.get(id=pk)
        name = person.name
        if person.image:
            person.image.delete(save=False)
        person.delete()
        messages.success(request, f"'{name}' has been removed from the watchlist.")
    except BlacklistPerson.DoesNotExist:
        messages.error(request, "Person not found.")
    return redirect("wanted_persons")

snapshot_requests = {
    "camera1": False,
    "camera2": False,
}

@login_required
def capture_snapshot(request):
    camera_id = request.GET.get('camera', '1')
    cam_key = f"camera_{camera_id}"
    global snapshot_requests
    snapshot_requests[cam_key] = True
    return JsonResponse({"status": "success", "message": f"Snapshot requested for camera {camera_id}."})

@login_required
def download_snapshot(request, pk):
    try:
        from django.shortcuts import get_object_or_404
        from django.http import HttpResponse, JsonResponse
        import io
        from PIL import Image

        snapshot = get_object_or_404(Snapshot, pk=pk)
        fmt = request.GET.get('format', 'jpg').lower()
        
        # Determine content type and Pillow format
        if fmt == 'png':
            content_type = 'image/png'
            pil_format = 'PNG'
        elif fmt == 'pdf':
            content_type = 'application/pdf'
            pil_format = 'PDF'
        elif fmt == 'bmp':
            content_type = 'image/bmp'
            pil_format = 'BMP'
        else:
            fmt = 'jpg'
            content_type = 'image/jpeg'
            pil_format = 'JPEG'
            
        with Image.open(snapshot.image.path) as img:
            # Ensure RGB if needed
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            buffer = io.BytesIO()
            img.save(buffer, format=pil_format)
            
            response = HttpResponse(buffer.getvalue(), content_type=content_type)
            filename = f"snapshot_{snapshot.id}.{fmt}"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)


@login_required
def delete_snapshot(request, pk):
    from django.shortcuts import get_object_or_404, redirect
    from django.contrib import messages
    if request.method == 'POST':
        snapshot = get_object_or_404(Snapshot, pk=pk)
        if snapshot.image:
            snapshot.image.delete(save=False)
        snapshot.delete()
    return redirect('snapshots')

def generate_frames(camera_id, camera_name, camera_key):

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
            camera = active_cameras.get(camera_key)
            if camera is not None and camera.isOpened():
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

        camera = active_cameras.get(camera_key)
        if camera is None or not camera.isOpened():
            camera = get_camera_instance(camera_id)
            
        success, frame = False, None
        if camera is not None and camera.isOpened():
            success, frame = camera.read()

        if camera_key == "camera2":
            print(
                "CAM2:",
                success,
                frame.shape if success else None
            )

        if not success or frame is None or (hasattr(frame, 'size') and frame.size == 0):
            print(f"{camera_name} read failure: success={success}, frame={None if frame is None else getattr(frame, 'shape', 'unknown')}")
            
            if camera is not None:
                camera.release()
            if camera_key in active_cameras:
                del active_cameras[camera_key]
                
            # Camera not available – yield a placeholder and keep retrying
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (30, 30, 30)  # dark background
            cv2.putText(
                placeholder,
                "Camera Not Available",
                (100, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 100, 255),
                2,
            )
            cv2.putText(
                placeholder,
                "USB Camera Not Connected",
                (120, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                1,
            )
            cv2.putText(
                placeholder,
                "Retrying...",
                (240, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 100),
                1,
            )
            _ret, _buf = cv2.imencode(
                ".jpg",
                placeholder,
                [cv2.IMWRITE_JPEG_QUALITY, 50]
            )
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + _buf.tobytes() +
                b'\r\n'
            )
            time.sleep(2)  # wait before retrying
            continue

        camera_last_seen[camera_key] = datetime.now()
        frame_count += 1

        # ==========================
        # SNAPSHOT HANDLING
        # ==========================
        if snapshot_requests.get(camera_key):
            try:
                now = datetime.now()
                filename = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
                snap_folder = os.path.join("media", "snapshots")
                os.makedirs(snap_folder, exist_ok=True)
                snap_path = os.path.join(snap_folder, filename)
                
                cv2.imwrite(snap_path, frame)
                
                Snapshot.objects.create(
                    image=f"snapshots/{filename}",
                    camera_name=camera_name
                )
                print(f"Snapshot saved for {camera_name}")
            except Exception as e:
                print(f"Error saving snapshot: {e}")
            
            snapshot_requests[camera_key] = False

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
        # if len(faces) > 0 and frame_count % 15 == 0: