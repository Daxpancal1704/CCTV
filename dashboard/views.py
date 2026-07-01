from django.shortcuts import render
print("Views module loaded successfully.")
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
import torch.nn as nn

# --- PATCH FOR ULTRALYTICS YOLO FUSION BUG ---
# Prevents "AttributeError: 'Conv' object has no attribute 'bn'"
original_delattr = nn.Module.__delattr__
def safe_delattr(self, name):
    if name == 'bn' and not hasattr(self, name):
        return
    original_delattr(self, name)
nn.Module.__delattr__ = safe_delattr
# ---------------------------------------------

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
from .models import Camera, Employee, BlacklistPerson, VisitorAnalytics

model = YOLO("yolov8n.pt")
if torch.cuda.is_available():
    model.to("cuda")
    print("GPU Enabled")
else:
    print("CPU Mode")

def check_role(user, allowed_roles):
    if user.is_superuser:
        return True
    if hasattr(user, 'profile'):
        return user.profile.role in allowed_roles
    return False

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
total_events = 0
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


    try:
        laptop_cam = Camera.objects.filter(name__icontains="leptop").first()
        if laptop_cam:
            if not laptop_cam.is_enabled:
                laptop_cam.is_enabled = True
                laptop_cam.save()
            if laptop_cam.source_url == "0" and laptop_cam.camera_type != "Webcam":
                laptop_cam.camera_type = "Webcam"
                laptop_cam.save()
    except Exception as e:
        print("Error updating laptop camera:", e)

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
        cam_instance = active_cameras.get(cam_key)
        if camera_is_active(cam_instance, cam_key) if cam_instance else False:
            online_count += 1
            cam.is_online = True
        else:
            cam.is_online = False
            
    total_cams = cameras.count()
    offline_count = max(0, total_cams - online_count)

    # Known vs Unknown face counts for doughnut chart
    known_count = Visitorlogo.objects.exclude(visitor_name="Unknown").count()
    unknown_count = Visitorlogo.objects.filter(visitor_name="Unknown").count()
    unknown_faces_list = Visitorlogo.objects.filter(visitor_name="Unknown").order_by('-detection_time')[:4]
    
    # Calculate today's stats
    today = date.today()
    yesterday = today - timedelta(days=1)
    
    def get_change(today_count, yesterday_count):
        if yesterday_count == 0:
            return 100.0 if today_count > 0 else 0.0
        return round(((today_count - yesterday_count) / yesterday_count) * 100, 1)

    entries_count = Attendance.objects.filter(date=today).count()
    entries_yesterday = Attendance.objects.filter(date=yesterday).count()
    entries_change = get_change(entries_count, entries_yesterday)
    
    exits_count = Attendance.objects.filter(date=today, exit_time__isnull=False).count()
    exits_yesterday = Attendance.objects.filter(date=yesterday, exit_time__isnull=False).count()
    exits_change = get_change(exits_count, exits_yesterday)
    
    unknown_today = Visitorlogo.objects.filter(visitor_name="Unknown", detection_time__date=today).count()
    unknown_yesterday = Visitorlogo.objects.filter(visitor_name="Unknown", detection_time__date=yesterday).count()
    unknown_change = get_change(unknown_today, unknown_yesterday)
    
    visitor_count = Visitorlogo.objects.filter(detection_time__date=today).count()
    visitor_yesterday = Visitorlogo.objects.filter(detection_time__date=yesterday).count()
    visitor_change = get_change(visitor_count, visitor_yesterday)
    
    total_employees = Employee.objects.count()
    emp_present = Attendance.objects.filter(date=today, status__iexact='Present').count()
    emp_absent = Attendance.objects.filter(date=today, status__iexact='Absent').count()
    emp_leave = Attendance.objects.filter(date=today, status__iexact='Leave').count()
    
    ai_events_count = CameraLog.objects.filter(created_at__date=today).count()
    recent_alerts_count = Alert.objects.filter(created_at__date=today).count()

    # Events Summary
    face_detections_count = Visitorlogo.objects.filter(detection_time__date=today).count()
    motion_events_count = Alert.objects.filter(created_at__date=today, alert_type__icontains="Motion").count()
    crowd_events_count = Alert.objects.filter(created_at__date=today, alert_type__icontains="Crowd").count()
    other_events_count = Alert.objects.filter(created_at__date=today).exclude(alert_type__icontains="Motion").exclude(alert_type__icontains="Crowd").count()
    
    # Storage Overview
    import shutil
    try:
        total, used, free = shutil.disk_usage("/")
        def format_size(bytes_val):
            gb = bytes_val / (1024 ** 3)
            if gb > 1000:
                return f"{gb/1024:.1f} TB"
            return f"{int(gb)} GB"
        storage_total = format_size(total)
        storage_used = format_size(used)
        storage_free = format_size(free)
        storage_percent = int((used / total) * 100) if total > 0 else 0
    except Exception:
        storage_total, storage_used, storage_free, storage_percent = "1.5 TB", "892 GB", "608 GB", 62

    # Detection Trend (Last 7 Days)
    import json
    trend_labels = []
    trend_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        trend_labels.append(day.strftime('%d %b'))
        trend_data.append(Visitorlogo.objects.filter(detection_time__date=day).count())
    trend_labels_json = json.dumps(trend_labels)
    trend_data_json = json.dumps(trend_data)

    # Calculate Attendance Present Percentage
    attendance_present_percentage = 0
    if total_employees > 0:
        attendance_present_percentage = int((emp_present / total_employees) * 100)

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
        "unknown_faces_list": unknown_faces_list,
        "entries_count": entries_count,
        "entries_change": entries_change,
        "exits_count": exits_count,
        "exits_change": exits_change,
        "total_employees": total_employees,
        "emp_present": emp_present,
        "emp_absent": emp_absent,
        "emp_leave": emp_leave,
        "attendance_present_percentage": attendance_present_percentage,
        "ai_events_count": ai_events_count,
        "recent_alerts_count": recent_alerts_count,
        "face_detections_count": face_detections_count,
        "visitor_change": visitor_change,
        "unknown_change": unknown_change,
        "motion_events_count": motion_events_count,
        "crowd_events_count": crowd_events_count,
        "other_events_count": other_events_count,
        "storage_total": storage_total,
        "storage_used": storage_used,
        "storage_free": storage_free,
        "storage_percent": storage_percent,
        "trend_labels": trend_labels_json,
        "trend_data": trend_data_json,
    }

    return render(request, "dashboard/index.html", context)

@login_required
def live_monitoring(request):
    if not check_role(request.user, ['Super admin', 'Admin', 'Security Officer', 'Operator', 'Viewer']):
        from django.contrib import messages
        from django.shortcuts import redirect
        messages.error(request, "Permission Denied: Auditors cannot access live monitoring.")
        return redirect('home')
    cameras = Camera.objects.filter(is_enabled=True).order_by("camera_no")
    for cam in cameras:
        cam_key = f"camera_{cam.id}"
        cam_instance = active_cameras.get(cam_key)
        cam.is_online = camera_is_active(cam_instance, cam_key) if cam_instance else False
        
    groups = cameras.values_list('group', flat=True).distinct()
    
    # Calculate today's stats for Summary panel
    today = date.today()
    visitor_count = Visitorlogo.objects.filter(detection_time__date=today).count()
    ai_events_count = CameraLog.objects.filter(created_at__date=today).count()
    unknown_count = Visitorlogo.objects.filter(visitor_name="Unknown", detection_time__date=today).count()
    motion_events_count = Alert.objects.filter(created_at__date=today, alert_type__icontains="Motion").count()
    
    width, height, fps = 640, 480, 30
    context = {
        "cameras": cameras,
        "groups": groups,
        "visitor_count": visitor_count,
        "ai_events_count": ai_events_count,
        "unknown_count": unknown_count,
        "motion_events_count": motion_events_count,
        "width": width,
        "height": height,
        "fps": fps,
    }
    return render(request, "dashboard/live_monitoring.html", context)

@login_required
def cameras(request):
    cameras = Camera.objects.all().order_by("camera_no")
    
    online_count = 0
    recording_count = 0
    groups_set = set()
    
    # Types counts (mapping database camera types to dashboard types)
    type_counts = {
        "IP_Camera": 0,
        "PTZ_Camera": 0,
        "Dome_Camera": 0,
        "Bullet_Camera": 0,
    }
    
    # Locations counts
    location_counts = {}
    
    for cam in cameras:
        cam_key = f"camera_{cam.id}"
        cam_instance = active_cameras.get(cam_key)
        cam.is_online = camera_is_active(cam_instance, cam_key) if cam_instance else False
        
        if cam.is_online:
            online_count += 1
            
        if cam.recording_status == "Recording":
            recording_count += 1
            
        if cam.group:
            groups_set.add(cam.group)
        else:
            groups_set.add("Default")
            
        # Group by type (mapping choices RTSP, IP, USB, Webcam, and new detailed ones)
        ctype = cam.camera_type
        if ctype in ["IP", "RTSP", "IP Camera"]:
            type_counts["IP_Camera"] += 1
        elif ctype in ["USB", "PTZ Camera"]:
            type_counts["PTZ_Camera"] += 1
        elif ctype in ["Webcam", "Dome Camera"]:
            type_counts["Dome_Camera"] += 1
        elif ctype == "Bullet Camera":
            type_counts["Bullet_Camera"] += 1
        else:
            type_counts["Bullet_Camera"] += 1
            
        # Group by location
        loc = cam.location or "Default Location"
        location_counts[loc] = location_counts.get(loc, 0) + 1

    total_cameras = cameras.count()
    offline_count = max(0, total_cameras - online_count)
    total_groups = len(groups_set)
    
    # Normalize types counts to have realistic default ratios if empty
    if total_cameras == 0:
        type_counts = {
            "IP_Camera": 24,
            "PTZ_Camera": 4,
            "Dome_Camera": 2,
            "Bullet_Camera": 2,
        }
        total_cameras = 32
        online_count = 28
        offline_count = 4
        recording_count = 26
        total_groups = 6
        location_counts = {
            "Ground Floor": 14,
            "1st Floor": 9,
            "2nd Floor": 5,
            "Parking Area": 4
        }
    
    # Sort and slice top locations
    sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:4]
    
    # Storage details
    storage_total = 4.0
    storage_used = 2.48
    storage_free = 1.52
    storage_percent = 62
    
    context = {
        "cameras": cameras,
        "total_cameras": total_cameras,
        "online_count": online_count,
        "offline_count": offline_count,
        "recording_count": recording_count,
        "total_groups": total_groups,
        "type_counts": type_counts,
        "top_locations": sorted_locations,
        "storage_total": storage_total,
        "storage_used": storage_used,
        "storage_free": storage_free,
        "storage_percent": storage_percent,
    }
    return render(request, "dashboard/cameras.html", context)

@login_required
def delete_camera(request, camera_id):
    if not check_role(request.user, ['Super admin', 'Admin']):
        return JsonResponse({"status": "error", "message": "Permission Denied: Only Super admin and Admin can delete cameras."}, status=403)
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
    if not check_role(request.user, ['Super admin', 'Admin']):
        return JsonResponse({"status": "error", "message": "Permission Denied: Only Super admin and Admin can add cameras."}, status=403)
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
    
    from django.utils import timezone
    import datetime
    
    today = timezone.now().date()
    
    # Apply date filters
    date_range = request.GET.get('date_range')
    period = request.GET.get('period')
    if date_range:
        try:
            import datetime as dt_parser
            parsed = dt_parser.datetime.strptime(date_range.split('-')[0].strip(), "%d %b, %Y").date()
            today = parsed
        except:
            pass
    elif period == 'yesterday':
        today = today - datetime.timedelta(days=1)
        
    yesterday = today - datetime.timedelta(days=1)
    
    alert_qs = Alert.objects.all()
    recog_qs = RecognitionLog.objects.all()
    vis_qs = Visitorlogo.objects.all()
    
    # Apply camera filter
    camera_filter = request.GET.get('camera')
    if camera_filter:
        alert_qs = alert_qs.filter(message__icontains=camera_filter)
        recog_qs = recog_qs.filter(camera_name=camera_filter)
        vis_qs = vis_qs.filter(camera_name=camera_filter)
        
    # Apply detection type filter
    det_type = request.GET.get('detection_type')
    if det_type:
        if det_type == 'Known':
            alert_qs = alert_qs.none()
            vis_qs = vis_qs.exclude(visitor_name='Unknown')
            recog_qs = recog_qs.exclude(person_name__iexact='Unknown')
        elif det_type == 'Unknown':
            alert_qs = alert_qs.filter(alert_type__icontains='Unknown Face')
            vis_qs = vis_qs.filter(visitor_name='Unknown')
            recog_qs = recog_qs.filter(person_name__iexact='Unknown')
        elif det_type == 'Mask':
            alert_qs = alert_qs.filter(alert_type__icontains='Mask')
            vis_qs = vis_qs.none()
            recog_qs = recog_qs.none()
        elif det_type == 'Multiple':
            alert_qs = alert_qs.filter(alert_type__icontains='Multiple')
            vis_qs = vis_qs.none()
            recog_qs = recog_qs.none()
        elif det_type == 'Quality':
            alert_qs = alert_qs.filter(alert_type__icontains='Quality')
            vis_qs = vis_qs.none()
            recog_qs = recog_qs.none()
            
    alerts = alert_qs.order_by('-created_at')[:50]
    
    def get_change(today_count, yesterday_count):
        if yesterday_count == 0:
            return 100.0 if today_count > 0 else 0.0
        return round(((today_count - yesterday_count) / yesterday_count) * 100, 1)

    # Base querysets
    # Total Detections
    total_detections = alert_qs.count() + recog_qs.count()
    total_today = alert_qs.filter(created_at__date=today).count() + recog_qs.filter(detection_time__date=today).count()
    total_yesterday = alert_qs.filter(created_at__date=yesterday).count() + recog_qs.filter(detection_time__date=yesterday).count()
    total_change = get_change(total_today, total_yesterday)
    
    # Known Faces
    known_faces = recog_qs.exclude(person_name__iexact='Unknown').count()
    known_today = recog_qs.exclude(person_name__iexact='Unknown').filter(detection_time__date=today).count()
    known_yesterday = recog_qs.exclude(person_name__iexact='Unknown').filter(detection_time__date=yesterday).count()
    known_change = get_change(known_today, known_yesterday)
    
    # Unknown Faces
    unknown_faces = recog_qs.filter(person_name__iexact='Unknown').count() + alert_qs.filter(alert_type__icontains='Unknown Face').count()
    unknown_today = recog_qs.filter(person_name__iexact='Unknown', detection_time__date=today).count() + alert_qs.filter(alert_type__icontains='Unknown Face', created_at__date=today).count()
    unknown_yesterday = recog_qs.filter(person_name__iexact='Unknown', detection_time__date=yesterday).count() + alert_qs.filter(alert_type__icontains='Unknown Face', created_at__date=yesterday).count()
    unknown_change = get_change(unknown_today, unknown_yesterday)
    
    # Mask, Multiple Faces, Face Quality (from Alerts)
    mask_detected = alert_qs.filter(alert_type__icontains='Mask').count()
    mask_today = alert_qs.filter(alert_type__icontains='Mask', created_at__date=today).count()
    mask_yesterday = alert_qs.filter(alert_type__icontains='Mask', created_at__date=yesterday).count()
    mask_change = get_change(mask_today, mask_yesterday)
    
    multiple_faces = alert_qs.filter(alert_type__icontains='Multiple').count()
    multiple_today = alert_qs.filter(alert_type__icontains='Multiple', created_at__date=today).count()
    multiple_yesterday = alert_qs.filter(alert_type__icontains='Multiple', created_at__date=yesterday).count()
    multiple_change = get_change(multiple_today, multiple_yesterday)
    
    face_quality = alert_qs.filter(alert_type__icontains='Quality').count()
    quality_today = alert_qs.filter(alert_type__icontains='Quality', created_at__date=today).count()
    quality_yesterday = alert_qs.filter(alert_type__icontains='Quality', created_at__date=yesterday).count()
    quality_change = get_change(quality_today, quality_yesterday)

    total_pie = known_faces + unknown_faces + mask_detected + multiple_faces + face_quality
    
    def get_pct(val, total):
        return round((val / total) * 100, 1) if total > 0 else 0.0

    known_pct = get_pct(known_faces, total_pie)
    unknown_pct = get_pct(unknown_faces, total_pie)
    mask_pct = get_pct(mask_detected, total_pie)
    multiple_pct = get_pct(multiple_faces, total_pie)
    quality_pct = get_pct(face_quality, total_pie)

    trend_labels = []
    trend_known = []
    trend_unknown = []
    trend_mask = []
    trend_multiple = []
    trend_quality = []

    for i in range(6, -1, -1):
        day = today - datetime.timedelta(days=i)
        trend_labels.append(day.strftime('%d %b'))
        
        t_known = recog_qs.exclude(person_name__iexact='Unknown').filter(detection_time__date=day).count()
        trend_known.append(t_known)
        
        t_unk = recog_qs.filter(person_name__iexact='Unknown', detection_time__date=day).count() + alert_qs.filter(alert_type__icontains='Unknown Face', created_at__date=day).count()
        trend_unknown.append(t_unk)
        
        t_mask = alert_qs.filter(alert_type__icontains='Mask', created_at__date=day).count()
        trend_mask.append(t_mask)
        
        t_mult = alert_qs.filter(alert_type__icontains='Multiple', created_at__date=day).count()
        trend_multiple.append(t_mult)
        
        t_qual = alert_qs.filter(alert_type__icontains='Quality', created_at__date=day).count()
        trend_quality.append(t_qual)

    import json
    trend_labels_json = json.dumps(trend_labels)
    trend_known_json = json.dumps(trend_known)
    trend_unknown_json = json.dumps(trend_unknown)
    trend_mask_json = json.dumps(trend_mask)
    trend_multiple_json = json.dumps(trend_multiple)
    trend_quality_json = json.dumps(trend_quality)
    
    cameras = Camera.objects.filter(is_enabled=True).order_by("camera_no")

    context = {
        "cameras": cameras,
        "alerts": alerts, 
        "logs": logs,
        "total_detections": total_detections,
        "total_change": total_change,
        "known_faces": known_faces,
        "known_change": known_change,
        "unknown_faces": unknown_faces,
        "unknown_change": unknown_change,
        "mask_detected": mask_detected,
        "mask_change": mask_change,
        "multiple_faces": multiple_faces,
        "multiple_change": multiple_change,
        "face_quality": face_quality,
        "quality_change": quality_change,
        "total_pie": total_pie,
        "known_pct": known_pct,
        "unknown_pct": unknown_pct,
        "mask_pct": mask_pct,
        "multiple_pct": multiple_pct,
        "quality_pct": quality_pct,
        "trend_labels": trend_labels_json,
        "trend_known": trend_known_json,
        "trend_unknown": trend_unknown_json,
        "trend_mask": trend_mask_json,
        "trend_multiple": trend_multiple_json,
        "trend_quality": trend_quality_json,
    }
    
    # Calculate stats per camera for the table
    camera_stats = []
    for camera in cameras:
        cam_known = recog_qs.filter(camera_name=camera.name).exclude(person_name__iexact='Unknown').count()
        cam_unk_vis = vis_qs.filter(camera_name=camera.name, visitor_name='Unknown').count()
        cam_unk_alert = alert_qs.filter(alert_type__icontains='Unknown Face', message__icontains=camera.name).count()
        cam_unk = cam_unk_vis + cam_unk_alert
        cam_mask = alert_qs.filter(alert_type__icontains='Mask', message__icontains=camera.name).count()
        cam_multiple = alert_qs.filter(alert_type__icontains='Multiple', message__icontains=camera.name).count()
        cam_quality = alert_qs.filter(alert_type__icontains='Quality', message__icontains=camera.name).count()
        total = cam_known + cam_unk + cam_mask + cam_multiple + cam_quality
        
        camera_stats.append({
            'name': camera.name,
            'known': cam_known,
            'unknown': cam_unk,
            'mask': cam_mask,
            'multiple': cam_multiple,
            'quality': cam_quality,
            'total': total,
            'known_pct': get_pct(cam_known, total),
            'unknown_pct': get_pct(cam_unk, total),
            'mask_pct': get_pct(cam_mask, total),
            'multiple_pct': get_pct(cam_multiple, total),
            'quality_pct': get_pct(cam_quality, total),
        })
        
    context["total_snapshots"] = Snapshot.objects.count()
    context["total_alerts_db"] = Alert.objects.count()
    context["total_recognitions_db"] = RecognitionLog.objects.count()
    context["active_cameras_count"] = Camera.objects.filter(is_enabled=True).count()
    
    context["camera_stats"] = camera_stats
    return render(request, "dashboard/ai_events.html", context)

@login_required
@login_required
def attendance_view(request):
    # Clean up any 'Unknown' records that were added before the fix
    Attendance.objects.filter(employee_name='Unknown').delete()
    attendance = Attendance.objects.all()
    from .models import Employee
    
    # 1. Search filter
    search_query = request.GET.get('q', '')
    if search_query:
        attendance = attendance.filter(employee_name__icontains=search_query)

    # 2. Department filter
    department = request.GET.get('department', '')
    if department and department != 'All Departments':
        emp_names = Employee.objects.filter(role__iexact=department).values_list('name', flat=True)
        attendance = attendance.filter(employee_name__in=emp_names)
    
    # 3. Status filter
    status = request.GET.get('status', '')
    if status:
        attendance = attendance.filter(status__iexact=status)
        
    # 4. Date filter
    date_filter = request.GET.get('date', '')
    if date_filter:
        attendance = attendance.filter(date=date_filter)
        
    attendance = attendance.order_by('-date', '-entry_time')[:50]
    
    # Get all distinct roles for the department dropdown
    all_departments = Employee.objects.values_list('role', flat=True).distinct()
    
    # Attach role to each attendance record
    for att in attendance:
        emp = Employee.objects.filter(name=att.employee_name).first()
        att.role = emp.role if emp else "Employee"
        att.emp_id = f"EMP{emp.id:03d}" if emp else "EMP---"
        
        # Fake calculations for UI
        if att.entry_time:
            # 9:30 AM is late
            from datetime import time
            late_time = time(9, 30)
            if att.entry_time > late_time:
                att.is_late = True
                # mock late minutes
                att.late_duration = f"{(att.entry_time.hour - 9) * 60 + (att.entry_time.minute - 30)}m"
            else:
                att.is_late = False

        if att.exit_time:
            early_time = time(17, 30)
            if att.exit_time < early_time:
                att.is_early_exit = True
                att.early_duration = f"{(17 - att.exit_time.hour) * 60 + (30 - att.exit_time.minute)}m"
            else:
                att.is_early_exit = False
                
        # Mock break times and working hours
        if att.entry_time and att.exit_time:
            att.working_hours = f"{att.exit_time.hour - att.entry_time.hour}h {abs(att.exit_time.minute - att.entry_time.minute)}m"
        else:
            att.working_hours = "-"
            
    # Dashboard Data Calculations
    import datetime
    today = datetime.date.today()
    total_employees = Employee.objects.count()
    
    today_attendance = Attendance.objects.filter(date=today)
    
    present_today = today_attendance.filter(status__iexact="Present").count()
    absent_today = today_attendance.filter(status__iexact="Absent").count()
    on_leave = today_attendance.filter(status__iexact="Leave").count()
    
    # Let's count Late Entries (After 09:30 AM)
    from datetime import time
    late_time = time(9, 30)
    late_entries = today_attendance.filter(status__iexact="Present", entry_time__gt=late_time).count()
    
    # Let's count Early Exits (Before 17:30 PM)
    early_time = time(17, 30)
    early_exits = today_attendance.filter(status__iexact="Present", exit_time__lt=early_time, exit_time__isnull=False).count()
    
    def get_pct(val, total):
        return round((val / total) * 100, 2) if total > 0 else 0.0

    # For charts
    # Department Wise (Mocked using roles)
    roles = Employee.objects.values_list('role', flat=True).distinct()
    dept_labels = []
    dept_present = []
    dept_absent = []
    for r in roles:
        r_str = str(r)
        dept_labels.append(r_str)
        emp_names = Employee.objects.filter(role=r).values_list('name', flat=True)
        r_present = today_attendance.filter(employee_name__in=emp_names, status__iexact="Present").count()
        r_absent = today_attendance.filter(employee_name__in=emp_names, status__iexact="Absent").count()
        dept_present.append(r_present)
        dept_absent.append(r_absent)
        
    import json
    dept_labels_json = json.dumps(dept_labels)
    dept_present_json = json.dumps(dept_present)
    dept_absent_json = json.dumps(dept_absent)

    # Monthly Summary (Last 30 Days)
    month_labels = []
    month_present = []
    month_absent = []
    month_leave = []
    
    for i in range(29, -1, -1):
        day = today - datetime.timedelta(days=i)
        month_labels.append(day.strftime('%d %b'))
        day_att = Attendance.objects.filter(date=day)
        month_present.append(day_att.filter(status__iexact="Present").count())
        month_absent.append(day_att.filter(status__iexact="Absent").count())
        month_leave.append(day_att.filter(status__iexact="Leave").count())
        
    month_labels_json = json.dumps(month_labels)
    month_present_json = json.dumps(month_present)
    month_absent_json = json.dumps(month_absent)
    month_leave_json = json.dumps(month_leave)

    all_employees = Employee.objects.all()

    import calendar
    _, num_days = calendar.monthrange(today.year, today.month)
    current_month_name = today.strftime('%B %Y')
    
    monthly_attendance_data = []
    # Pre-fetch current month attendance for all
    current_month_att = Attendance.objects.filter(date__year=today.year, date__month=today.month)
    
    for emp in all_employees:
        emp_month_att = current_month_att.filter(employee_name=emp.name)
        att_dict = {att.date.day: att.status for att in emp_month_att}
        
        emp_data = {
            'emp': emp,
            'days': [],
            'total_present': 0,
            'total_absent': 0,
            'total_leave': 0
        }
        
        for day in range(1, num_days + 1):
            status_day = att_dict.get(day)
            if status_day:
                s_lower = status_day.lower()
                if s_lower == 'present':
                    emp_data['total_present'] += 1
                    emp_data['days'].append('P')
                elif s_lower == 'absent':
                    emp_data['total_absent'] += 1
                    emp_data['days'].append('A')
                elif s_lower == 'leave':
                    emp_data['total_leave'] += 1
                    emp_data['days'].append('L')
                else:
                    emp_data['days'].append(status_day[:1].upper())
            else:
                date_obj = datetime.date(today.year, today.month, day)
                if date_obj > today:
                    emp_data['days'].append('-')
                elif date_obj.weekday() >= 5: # Weekend
                    emp_data['days'].append('O')
                else:
                    emp_data['days'].append('-')
                    
        monthly_attendance_data.append(emp_data)

    context = {
        "attendance": attendance,
        "current_status": status,
        "current_date": date_filter,
        "total_employees": total_employees,
        "present_today": present_today,
        "present_pct": get_pct(present_today, total_employees),
        "absent_today": absent_today,
        "absent_pct": get_pct(absent_today, total_employees),
        "late_entries": late_entries,
        "late_pct": get_pct(late_entries, total_employees),
        "early_exits": early_exits,
        "early_pct": get_pct(early_exits, total_employees),
        "on_leave": on_leave,
        "leave_pct": get_pct(on_leave, total_employees),
        "dept_labels": dept_labels_json,
        "dept_present": dept_present_json,
        "dept_absent": dept_absent_json,
        "month_labels": month_labels_json,
        "month_present": month_present_json,
        "month_absent": month_absent_json,
        "month_leave": month_leave_json,
        "search_query": search_query,
        "current_department": department,
        "all_departments": all_departments,
        "all_employees": all_employees,
        "monthly_attendance_data": monthly_attendance_data,
        "num_days_range": range(1, num_days + 1),
        "current_month_name": current_month_name,
    }
    return render(request, "dashboard/attendance.html", context)

@login_required
def delete_attendance(request):
    if request.method == "POST":
        ids = request.POST.getlist('ids[]')
        if ids:
            Attendance.objects.filter(id__in=ids).delete()
            messages.success(request, f"Successfully deleted {len(ids)} attendance record(s).")
        else:
            single_id = request.POST.get('id')
            if single_id:
                Attendance.objects.filter(id=single_id).delete()
                messages.success(request, "Successfully deleted attendance record.")
    return redirect('attendance')


@login_required
def export_attendance_csv(request):
    import csv
    from django.http import HttpResponse

    # Clean up before exporting
    Attendance.objects.filter(employee_name='Unknown').delete()
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="attendance_records.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Employee Name', 'Date', 'Entry Time', 'Exit Time', 'Status'])
    
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
        
    context = {}
    if request.user.is_superuser or request.user.is_staff:
        from django.contrib.auth.models import User
        from django.contrib.sessions.models import Session
        from django.utils import timezone
        
        all_users = User.objects.all().order_by('-last_login')
        
        # Get active session user ids
        active_sessions = Session.objects.filter(expire_date__gte=timezone.now())
        active_user_ids = set()
        for session in active_sessions:
            data = session.get_decoded()
            user_id = data.get('_auth_user_id')
            if user_id:
                active_user_ids.add(int(user_id))
                
        context['all_users'] = all_users
        context['active_user_ids'] = active_user_ids
        context['active_users_count'] = len(active_user_ids)
        
    return render(request, "dashboard/profile.html", context)

@login_required
def users_view(request):
    is_allowed = request.user.is_superuser or (hasattr(request.user, 'profile') and request.user.profile.role in ['Super admin', 'Admin'])
    if not is_allowed:
        from django.contrib import messages
        messages.error(request, "Permission Denied: Only Super admins and Admins can access User Management.")
        return redirect('home')

    from django.contrib.auth.models import User
    from dashboard.models import UserProfile
    from django.contrib import messages

    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        first_name = request.POST.get("first_name", "")
        last_name = request.POST.get("last_name", "")
        role = request.POST.get("role", "Viewer")

        if not username or not password:
            messages.error(request, "Username and password are required.")
            return redirect("users")

        if User.objects.filter(username=username).exists():
            messages.error(request, f"User with username '{username}' already exists.")
            return redirect("users")

        # Admin cannot create a Super admin account
        is_current_super = request.user.is_superuser or (hasattr(request.user, 'profile') and request.user.profile.role == 'Super admin')
        if role == 'Super admin' and not is_current_super:
            messages.error(request, "Permission Denied: Only Super admins can create Super admin accounts.")
            return redirect("users")

        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )
            UserProfile.objects.create(user=user, role=role)
            messages.success(request, f"Successfully created user '{username}' with role '{role}'.")
        except Exception as e:
            messages.error(request, f"Error creating user: {e}")
        return redirect("users")

    users = User.objects.all().order_by('-date_joined')
    from django.contrib.sessions.models import Session
    from django.utils import timezone
    active_sessions = Session.objects.filter(expire_date__gte=timezone.now())
    active_user_ids = set()
    for session in active_sessions:
        data = session.get_decoded()
        user_id = data.get('_auth_user_id')
        if user_id:
            active_user_ids.add(int(user_id))

    is_current_super = request.user.is_superuser or (hasattr(request.user, 'profile') and request.user.profile.role == 'Super admin')
    if is_current_super:
        available_roles = [r[0] for r in UserProfile.ROLE_CHOICES]
    else:
        available_roles = [r[0] for r in UserProfile.ROLE_CHOICES if r[0] != 'Super admin']

    context = {
        "users": users,
        "active_user_ids": active_user_ids,
        "active_users_count": len(active_user_ids),
        "roles": available_roles,
        "is_super_admin": is_current_super
    }
    return render(request, "dashboard/users.html", context)

@login_required
@require_http_methods(["POST"])
def delete_user(request, pk):
    is_allowed = request.user.is_superuser or (hasattr(request.user, 'profile') and request.user.profile.role in ['Super admin', 'Admin'])
    if not is_allowed:
        from django.contrib import messages
        messages.error(request, "Permission Denied: Only Super admins and Admins can access User Management.")
        return redirect('home')

    if request.user.id == pk:
        from django.contrib import messages
        messages.error(request, "You cannot delete your own account.")
        return redirect("users")

    from django.contrib.auth.models import User
    from django.contrib import messages
    try:
        user = User.objects.get(id=pk)
        
        # Check target user's role
        is_target_super = user.is_superuser or (hasattr(user, 'profile') and user.profile.role == 'Super admin')
        is_current_super = request.user.is_superuser or (hasattr(request.user, 'profile') and request.user.profile.role == 'Super admin')
        
        if is_target_super and not is_current_super:
            messages.error(request, "Permission Denied: Admins cannot delete Super admin accounts.")
            return redirect("users")

        username = user.username
        user.delete()
        messages.success(request, f"Successfully deleted user '{username}'.")
    except User.DoesNotExist:
        messages.error(request, "User not found.")
    return redirect("users")

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
    if not check_role(request.user, ['Super admin', 'Admin']):
        messages.error(request, "Permission Denied: Only Super admin and Admin can access Face Registration.")
        return redirect('home')
    if request.method == "POST":
        name = request.POST.get("person_name")
        role = request.POST.get("person_role")
        gender = request.POST.get("person_gender")
        employee_id_pk = request.POST.get("employee_id_pk")
        if employee_id_pk:
            employee = Employee.objects.get(pk=employee_id_pk)
            employee.name = name
            employee.role = role
            employee.gender = gender
            employee.employee_id = request.POST.get("employee_id")
            employee.department = request.POST.get("department")
            employee.designation = request.POST.get("designation")
            employee.phone_number = request.POST.get("phone_number")
            employee.email = request.POST.get("email")
            employee.date_of_birth = request.POST.get("date_of_birth") or None
            employee.address = request.POST.get("address")
            employee.notes = request.POST.get("notes")
            msg = f"Successfully updated {name}!"
        else:
            employee = Employee(
                name=name, 
                role=role, 
                gender=gender,
                employee_id=request.POST.get("employee_id"),
                department=request.POST.get("department"),
                designation=request.POST.get("designation"),
                phone_number=request.POST.get("phone_number"),
                email=request.POST.get("email"),
                date_of_birth=request.POST.get("date_of_birth") or None,
                address=request.POST.get("address"),
                notes=request.POST.get("notes")
            )
            msg = f"Successfully registered {name}!"
        
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
        for i in range(4, 11):
            process_image(str(i))
        
        employee.save()
        messages.success(request, msg)
        return redirect("face_register")

    registered_faces = Employee.objects.all().order_by('-id')
    stats = {
        'total': registered_faces.count(),
        'employees': registered_faces.filter(role__iexact='Employee').count(),
        'students': registered_faces.filter(role__iexact='Student').count(),
        'visitors': registered_faces.filter(role__iexact='Visitor').count(),
        'blacklisted': registered_faces.filter(role__iexact='Blacklist').count(),
        'watchlist': registered_faces.filter(role__iexact='Watchlist').count(),
    }
    
    total = stats['total'] or 1
    stats['emp_pct'] = round((stats['employees'] / total) * 100, 2)
    stats['stu_pct'] = round((stats['students'] / total) * 100, 2)
    stats['vis_pct'] = round((stats['visitors'] / total) * 100, 2)
    stats['blk_pct'] = round((stats['blacklisted'] / total) * 100, 2)
    stats['wat_pct'] = round((stats['watchlist'] / total) * 100, 2)

    context = {
        "registered_faces": registered_faces,
        "stats": stats
    }
    return render(request, "dashboard/face_register.html", context)

@login_required
@require_http_methods(["POST"])
def delete_face(request, pk):
    if not check_role(request.user, ['Super admin', 'Admin']):
        messages.error(request, "Permission Denied: Only Super admin and Admin can delete faces.")
        return redirect('home')
    try:
        employee = Employee.objects.get(id=pk)
        name = employee.name
        
        # Delete image files from disk to prevent storage leak and cache collision
        image_fields = [employee.image_front, employee.image_right, employee.image_left, employee.image]
        image_fields.extend([getattr(employee, f'image_{i}') for i in range(4, 11)])
        for img_field in image_fields:
            if img_field and img_field.name:
                img_field.delete(save=False)
                
        employee.delete()
        messages.success(request, f"Successfully deleted {name}.")
    except Employee.DoesNotExist:
        messages.error(request, "Employee not found.")
    return redirect("face_register")

@login_required
def wanted_persons(request):
    if request.method == "POST":
        if not check_role(request.user, ['Super admin', 'Admin']):
            messages.error(request, "Permission Denied: Only Super admin and Admin can add persons to the watchlist.")
            return redirect("wanted_persons")
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
    # Check if current user is allowed to add/delete (for frontend conditional rendering)
    can_manage = check_role(request.user, ['Super admin', 'Admin'])
    return render(request, "dashboard/wanted_persons.html", {"wanted_persons": persons, "can_manage": can_manage})

@login_required
@require_http_methods(["POST"])
def delete_wanted(request, pk):
    if not check_role(request.user, ['Super admin', 'Admin']):
        messages.error(request, "Permission Denied: Only Super admin and Admin can remove persons from the watchlist.")
        return redirect("wanted_persons")
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

import threading
import queue
from django.db import close_old_connections

def _recognition_worker(task_queue, result_queue):
    """Permanent background thread: reads face paths from task_queue, writes results to result_queue."""
    close_old_connections()
    while True:
        try:
            face_path = task_queue.get(timeout=5)
            if face_path is None:  # poison pill to stop
                break
            try:
                close_old_connections()
                name = recognize_face(face_path)
                try:
                    em = detect_emotion(face_path)
                except Exception:
                    em = "Unknown"
                try:
                    bl = check_blacklist(face_path)
                except Exception:
                    bl = None
                result_queue.put((name, em, bl))
                print(f"[BG Worker] name={name} | emotion={em} | blacklist={bl}")
            except Exception as e:
                print(f"[BG Worker] Error: {e}")
                result_queue.put(("Unknown", "Unknown", None))
            finally:
                task_queue.task_done()
                close_old_connections()
        except queue.Empty:
            pass


def generate_frames(camera_id, camera_name, camera_key):

    global people_count
    global face_count
    global entry_count, exit_count, occupancy, total_events
    global last_unknown_time
    global last_recognized_time
    global last_recognized_name
    global last_occupancy_time
    global camera_last_seen
    global name_id_map, next_person_id, unknown_counter
    
    frame_count = 0
    yolo_results = []  # persists across frames when YOLO skipped

    recognized_name = "Unknown"
    emotion = "Unknown"
    current_person_id = None
    blacklisted_person = None
    last_face_seen_time = 0.0
    last_recognition_trigger = 0.0
    face_in_frame = False          # tracks whether face was present last frame
    unknown_streak = 0

    # Per-camera recognition worker thread
    task_queue = queue.Queue(maxsize=1)  # maxsize=1: drop old tasks if worker is busy
    result_queue = queue.Queue()
    worker_thread = threading.Thread(
        target=_recognition_worker,
        args=(task_queue, result_queue),
        daemon=True
    )
    worker_thread.start()
    
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
            # Flush stale frames from buffer — grab without decoding
            for _ in range(3):
                camera.grab()
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
        # YOLO OBJECT DETECTION (every 3rd frame to reduce CPU load)
        # ==========================
        if frame_count % 3 == 0:
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
        # ==========================
        # FACE RECOGNITION — Queue-based async (non-blocking)
        # ==========================
        # 1. Drain result_queue — keep only the LATEST result
        latest_result = None
        while True:
            try:
                latest_result = result_queue.get_nowait()
            except queue.Empty:
                break
        if latest_result is not None:
            new_name, new_emotion, new_blacklisted = latest_result
            if new_name != "Unknown":
                recognized_name = new_name
                unknown_streak = 0
            else:
                unknown_streak += 1
                if unknown_streak >= 3:
                    recognized_name = "Unknown"
                    
            emotion = new_emotion if new_emotion != "Unknown" else emotion
            blacklisted_person = new_blacklisted if new_blacklisted else blacklisted_person
            # Update person ID
            if recognized_name != "Unknown":
                if recognized_name not in name_id_map:
                    name_id_map[recognized_name] = next_person_id
                    next_person_id += 1
                current_person_id = name_id_map[recognized_name]
            else:
                unknown_counter += 1
                current_person_id = f"U{unknown_counter}"
            # DB work
            try:
                if recognized_name != "Unknown":
                    today = date.today()
                    att = Attendance.objects.filter(employee_name=recognized_name, date=today).first()
                    if not att:
                        Attendance.objects.create(employee_name=recognized_name, status="Present")
                    else:
                        # Break Logic: If seen again after 30+ minutes, mark as Break
                        if att.exit_time and not att.break_out:
                            now = datetime.now()
                            exit_dt = datetime.combine(today, att.exit_time)
                            if (now - exit_dt).total_seconds() > 30 * 60:
                                att.break_out = att.exit_time
                                att.break_in = now.time()
                                att.save()
            except Exception as att_err:
                print("Attendance error:", att_err)
            try:
                now_dt = datetime.now()
                should_save = False
                if recognized_name == "Unknown":
                    should_save = (last_unknown_time is None or now_dt - last_unknown_time > timedelta(seconds=30))
                else:
                    should_save = (last_recognized_name != recognized_name or
                                   last_recognized_time is None or
                                   now_dt - last_recognized_time > timedelta(seconds=30))
                if should_save:
                    if recognized_name == "Unknown":
                        Alert.objects.create(alert_type="Unknown Face",
                                             message=f"Unknown person detected in {camera_name}")
                    else:
                        Alert.objects.create(alert_type="Known Face",
                                             message=f"{recognized_name} detected in {camera_name}")
                    vis_file = now_dt.strftime("%Y%m%d_%H%M%S") + ".jpg"
                    vis_folder = os.path.join("media", "visitors")
                    os.makedirs(vis_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(vis_folder, vis_file), frame)
                    Visitorlogo.objects.create(visitor_name=recognized_name,
                                               Snapshot=f"visitors/{vis_file}",
                                               camera_name=camera_name)
                    if recognized_name == "Unknown":
                        last_unknown_time = now_dt
                    else:
                        last_recognized_time = now_dt
                        last_recognized_name = recognized_name
            except Exception as vis_err:
                print("Visitor log error:", vis_err)
            if blacklisted_person:
                is_blacklisted = True
                try:
                    Alert.objects.create(alert_type="Blacklisted Person",
                                         message=f"{blacklisted_person} detected in {camera_name}")
                    if can_send_email(f"blacklist_{blacklisted_person}"):
                        send_alert_email("🚨 BLACKLISTED PERSON DETECTED",
                                         f"{blacklisted_person} detected in {camera_name}")
                    bl_file = datetime.now().strftime("blacklist_%Y%m%d_%H%M%S.jpg")
                    bl_path = os.path.join("media", "blacklist_events", bl_file)
                    os.makedirs("media/blacklist_events", exist_ok=True)
                    cv2.imwrite(bl_path, frame)
                except Exception as bl_err:
                    print("Blacklist alert error:", bl_err)

        # 2. Trigger a new recognition if enough time has passed
        now_t = time.time()
        if len(faces) > 0 and task_queue.empty() and (now_t - last_recognition_trigger) > 2.0:
            last_recognition_trigger = now_t
            try:
                x, y, w, h = faces[0]
                fh, fw = frame.shape[:2]
                pad_w = int(w * 0.25)
                pad_h = int(h * 0.25)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(fw, x + w + pad_w)
                y2 = min(fh, y + h + pad_h)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    face_temp_path = f"media/current_face_{camera_id}_{id(task_queue)}.jpg"
                    cv2.imwrite(face_temp_path, face_img)
                    try:
                        task_queue.put_nowait(face_temp_path)  # non-blocking put
                    except queue.Full:
                        pass  # worker still busy, skip this frame
            except Exception as trigger_err:
                print("Trigger error:", trigger_err)

        # Keep is_blacklisted True if blacklisted_person is set
        if blacklisted_person:
            is_blacklisted = True

        # ==========================
        # ENTRY / EXIT TRACKING
        # ==========================
        face_now = len(faces) > 0
        if face_now and not face_in_frame:
            # Face just appeared — ENTRY
            entry_count += 1
            total_events += 1
            occupancy = max(0, entry_count - exit_count)
            face_in_frame = True
            print(f"[Entry] entry={entry_count} exit={exit_count} occupancy={occupancy}")
            
            try:
                analytics, _ = VisitorAnalytics.objects.get_or_create(date=date.today())
                analytics.entry_count = entry_count
                analytics.occupancy = occupancy
                analytics.save()
            except Exception as e:
                print("Analytics Entry error:", e)

        elif not face_now and face_in_frame:
            # Face disappeared — EXIT (only after 2s grace period)
            if time.time() - last_face_seen_time > 2.0:
                exit_count += 1
                total_events += 1
                occupancy = max(0, entry_count - exit_count)
                face_in_frame = False
                print(f"[Exit] entry={entry_count} exit={exit_count} occupancy={occupancy}")
                
                try:
                    today = date.today()
                    analytics, _ = VisitorAnalytics.objects.get_or_create(date=today)
                    analytics.exit_count = exit_count
                    analytics.occupancy = occupancy
                    analytics.save()
                    
                    # Update Attendance exit_time
                    if recognized_name != "Unknown":
                        att = Attendance.objects.filter(employee_name=recognized_name, date=today).first()
                        if att:
                            att.exit_time = datetime.now().time()
                            att.save()
                except Exception as e:
                    print("Analytics Exit error:", e)

        # Reset when face disappears (2-second grace period)
        if len(faces) > 0:
            last_face_seen_time = time.time()
        else:
            if time.time() - last_face_seen_time > 2.0:
                recognized_name = "Unknown"
                unknown_streak = 0
                emotion = "Unknown"
                current_person_id = None
                blacklisted_person = None

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

      



@login_required
def video_feed(request, camera_id):
    if not check_role(request.user, ['Super admin', 'Admin', 'Security Officer', 'Operator', 'Viewer']):
        return HttpResponse("Permission Denied: Auditors cannot access video feeds.", status=403)

    try:

        camera = Camera.objects.get(id=camera_id)

    except Camera.DoesNotExist:

        return HttpResponse("Camera not found", status=404)

        

    cam_key = f"camera_{camera.id}"

    

    return StreamingHttpResponse(

        generate_frames(camera.id, camera.name, cam_key),

        content_type='multipart/x-mixed-replace; boundary=frame'

    )

def occupancy_api(request):
    visitors_today = Visitorlogo.objects.filter(detection_time__date=date.today()).count()
    return JsonResponse({
        "people": people_count,
        "entry": entry_count,
        "exit": exit_count,
        "occupancy": occupancy,
        "visitors_today": visitors_today,
        "total_events": total_events
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
        return False
    except Exception:
        return False

def camera_status(request):
    global last_camera_status
    global last_camera_status_time
    
    cameras = Camera.objects.filter(is_enabled=True)
    camera_states = {}
    
    for cam in cameras:
        cam_key = f"camera_{cam.id}"
        cam_instance = active_cameras.get(cam_key)
        camera_states[cam_key] = {
            "is_open": camera_is_active(cam_instance, cam_key) if cam_instance else False,
            "name": cam.name
        }

    now = datetime.now().strftime("%I:%M %p")
    response_data = {}

    for camera_key, data in camera_states.items():
        is_open = data["is_open"]
        readable_name = data["name"]
        
        previous_status = last_camera_status.get(camera_key, False)
        print(f"{camera_key} | Previous={previous_status} | Current={is_open}")

        # OFFLINE ALERT
        if previous_status and not is_open:
            print(f"{readable_name} OFFLINE")
            Alert.objects.create(
                alert_type="Camera Offline",
                message=f"{readable_name} Offline"
            )
            if can_send_email(f"{camera_key}_offline"):
                send_alert_email("Camera Offline Alert", f"{readable_name} is Offline.")
            last_camera_status_time[camera_key] = now

        # ONLINE ALERT
        elif not previous_status and is_open:
            print(f"{readable_name} ONLINE")
            Alert.objects.create(
                alert_type="Camera Online",
                message=f"{readable_name} Online"
            )
            if can_send_email(f"{camera_key}_online"):
                send_alert_email("Camera Online Alert", f"{readable_name} is Online again.")
            last_camera_status_time[camera_key] = None

        last_camera_status[camera_key] = is_open
        response_data[camera_key] = "Online" if is_open else "Offline"
        response_data[f"{camera_key}_time"] = last_camera_status_time.get(camera_key)

    return JsonResponse(response_data)

@login_required
def control_camera(request):
    if not check_role(request.user, ['Super admin', 'Admin', 'Security Officer', 'Operator']):
        return JsonResponse({"status": "error", "message": "Permission Denied: You do not have permission to control cameras."}, status=403)
    
    camera_id = None
    action = ""
    status = ""

    if request.method == "POST":
        import json
        try:
            data = json.loads(request.body)
            camera_id = data.get("camera_id") or data.get("camera")
            action = data.get("action", "").lower()
            status = data.get("status", "")
        except Exception:
            return JsonResponse({"message": "Invalid JSON body", "status": "error"}, status=400)
    else:
        camera_id = request.GET.get("camera")
        action = request.GET.get("action", "").lower()

    if not camera_id:
        return JsonResponse({"message": "Camera ID required", "status": "error"}, status=400)
        
    try:
        cam_obj = Camera.objects.get(id=camera_id)
    except Camera.DoesNotExist:
        return JsonResponse({"message": "Camera not found", "status": "error"}, status=404)

    cam_key = f"camera_{camera_id}"
    
    if action == "toggle_recording":
        if not status:
            status = "Recording" if cam_obj.recording_status != "Recording" else "Not Recording"
        cam_obj.recording_status = status
        cam_obj.save()
        CameraLog.objects.create(event=f"Manual recording for {cam_obj.name} set to {status}")
        return JsonResponse({
            "status": "success", 
            "message": f"Recording status changed to {status}",
            "recording_status": status
        })

    if action == "start":
        camera_enabled[cam_key] = True
        CameraLog.objects.create(event=f"{cam_obj.name} Started")
        return JsonResponse({"message": f"{cam_obj.name} started", "status": "started"})

    if action == "stop":
        camera_enabled[cam_key] = False
        if cam_key in active_cameras and active_cameras[cam_key].isOpened():
            active_cameras[cam_key].release()
        CameraLog.objects.create(event=f"{cam_obj.name} Stopped")
        return JsonResponse({"message": f"{cam_obj.name} stopped", "status": "stopped"})

    return JsonResponse({"message": "Invalid camera action", "status": "error"}, status=400)

@login_required
def capture_snapshot(request):
    if not check_role(request.user, ['Super admin', 'Admin', 'Security Officer', 'Operator']):
        return JsonResponse({"status": "error", "message": "Permission Denied: You do not have permission to capture snapshots."}, status=403)
    camera_id = request.GET.get("camera")
    if not camera_id:
        return JsonResponse({"status": "error", "message": "Camera ID required"}, status=400)
        
    try:
        cam_obj = Camera.objects.get(id=camera_id)
    except Camera.DoesNotExist:
        return JsonResponse({"status": "error", "message": "Camera not found"}, status=404)
        
    cam_key = f"camera_{camera_id}"
    camera = active_cameras.get(cam_key)
    
    if camera and camera.isOpened():
        success, frame = camera.read()
        if success:
            filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
            path = os.path.join("media", "snapshots", filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, frame)
            
            Snapshot.objects.create(
                image=f"snapshots/{filename}",
                camera_name=cam_obj.name
            )
            CameraLog.objects.create(event=f"{cam_obj.name} Snapshot Captured")
            return JsonResponse({"status": "success", "message": f"{cam_obj.name} Snapshot Saved"})
            
    return JsonResponse({"status": "error", "message": "Camera not ready or read failed"}, status=500)

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
    return JsonResponse({"message": "saved"})

def face_count_api(request):
    return JsonResponse({"count": face_count})

def alerts_api(request):
    alerts = Alert.objects.order_by('-created_at')[:20]
    total_count = Alert.objects.count()
    last_seen = request.session.get('last_alert_count', 0)
    unread_count = max(0, total_count - last_seen)
    
    alert_list = [
        {
            "alert_type": alert.alert_type,
            "message": alert.message,
            "created_at": alert.created_at.strftime("%B %d, %Y, %I:%M %p")
        }
        for alert in alerts
    ]
    return JsonResponse({
        "alerts": alert_list,
        "total_count": total_count,
        "unread_count": unread_count
    })

def mark_alerts_read(request):
    if request.method == "POST":
        request.session['last_alert_count'] = Alert.objects.count()
        return JsonResponse({"status": "success"})
    return JsonResponse({"status": "error"}, status=400)

@login_required
@require_http_methods(["POST"])
@csrf_exempt
def delete_alerts(request):
    try:
        import json
        data = json.loads(request.body)
        alert_ids = [int(i) for i in data.get('alert_ids', []) if str(i).isdigit()]
        if not alert_ids:
            return JsonResponse({"status": "error", "message": "No alert IDs provided"}, status=400)
        deleted_count, _ = Alert.objects.filter(id__in=alert_ids).delete()
        return JsonResponse({"status": "success", "deleted_count": deleted_count})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@login_required
@require_http_methods(["POST"])
@csrf_exempt
def delete_logs(request):
    try:
        import json
        data = json.loads(request.body)
        log_ids = [int(i) for i in data.get('log_ids', []) if str(i).isdigit()]
        if not log_ids:
            return JsonResponse({"status": "error", "message": "No log IDs provided"}, status=400)
        deleted_count, _ = CameraLog.objects.filter(id__in=log_ids).delete()
        return JsonResponse({"status": "success", "deleted_count": deleted_count})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
def download_report(request):
    from datetime import datetime
    
    start_date_str = request.GET.get('start_date')
    end_date_str = request.GET.get('end_date')
    
    rec_filter, alert_filter, snap_filter, log_filter, att_filter = {}, {}, {}, {}, {}
    date_range_str = "All Time"
    
    if start_date_str and end_date_str:
        rec_filter['detection_time__range'] = [f"{start_date_str} 00:00:00", f"{end_date_str} 23:59:59"]
        alert_filter['created_at__range'] = [f"{start_date_str} 00:00:00", f"{end_date_str} 23:59:59"]
        snap_filter['created_at__range'] = [f"{start_date_str} 00:00:00", f"{end_date_str} 23:59:59"]
        log_filter['created_at__range'] = [f"{start_date_str} 00:00:00", f"{end_date_str} 23:59:59"]
        att_filter['date__range'] = [start_date_str, end_date_str]
        date_range_str = f"{start_date_str} to {end_date_str}"

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="Smart_CCTV_Report.pdf"'

    p = canvas.Canvas(response)

    # =========================
    # HEADER
    # =========================
    p.setFont("Helvetica-Bold", 20)
    p.drawString(130, 800, "SMART CCTV REPORT")

    report_id = datetime.now().strftime("CCTV-%Y%m%d-%H%M")
    p.setFont("Helvetica", 10)
    p.drawString(50, 770, f"Report ID : {report_id}")
    p.drawString(50, 750, f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p.drawString(50, 730, f"Time Range : {date_range_str}")

    # =========================
    # SUMMARY
    # =========================
    total_visitors = RecognitionLog.objects.filter(**rec_filter).count()
    total_alerts = Alert.objects.filter(**alert_filter).count()
    total_snapshots = Snapshot.objects.filter(**snap_filter).count()
    known_faces = RecognitionLog.objects.filter(**rec_filter).filter(status="Known").count()
    unknown_faces = RecognitionLog.objects.filter(**rec_filter).filter(status="Unknown").count()

    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, 700, "System Statistics")
    p.setFont("Helvetica", 10)
    p.drawString(50, 680, f"Total Visitors : {total_visitors}")
    p.drawString(50, 665, f"Known Faces : {known_faces}")
    p.drawString(50, 650, f"Unknown Faces : {unknown_faces}")
    p.drawString(50, 635, f"Total Alerts : {total_alerts}")
    p.drawString(50, 620, f"Total Snapshots : {total_snapshots}")

    # =========================
    # CAMERA STATUS
    # =========================
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, 590, "Camera Status")
    p.setFont("Helvetica", 10)
    p.drawString(50, 570, "Camera 1 : Online")
    p.drawString(50, 555, "Camera 2 : Offline")

    # =========================
    # CAMERA LOGS
    # =========================
    y = 520
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, "Camera Activity Logs")
    y -= 20
    logs = CameraLog.objects.filter(**log_filter).order_by('-created_at')[:5]
    p.setFont("Helvetica", 9)
    for log in logs:
        p.drawString(50, y, f"{log.event}")
        y -= 15

    # =========================
    # VISITOR HISTORY
    # =========================
    y -= 20
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, "Visitor History")
    y -= 20
    visitors = RecognitionLog.objects.filter(**rec_filter).order_by('-detection_time')[:5]
    p.setFont("Helvetica", 9)
    for visitor in visitors:
        p.drawString(50, y, f"{visitor.person_name} | {visitor.status}")
        y -= 15

    # =========================
    # ATTENDANCE
    # =========================
    y -= 20
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, "Attendance Report")
    y -= 20
    attendance = Attendance.objects.filter(**att_filter).order_by('-entry_time')[:5]
    p.setFont("Helvetica", 9)
    for row in attendance:
        p.drawString(50, y, f"{row.employee_name} | {row.status}")
        y -= 15

    # =========================
    # NEW PAGE
    # =========================
    p.showPage()
    p.setFont("Helvetica-Bold", 18)
    p.drawString(180, 800, "Alert History")
    y = 760
    alerts = Alert.objects.filter(**alert_filter).order_by('-created_at')[:15]
    p.setFont("Helvetica", 10)
    for alert in alerts:
        p.drawString(50, y, f"{alert.alert_type} | {alert.message}")
        y -= 20

    # =========================
    # SNAPSHOTS PAGE
    # =========================
    p.showPage()
    p.setFont("Helvetica-Bold", 18)
    p.drawString(180, 800, "Recent Snapshots")
    snapshots = Snapshot.objects.filter(**snap_filter).order_by('-created_at')[:4]
    x = 50
    y = 600
    for snap in snapshots:
        try:
            p.drawImage(ImageReader(snap.image.path), x, y, width=200, height=150)
            x += 250
            if x > 300:
                x = 50
                y -= 200
        except:
            pass

    # =========================
    # FOOTER
    # =========================
    p.setFont("Helvetica", 8)
    p.drawString(180, 20, "Generated By Smart CCTV Monitoring System")
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
    send_alert_email("Smart CCTV Test", "Email system working successfully.")
    return JsonResponse({"message": "Email Sent"})

@csrf_exempt
def camera_detail_api(request, camera_id):
    import json
    try:
        camera = Camera.objects.get(id=camera_id)
    except Camera.DoesNotExist:
        return JsonResponse({"error": "Camera not found"}, status=404)

    if request.method == "GET":
        return JsonResponse({
            "id": camera.id,
            "name": camera.name,
            "source_url": camera.source_url,
            "camera_type": camera.camera_type,
            "is_enabled": camera.is_enabled,
            "location": camera.location,
            "stream_quality": camera.stream_quality
        })
    elif request.method == "POST":
        try:
            data = json.loads(request.body)
            camera.name = data.get("name", camera.name)
            camera.source_url = data.get("source_url", camera.source_url)
            camera.camera_type = data.get("camera_type", camera.camera_type)
            camera.is_enabled = data.get("is_enabled", camera.is_enabled)
            camera.location = data.get("location", camera.location)
            camera.stream_quality = data.get("stream_quality", camera.stream_quality)
            camera.save()
            return JsonResponse({"status": "success", "message": "Camera updated successfully."})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Method not allowed"}, status=405)

def wanted_persons(request):
    if request.method == "POST":
        name = request.POST.get("wanted_name")
        image = request.FILES.get("wanted_image")
        if name and image:
            BlacklistPerson.objects.create(name=name, image=image)
            from django.contrib import messages
            messages.success(request, f"{name} has been added to the Wanted List.")
            from django.shortcuts import redirect
            return redirect("wanted_persons")
        else:
            from django.contrib import messages
            messages.error(request, "Name and image are required.")
    persons = BlacklistPerson.objects.all().order_by("-added_at")
    return render(request, "dashboard/wanted_persons.html", {"wanted_persons": persons})

@require_http_methods(["POST"])
@csrf_exempt
def delete_wanted(request, pk):
    try:
        person = BlacklistPerson.objects.get(id=pk)
        person.delete()
        return JsonResponse({"status": "success"})
    except BlacklistPerson.DoesNotExist:
        return JsonResponse({"status": "error", "message": "Person not found"}, status=404)

def export_ai_events_csv(request):
    import csv
    from django.http import HttpResponse
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="ai_events_report.csv"'
    writer = csv.writer(response)
    writer.writerow(['Date/Time', 'Alert Type', 'Message'])

    alert_qs = Alert.objects.all()
    
    camera_filter = request.GET.get('camera')
    if camera_filter:
        alert_qs = alert_qs.filter(message__icontains=camera_filter)
        
    det_type = request.GET.get('detection_type')
    if det_type:
        if det_type == 'Known':
            alert_qs = alert_qs.none()
        elif det_type == 'Unknown':
            alert_qs = alert_qs.filter(alert_type__icontains='Unknown Face')
        elif det_type == 'Mask':
            alert_qs = alert_qs.filter(alert_type__icontains='Mask')
        elif det_type == 'Multiple':
            alert_qs = alert_qs.filter(alert_type__icontains='Multiple')
        elif det_type == 'Quality':
            alert_qs = alert_qs.filter(alert_type__icontains='Quality')

    alerts = alert_qs.order_by('-created_at')[:500]
    for alert in alerts:
        writer.writerow([
            alert.created_at.strftime('%d %b %Y %I:%M %p'),
            alert.alert_type,
            alert.message
        ])
    return response
