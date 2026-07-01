import os
from django.db import models  # type: ignore

class CameraLog(models.Model):

    event = models.CharField(
        max_length=100
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return self.event


class Camera(models.Model):

   

    # Choices expanded to match modern UI design
    CAMERA_TYPE_CHOICES = [
        ("RTSP", "RTSP"),
        ("IP", "IP"),
        ("USB", "USB"),
        ("Webcam", "Webcam"),
        ("IP Camera", "IP Camera"),
        ("PTZ Camera", "PTZ Camera"),
        ("Dome Camera", "Dome Camera"),
        ("Bullet Camera", "Bullet Camera"),
    ]

    HEALTH_STATUS_CHOICES = [
        ("Healthy", "Healthy"),
        ("Warning", "Warning"),
        ("Offline", "Offline"),
        ("Maintenance", "Maintenance"),
    ]

    RECORDING_STATUS_CHOICES = [
        ("Recording", "Recording"),
        ("Not Recording", "Not Recording"),
        ("Paused", "Paused"),
    ]

    STREAM_QUALITY_CHOICES = [
        ("Auto", "Auto"),
        ("Low", "Low"),
        ("Medium", "Medium"),
        ("High", "High"),
        ("HD", "HD"),
    ]

    name = models.CharField(
        max_length=100,
        unique=True
    )

    source_url = models.CharField(
        max_length=255,
        blank=True
    )

    camera_no = models.PositiveIntegerField(
    unique=True
)

    camera_type = models.CharField(
        max_length=20,
        choices=CAMERA_TYPE_CHOICES,
        default="Webcam"
    )

    group = models.CharField(
        max_length=100,
        default="Default"
    )

    health_status = models.CharField(
        max_length=20,
        choices=HEALTH_STATUS_CHOICES,
        default="Healthy"
    )

    is_online = models.BooleanField(
        default=False
    )

    is_enabled = models.BooleanField(
        default=True
    )

    recording_status = models.CharField(
        max_length=30,
        choices=RECORDING_STATUS_CHOICES,
        default="Not Recording"
    )

    location = models.CharField(
        max_length=150,
        blank=True
    )

    floor_mapping = models.CharField(
        max_length=150,
        blank=True
    )

    live_fps = models.FloatField(
        default=0
    )

    bitrate_kbps = models.PositiveIntegerField(
        default=0
    )

    stream_quality = models.CharField(
        max_length=20,
        choices=STREAM_QUALITY_CHOICES,
        default="Auto"
    )

    audio_enabled = models.BooleanField(
        default=False
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    def __str__(self):
        return self.name
    
class Snapshot(models.Model):

    image = models.ImageField(
        upload_to='snapshots/'
    )

    camera_name = models.CharField(
        max_length=100,
        default="Unknown Camera"
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

class Employee(models.Model):

    name = models.CharField(max_length=100)
    employee_id = models.CharField(max_length=50, null=True, blank=True)
    department = models.CharField(max_length=100, null=True, blank=True)
    designation = models.CharField(max_length=100, null=True, blank=True)
    phone_number = models.CharField(max_length=20, null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    
    role = models.CharField(max_length=50, default="Employee")
    gender = models.CharField(max_length=20, default="Unknown")
    
    registered_on = models.DateTimeField(auto_now_add=True, null=True)

    image = models.ImageField(
        upload_to='employees/',
        null=True, blank=True
    )

    image_front = models.ImageField(upload_to='employees/front/', null=True, blank=True)
    image_right = models.ImageField(upload_to='employees/right/', null=True, blank=True)
    image_left = models.ImageField(upload_to='employees/left/', null=True, blank=True)
    
    image_4 = models.ImageField(upload_to='employees/extra/', null=True, blank=True)
    image_5 = models.ImageField(upload_to='employees/extra/', null=True, blank=True)
    image_6 = models.ImageField(upload_to='employees/extra/', null=True, blank=True)
    image_7 = models.ImageField(upload_to='employees/extra/', null=True, blank=True)
    image_8 = models.ImageField(upload_to='employees/extra/', null=True, blank=True)
    image_9 = models.ImageField(upload_to='employees/extra/', null=True, blank=True)
    image_10 = models.ImageField(upload_to='employees/extra/', null=True, blank=True)
    
class RecognitionLog(models.Model):

    person_name = models.CharField(
        max_length=100
    )

    status = models.CharField(
        max_length=20
    )

    camera_name = models.CharField(
        max_length=100,
        default="Unknown Camera"
    )

    detection_time = models.DateTimeField(
        auto_now_add=True
    )

class  Visitorlogo(models.Model):

    visitor_name = models.CharField(
        max_length=100,
        default="Unknown Visitor"
    )

    camera_name = models.CharField(
        max_length=50,
        default="Unknown"
    )

    Snapshot = models.ImageField(
        upload_to='visitors/'
    )

    detection_time = models.DateTimeField(
        auto_now_add=True
    )


    def __str__(self):
        return self.visitor_name
    
class Alert(models.Model):

    alert_type = models.CharField(
        max_length=100,
    )

    message = models.TextField()

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    def __str__(self):  
        return  self.alert_type
    
class Attendance(models.Model):

    employee_name = models.CharField(
        max_length=100
    )

    date = models.DateField(
        auto_now_add=True
    )

    entry_time = models.TimeField(
        auto_now_add=True
    )

    exit_time = models.TimeField(null=True, blank=True)
    
    break_out = models.TimeField(null=True, blank=True)
    break_in = models.TimeField(null=True, blank=True)

    status = models.CharField(
        max_length=20,
        default="Present"
    )

    def __str__(self):
        return f"{self.employee_name} - {self.date}"

class Report(models.Model):

    generated_at = models.DateTimeField(
        auto_now_add=True
    )

    pdf_file = models.FileField(
        upload_to="reports/"
    )

    def __str__(self):
        return f"Report {self.id}"
    
class VisitorAnalytics(models.Model):

    date = models.DateField(auto_now_add=True)

    entry_count = models.IntegerField(default=0)

    exit_count = models.IntegerField(default=0)

    occupancy = models.IntegerField(default=0)

    def save(self, *args, **kwargs):
        self.occupancy = self.entry_count - self.exit_count
        super().save(*args, **kwargs)


class BlacklistPerson(models.Model):

    name = models.CharField(
        max_length=100
    )

    image = models.ImageField(
        upload_to="blacklist/"
    )

    added_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return self.name        

from django.contrib.auth.models import User

class UserProfile(models.Model):
    ROLE_CHOICES = [
        ("Super admin", "Super admin"),
        ("Admin", "Admin"),
        ("Security Officer", "Security Officer"),
        ("Operator", "Operator"),
        ("Viewer", "Viewer"),
        ("Auditor", "Auditor"),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(max_length=30, choices=ROLE_CHOICES, default="Viewer")

    def __str__(self):
        return f"{self.user.username} - {self.role}"
