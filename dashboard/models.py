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

    image = models.ImageField(
        upload_to='employees/'
    )
    
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