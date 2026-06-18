from django.db import models

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

    created_at = models.DateTimeField(
        auto_now_add=True
    )


# class Employee(models.Model):

#     name=models.CharField(
#         max_length=100
#     )

#     image=models.ImageField(
#         upload_to='employees/'
#     )
#     def __str__(self):
#         return self.name



from django.db import models
import os

class Employee(models.Model):

    name = models.CharField(
        max_length=100,
        blank=True
    )

    image = models.ImageField(
        upload_to='employees/'
    )

    def save(self,*args,**kwargs):

        if not self.name:

            filename = os.path.basename(
                self.image.name
            )

            self.name = os.path.splitext(
                filename
            )[0]

        super().save(*args,**kwargs)

    def __str__(self):
        return self.name
    

class RecognitionLog(models.Model):

    person_name = models.CharField(
        max_length=100
    )

    status = models.CharField(
        max_length=20
    )

    detection_time = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return self.person_name