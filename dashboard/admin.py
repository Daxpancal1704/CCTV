from django.contrib import admin
from .models import CameraLog
from .models import Snapshot
from .models import Employee
from .models import RecognitionLog


admin.site.register(CameraLog)
admin.site.register(Snapshot)
admin.site.register(Employee)
admin.site.register(RecognitionLog)