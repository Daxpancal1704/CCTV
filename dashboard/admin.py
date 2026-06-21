from django.contrib import admin
from .models import CameraLog
from .models import Snapshot
from .models import Employee
from .models import RecognitionLog
from .models import Visitorlogo
from .models import Alert
from .models import Attendance
from .models import Report

admin.site.register(CameraLog)
admin.site.register(Snapshot)
admin.site.register(Employee)
admin.site.register(RecognitionLog)
admin.site.register(Visitorlogo)
admin.site.register(Alert)
admin.site.register(Attendance)
admin.site.register(Report)