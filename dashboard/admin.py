from django.contrib import admin
from .models import CameraLog
from .models import Camera
from .models import Snapshot
from .models import Employee
from .models import RecognitionLog
from .models import Visitorlogo
from .models import Alert
from .models import Attendance
from .models import Report
from.models import VisitorAnalytics
from.models import BlacklistPerson
from .models import UserProfile
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

admin.site.register(CameraLog)
admin.site.register(Camera)
admin.site.register(Snapshot)
admin.site.register(Employee)
admin.site.register(RecognitionLog)
admin.site.register(Visitorlogo)
admin.site.register(Alert)
admin.site.register(Attendance)
admin.site.register(Report)
admin.site.register(VisitorAnalytics)
admin.site.register(BlacklistPerson)
admin.site.register(UserProfile)

# Inline UserProfile registration under default UserAdmin
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'UserProfile'

class UserAdmin(BaseUserAdmin):
    inlines = [UserProfileInline]

admin.site.unregister(User)
admin.site.register(User, UserAdmin)
