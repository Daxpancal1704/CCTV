import os
import django
import sys
sys.path.append('c:\\private\\Project\\SmartCCTV')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()
from dashboard.models import Camera
for c in Camera.objects.all():
    print(c.id, c.name, c.camera_no)
