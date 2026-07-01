import os
import sys
import django

sys.path.append(r'c:\private\Project\SmartCCTV')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from dashboard.models import Employee

print("Employees Count:", Employee.objects.count())
for emp in Employee.objects.order_by('-id')[:5]:
    print(f"ID: {emp.id}, Name: {emp.name}")
    fields = ['image_front', 'image_right', 'image_left'] + [f'image_{i}' for i in range(4, 11)]
    for f in fields:
        val = getattr(emp, f)
        if val and val.name:
            print(f"  {f}: {val.name} (exists: {os.path.exists(val.path)})")
