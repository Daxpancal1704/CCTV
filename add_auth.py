import re
import os

views_path = r'c:\private\Project\SmartCCTV\dashboard\views.py'

with open(views_path, 'r') as f:
    content = f.read()

if 'from django.contrib.auth.decorators import login_required' not in content:
    content = content.replace('from django.shortcuts import render', 'from django.shortcuts import render\nfrom django.contrib.auth.decorators import login_required')

views_to_decorate = [
    'home', 'live_monitoring', 'cameras', 'ai_events', 'attendance_view',
    'visitors_view', 'snapshots_view', 'logs_view', 'settings_view',
    'alerts_page', 'reports_view', 'face_register', 'capture_snapshot',
    'download_snapshot', 'delete_snapshot'
]

for view in views_to_decorate:
    # Only add if not already decorated
    pattern = re.compile(r'^(?!@login_required\n)(def ' + view + r'\(request.*?:)$', re.MULTILINE)
    content = pattern.sub(r'@login_required\n\1', content)

with open(views_path, 'w') as f:
    f.write(content)

print("Added decorators")
