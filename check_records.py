import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SmartCCTV.settings')
django.setup()

from dashboard.models import Alert, RecognitionLog

print("Alert Types:", set(Alert.objects.values_list('alert_type', flat=True)))
print("Recognition Statuses:", set(RecognitionLog.objects.values_list('status', flat=True)))
