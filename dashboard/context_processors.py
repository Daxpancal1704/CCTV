from .models import Alert

def alert_context(request):
    try:
        total_alerts = Alert.objects.count()
        last_seen = request.session.get('last_alert_count', 0)
        unread_count = max(0, total_alerts - last_seen)
        recent = Alert.objects.order_by('-created_at')[:3]
    except Exception:
        unread_count = 0
        recent = []
    
    return {
        'alert_count': unread_count,
        'recent_alerts': recent
    }
