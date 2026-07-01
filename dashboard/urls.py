from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('live_monitoring/', views.live_monitoring, name='live_monitoring'),
    path('cameras/', views.cameras, name='cameras'),
    path('add_camera/', views.add_camera, name='add_camera'),
    path('api/camera/<int:camera_id>/', views.camera_detail_api, name='camera_detail_api'),
    path('delete_camera/<int:camera_id>/', views.delete_camera, name='delete_camera'),
    path('ai_events/', views.ai_events, name='ai_events'),
    path('attendance/', views.attendance_view, name='attendance'),
    path('delete_attendance/', views.delete_attendance, name='delete_attendance'),
    path('export_attendance/', views.export_attendance_csv, name='export_attendance_csv'), # Added export
    path('visitors/', views.visitors_view, name='visitors'),
    path('snapshots/', views.snapshots_view, name='snapshots'),
    path('alerts_page/', views.alerts_page, name='alerts_page'),
    path('export_alerts/', views.export_alerts_csv, name='export_alerts_csv'),
    path('export_ai_events/', views.export_ai_events_csv, name='export_ai_events_csv'),
    path('export_visitors/', views.export_visitors_csv, name='export_visitors_csv'),
    path('reports/', views.reports_view, name='reports'),
    path('face_register/', views.face_register, name='face_register'),
    path('delete_face/<int:pk>/', views.delete_face, name='delete_face'),
    path('wanted_persons/', views.wanted_persons, name='wanted_persons'),
    path('delete_wanted/<int:pk>/', views.delete_wanted, name='delete_wanted'),
    path('logs/', views.logs_view, name='logs'),
    path('users/', views.users_view, name='users'),
    path('delete_user/<int:pk>/', views.delete_user, name='delete_user'),
    path('profile/', views.profile_view, name='profile'),
    path('settings/', views.settings_view, name='settings'),
    # path('video_feed/', views.video_feed, name='video_feed'),
    path('camera_status/', views.camera_status, name='camera_status'),
    path('snapshot/', views.capture_snapshot, name='snapshot'),
    path('face_count/', views.face_count_api, name='face_count'),
    path('alerts/', views.alerts_api, name='alerts_api'),
    path('delete_alerts/', views.delete_alerts, name='delete_alerts'),
    path('mark_alerts_read/', views.mark_alerts_read, name='mark_alerts_read'),
    path('test_log/', views.test_log, name='test_log'),
    path('delete_logs/', views.delete_logs, name='delete_logs'),
    path('people_count/',views.people_count_api,name='people_count'),
    path('video_feed/<int:camera_id>/', views.video_feed, name='video_feed'),
    path('camera_control/', views.control_camera, name='camera_control'),
    path('occupancy/',views.occupancy_api,name='occupancy'),
    path('download_report/',views.download_report,name='download_report'),
    path('test_email/',views.test_email,name='test_email'),
    path('download_snapshot/<int:pk>/', views.download_snapshot, name='download_snapshot'),
    path('delete_snapshot/<int:pk>/', views.delete_snapshot, name='delete_snapshot'),
# path(
#     'entry_count/',
#     views.entry_count_api,
#     name='entry_count'
# ),

# path(
#     'exit_count/',
#     views.exit_count_api,
#     name='exit_count'
# ),

# path(
#     'occupancy/',
#     views.occupancy_api,
#     name='occupancy'
# ),

]
