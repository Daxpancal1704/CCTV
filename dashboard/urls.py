from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    # path('video_feed/', views.video_feed, name='video_feed'),
    path('camera_status/', views.camera_status, name='camera_status'),
    path('snapshot/', views.capture_snapshot, name='snapshot'),
    path('face_count/', views.face_count_api, name='face_count'),
    path('alerts/', views.alerts_api, name='alerts_api'),
    path('test_log/', views.test_log, name='test_log'),
    path('people_count/',views.people_count_api,name='people_count'),
    path('video_feed_cam1/',views.video_feed_cam1,name='video_feed_cam1'),
    path('video_feed_cam2/',views.video_feed_cam2,name='video_feed_cam2'),
    path('camera_control/', views.control_camera, name='camera_control'),
    path('occupancy/',views.occupancy_api,name='occupancy'),
    path('download_report/',views.download_report,name='download_report'),
    path('test_email/',views.test_email,name='test_email'),
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
