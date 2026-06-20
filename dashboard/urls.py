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
    # path('people_count/', views.people_count, name='people_count'),
    path(
    'people_count/',
    views.people_count_api,
    name='people_count'
),
path(
    'video_feed_cam1/',
    views.video_feed_cam1,
    name='video_feed_cam1'
),

path(
    'video_feed_cam2/',
    views.video_feed_cam2,
    name='video_feed_cam2'
),
path(
    'occupancy/',
    views.occupancy_api,
    name='occupancy'
),
]