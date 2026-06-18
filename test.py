import cv2

cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

print("Cam0:", cam0.isOpened())
print("Cam1:", cam1.isOpened())