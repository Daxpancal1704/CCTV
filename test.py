import cv2

cam1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    cv2.imshow("Cam1", frame1)
    cv2.imshow("Cam2", frame2)

    if cv2.waitKey(1) == 27:
        break

