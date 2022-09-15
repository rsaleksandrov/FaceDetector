"""
Based on example face detect from http://dlib.net/face_detector.py.html
"""

import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor5 = dlib.shape_predictor('data/shape_predictor_5_face_landmarks.dat')
predictor68 = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('data/testvideo.mp4')
# cap = cv2.VideoCapture('data/testfoto.jpg')

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 0)
        for d in dets:
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()),
                          (0, 0, 255), 2)
            shape = predictor5(gray, d)
            for i in range(0, 5):
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1,
                           (255, 0, 0), -1)
            shape = predictor68(gray, d)
            for i in range(0, 68):
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2,
                           (0, 255, 0), -1)

        cv2.imshow('Faces detect', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
