"""
Based on example face detect from http://dlib.net/face_detector.py.html
"""

import cv2
import dlib

detector = dlib.get_frontal_face_detector()
# win = dlib.image_window()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print('Can`t capture frame')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    # dets, scores, idx = detector.run(gray, 1, -1)
    # for i, d in enumerate(dets):
    #     print("Detection {}, score: {}, face_type:{}".format(d, scores[i],
    #                                                          idx[i]))
    for d in dets:
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()),
                      (0, 0, 255), 2)
    cv2.imshow('Faces detect', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# for f in sys.argv[1:]:
#     print("Processing file: {}".format(f))
#     img = dlib.load_rgb_image(f)
#     # The 1 in the second argument indicates that we should upsample the image
#     # 1 time.  This will make everything bigger and allow us to detect more
#     # faces.
#     dets = detector(img, 1)
#     print("Number of faces detected: {}".format(len(dets)))
#     for i, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#             i, d.left(), d.top(), d.right(), d.bottom()))
#
#     win.clear_overlay()
#     win.set_image(img)
#     win.add_overlay(dets)
#     dlib.hit_enter_to_continue()

# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.
# if (len(sys.argv[1:]) > 0):
#     img = dlib.load_rgb_image(sys.argv[1])
#     dets, scores, idx = detector.run(img, 1, -1)
#     for i, d in enumerate(dets):
#         print("Detection {}, score: {}, face_type:{}".format(
#             d, scores[i], idx[i]))
