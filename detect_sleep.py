import cv2
import time
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import os
from playsound import playsound  # Import playsound library

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def DlibVideoHandler(dlibFacePredictor):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlibFacePredictor)

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if ret:
            frame = dlibVideo(frame, detector, predictor, lStart, lEnd, rStart, rEnd)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            continue

    video_capture.release()
    cv2.destroyAllWindows()

def dlibVideo(frame, detector, predictor, lStart, lEnd, rStart, rEnd):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    global lock, start, threshold

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        eyes = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if eyes < threshold and lock == False:
            lock = True
            start = time.time()

        if eyes > threshold and lock == True:
            lock = False
            start = 0.0

        elif eyes < threshold and lock == True:
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
            if round(time.time() - start) > 5:
                dlibVideoAlert()

    return frame

def dlibVideoAlert():
    print('Alert: Eyes closed for more than 1 second!')
    # Play a sound alert
    playsound(r'C:\Users\WECC\Downloads\nokia-iphone-a28e6ab4-dccd-3836-962c-52d5d272ead7-42372.mp3')  # Replace 'alert.wav' with the path to your sound file

if __name__ == '__main__':
    global lock, start, threshold
    home = os.getcwd()
    threshold = 0.2  # Threshold for detecting closed eyes
    lock = False
    start = 0.0

    dlibFacePredictor = os.path.join(home, 'shape_predictor_68_face_landmarks.dat')
    DlibVideoHandler(dlibFacePredictor)
