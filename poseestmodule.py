import cv2
import mediapipe as mp
import time

vid = cv2.VideoCapture(r'C:\Users\ELCOT\Downloads\1.mp4')


class poseDetector():
    def __init__(self, mode=False, modcom=1, smooth=True, enseg=True, smoothseg=True, detcon=0.5, trackcn=0.5):
        self.mode = mode
        self.modcom = modcom
        self.smooth = smooth
        self.enseg = enseg
        self.smoothseg = smoothseg
        self.detcon = detcon
        self.trackcn = trackcn

        self.mppose = mp.solutions.pose
        self.pose = self.mppose.Pose(self.mode, self.modcom, self.smooth, self.enseg,
                                     self.smoothseg, self.detcon, self.trackcn)
        self.mpDraw = mp.solutions.drawing_utils

    def findpose(self, img, Draw=True):
        imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRBG)
        if Draw:
            if results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mppose.POSE_CONNECTIONS)

        return img

        # for id,lm in enumerate(results.pose_landmarks.landmark):
        # h,w,c=img.shape
        # cx,cy=int(lm.x*w),int(lm.y*h)
        # #cv2.circle(img,(cx,cy),13,(255,255,0),cv2.FILLED)


def main():
    ctime = 0
    ptime = 0
    detector = poseDetector()

    while True:
        suc, img = vid.read()
        img = detector.findpose(img)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 56, 56), 3)
        cv2.imshow("IMAGE", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
