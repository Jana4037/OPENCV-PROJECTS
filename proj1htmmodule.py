import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, max_hands=2,modelcomplex=1, detectcon=0.5, trakcon=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detectcon = detectcon
        self.trakcon = trakcon
        self.model_complexity = modelcomplex

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands,self.model_complexity, self.detectcon, self.trakcon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img


    def findpositions(self,img,hand=0,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[hand]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y *h)
                lmlist.append([id,cx,cy])
                if draw and id==hand:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmlist

def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    detector = handDetector()
    while True:
        ret, img = cap.read()
        img = detector.findHands(img)
        lmlist=detector.findpositions(img)
        if len(lmlist)!=0:
            print(lmlist[4])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (12, 45), cv2.FONT_HERSHEY_PLAIN, 3, (255, 250, 1), 3)
        cv2.imshow("IMAGE", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ != '__main__':
    pass
else:
    main()
