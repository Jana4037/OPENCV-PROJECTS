import cv2
import  mediapipe as mp
import time
vid=cv2.VideoCapture(0)   #r'C:\Users\ELCOT\Downloads\1.mp4'

mppose=mp.solutions.pose
pose=mppose.Pose()
mpDraw=mp.solutions.drawing_utils

ctime=0
ptime=0

while True:
    suc,img=vid.read()
    imgRBG=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRBG)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mppose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            #cv2.circle(img,(cx,cy),13,(255,255,0),cv2.FILLED)

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,56,56),3)
    cv2.imshow("IMAGE",img)
    key=cv2.waitKey(1)
    if key==ord('q'):
       break