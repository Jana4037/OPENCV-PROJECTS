import cv2
import mediapipe as mp
import time

vid=cv2.VideoCapture(0)

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
facemesh =mpFaceMesh.FaceMesh()
drawspec=mpDraw.DrawingSpec(thickness=1,circle_radius=2,color=(199,20,89))

ptime=0

while True:
    suc,img=vid.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=facemesh.process(imgRGB)

    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,facelms,mpFaceMesh.FACEMESH_FACE_OVAL,drawspec,drawspec)
            for id,lm in enumerate(facelms.landmark):
                ih,iw,ic=img.shape
                x,y=int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)


    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(24,255,79),3)
    cv2.imshow("IMAGE",img)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
