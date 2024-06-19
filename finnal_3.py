''''
实时人脸识别
	==> 每个脸存储在dataset/ dir，应该有一个唯一的数字整数ID，如1,2,3，等等
	==> LBPH计算模型(训练后的人脸数据)应该在trainer/ dir上

基于Anirban Kar的源代码: https://github.com/thecodacus/Face-Recognition

由Marcelo Rovai升级 - MJRoBot.org @ 21Feb18

'''

import cv2
import numpy as np
import os 
import RPi.GPIO as GPIO
import time



GPIO.setmode(GPIO.BCM) 
GPIO.setup(26, GPIO.OUT)
GPIO.output(26, GPIO.LOW)
GPIO.setup(19,GPIO.IN, pull_up_down=GPIO.PUD_UP)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# 开启id计数器
id = 0

# names 关联 ids: example ==> Marcelo: id=1,  等等
names = ['0 NewGuist']  # ！！！这里放您的用户名

                                   # 初始化并启动实时视频捕获
cam = cv2.VideoCapture(0)
cam.set(3, 640) # 设置视频宽度
cam.set(4, 480) # 设置视频高度

# 定义被识别为人脸的最小窗口大小
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    # img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    GPIO.output(26, 0)
    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # 检测是否置信度小于 100 ==> "0" 是完美匹配
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            GPIO.output(26, 1)
            print("gooood!")
            
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            GPIO.output(26, 0)
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    out_win="output_style_full_screen"
    cv2.namedWindow(out_win,cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty(out_win,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow(out_win, img)
    
    k = cv2.waitKey(10) & 0xff # 按下‘ESC’退出视频
    if k == 27:
        GPIO.output(26,0)
        break
    if (GPIO.input(19)==1):
        GPIO.output(26,0)
        cv2.destroyAllWindows()
        break

# 做一些清理
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
