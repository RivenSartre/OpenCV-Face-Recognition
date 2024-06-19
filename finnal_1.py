''''
从多个用户获取多个面部数据，并存储在一个数据库(数据集目录)上
	==> 面部数据将存在一个目录下: dataset/ (如果不存在的话，请建一个)
	==> 每个面部数据都有一个唯一的数字整数ID，如1、2、3等

基于Anirban Kar的源代码: https://github.com/thecodacus/Face-Recognition

由Marcelo Rovai升级 - MJRoBot.org @ 21Feb18

'''

import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # 设置视频宽度
cam.set(4, 480) # 设置视频高度

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 对于每个人，输入一个数字面部id
face_id = 0

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# 初始化个人面部样本计数
count = 0

while(True):

    ret, img = cam.read()
    #img = cv2.flip(img, -1) # 垂直翻转视频图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(faces)<=1:
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            cv2.putText(img,str(count),(40,50),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),1)
            # 将捕获的图像保存到数据集文件夹中
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
    elif len(faces)>1:
        cv2.putText(img,"Please do not stand in front of the camera",(40,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv2.putText(img,"with more than one person at a time",(40,70),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        count=0
    
    out_win="output_style_full_screen"
    cv2.namedWindow(out_win,cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(out_win,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow(out_win, img)

    k = cv2.waitKey(100) & 0xff # 按'ESC'退出视频
    if k == 27:
        break
    elif count >= 100: # 取30张脸样本，停止录像
         break

# 做点清理工作
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


