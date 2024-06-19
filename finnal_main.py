''''
实时人脸识别门禁
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

#5 channal 1, 23 channal 2

GPIO.setmode(GPIO.BCM)                            #引脚是BCM编号方式
GPIO.setup(5, GPIO.IN, pull_up_down=GPIO.PUD_UP)  #在5号引脚处设置上拉电阻
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)  #在5号引脚处设置上拉电阻
GPIO.setup(19,GPIO.IN, pull_up_down=GPIO.PUD_UP)
#26 channal led
GPIO.setup(26, GPIO.OUT)
GPIO.output(26, 0)                         #初始化，26号引脚设置为低电平
ledStatus = 1                                     #led灯的初始状态默认为暗
print("ready")
while True:
    if (GPIO.input(5)==1):
        print("dataset!")
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
            elif (GPIO.input(19)==1):
                cv2.destroyAllWindows()
                break

        # 做点清理工作
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

        import finnal_2

        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


        # 函数获取图像和标签数据
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L')  # 将其转换为灰度
                img_numpy = np.array(PIL_img, 'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)

            return faceSamples, ids


        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # 将模型保存到trainer/trainer.yml中
        recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

        # 输出人脸训练数值，结束程序
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
        GPIO.output(26,1)
        time.sleep(1)
        GPIO.output(26,0)
                
    
    if (GPIO.input(23)==1):
        print("Recgonition!")
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

GPIO.cleanup()

#except KeyboardInterrupt:
 #   GPIO.cleanup()