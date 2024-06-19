''''
训练存储在数据库中的多个人脸数据:
	==> 每个面都应该有一个唯一的数字整数ID，如1、2、3等
	==> LBPH计算模型结果将保存在 trainer/ directory中。(如果不存在，请创建一个)
	==> 使用PIL，用“pip install pillow”安装pillow包

基于Anirban Kar的源代码: https://github.com/thecodacus/Face-Recognition

由Marcelo Rovai升级 - MJRoBot.org @ 21Feb18

'''

import cv2
import numpy as np
from PIL import Image
import os

# 人脸图像数据库的路径
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# 函数获取图像和标签数据
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # 将其转换为灰度
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# 将模型保存到trainer/trainer.yml中
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# 输出人脸训练数值，结束程序
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))