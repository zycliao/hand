# -*- coding: utf-8 -*-

'''
利用opencv自带的基于深度学习训练的函数来做人脸识别，准确率比Haar cascades要高
'''
import numpy as np
import cv2
import os, glob, random


class FaceRecog(object):

    def __init__(self, folder='our_faces', sampleCount=8, modelPath="deploy.prototxt.txt",
                 weightPath="res10_300x300_ssd_iter_140000.caffemodel", confidence=0.8):
        # 定义相关的路径参数
        self.modelPath = modelPath
        self.weightPath = weightPath
        self.confidence = confidence  # 置信度参数，高于此数才认为是人脸，可调
        self.folder = folder
        self.sampleCount = sampleCount

    def predict(self, img):
        xTrain_, yTrain, xTest_, yTest = self.loadImageSet()
        num_train, num_test = xTrain_.shape[0], xTest_.shape[0]
        xTrain, data_mean, V = self.pca(xTrain_, 32)
        xTest = np.array((xTest_ - np.tile(data_mean, (num_test, 1))) * V)  # 得到测试脸在特征向量下的数据
        ###SVM方法###
        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_LINEAR)

        svm.train(xTrain.astype(np.float32), cv2.ml.ROW_SAMPLE, yTrain.astype(np.int32))

        y_val2 = [svm.predict(np.ravel(d)[None, :]) for d in xTest.astype(np.float32)]

        right = 0.0
        for i in range(len(y_val2)):
            if np.int(y_val2[i][1]) == yTest[i]:
                right += 1
                print y_val2[i][1]
        acc = right / len(y_val2)
        print u'支持向量机识别率: %.2f%%' % (acc * 100)

        # 单张图片识别
        image = img  # 最好图片名不用中文
        # image = cv2.imread(image)
        # image = cv2.resize(image , (640,480))
        resImage, (startX, startY, endX, endY) = self.face_detector(image)
        # if resImage is None:

        image_crop = image[startY:endY, startX:endX]
        # cv2.imshow("Output1", resImage)
        # cv2.imshow("Output2", image_crop)
        cv2.imwrite("./faceio/res_test4.jpg", resImage)
        cv2.imwrite("./faceio/crop_4.jpg", image_crop)
        # cv2.waitKey(0)
        d = "./faceio/crop_4.jpg"
        img_crop = cv2.imread(d.encode('gbk'), 0)
        # img_crop = image_crop.convert('L')

        # img_crop = img_crop.resize((112,92), Image.ANTIALIAS)
        img_crop = cv2.resize(img_crop, (112, 92))
        img_crop = np.array((np.ravel(img_crop) - data_mean) * V)
        result = svm.predict(np.ravel(img_crop.astype(np.float32))[None, :])

        print result[1]
        result = int(result[1])

        name_dictionary = {0: 'Dingnan', 1: 'JiWeibo', 2: 'TuoWanchen', 3: 'WangZhaowei', 4: 'ZhangHuayue',
                           5: 'LiaozhouYingcheng'}
        print 'U are ' + name_dictionary[result] + '~'

        greeting_dict = ['Hello~ ', 'You look great today ', 'Nice to meet U ', 'HaHa, I got U! ',
                         'You can\'t hide from me ']
        g = random.randint(0, 4)
        greeting = greeting_dict[g]

        text = name_dictionary[result]

        # 如果检测脸部在左上角，则把标签放在图片内，否则放在图片上面
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(resImage, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.putText(resImage, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow('resImage', resImage)
        cv2.waitKey()
        print greeting

    def face_detector(self, image):
        net = cv2.dnn.readNetFromCaffe(self.modelPath, self.weightPath)
        # 输入图片并重置大小符合模型的输入要求
        (h, w) = image.shape[:2]  # 获取图像的高和宽，用于画图
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()  # 预测结果
        (startX, startY, endX, endY) = (0, 0, 0, 0)

        # 可视化：在原图加上标签和框
        max_condidence = 0
        for i in range(0, detections.shape[2]):
            res_confidence = detections[0, 0, i, 2]
            if res_confidence > max_condidence:
                max_condidence = res_confidence

        for i in range(0, detections.shape[2]):
            # 获得置信度
            res_confidence = detections[0, 0, i, 2]

            # 选择置信度最大的像素
            if res_confidence == max_condidence:
                # 获得框的位置
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # 在图片上写上标签
                text = "{:.2f}%".format(res_confidence * 100)

                # 如果检测脸部在左上角，则把标签放在图片内，否则放在图片上面
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                # cv2.putText(image, text, (startX, y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        resImage = image
        return resImage, (startX, startY, endX, endY)

    def pca(self, data, k):
        data = np.float32(np.mat(data))
        rows, cols = data.shape  # 取大小
        print rows, cols
        data_mean = np.mean(data, 0)  # 求均值
        Z = data - np.tile(data_mean, (rows, 1))
        D, V = np.linalg.eig(Z * Z.T)  # 特征值与特征向量
        V1 = V[:, :k]  # 取前k个特征向量
        V1 = Z.T * V1
        print np.shape(V1)
        for i in xrange(k):  # 特征向量归一化
            V1[:, i] /= np.linalg.norm(V1[:, i])
        return np.array(Z * V1), data_mean, V1

    # 加载图像集，随机选择sampleCount张图片用于训练
    def loadImageSet(self):  # our_faces文件夹地址
        trainData = []
        testData = []
        yTrain = []
        yTest = []
        for k in range(5):
            folder2 = os.path.join(self.folder, 's%d' % (k + 1))
            data = []
            for d in glob.glob(os.path.join(folder2, '*.jpg')):
                im = cv2.imread(d.encode('gbk'), 0)
                im = cv2.resize(im, (112, 92))
                data.append(im)
            sample = random.sample(range(10), self.sampleCount)
            trainData.extend([data[i].ravel() for i in range(10) if i in sample])
            testData.extend([data[i].ravel() for i in range(10) if i not in sample])
            yTest.extend([k] * (10 - self.sampleCount))
            yTrain.extend([k] * self.sampleCount)

        return np.array(trainData), np.array(yTrain), np.array(testData), np.array(yTest)



if __name__ == '__main__':

    face = FaceRecog()
    face.predict('./4.jpg')
    ####  PCA  ####


# os.remove("res_test9.jpg")
# os.remove("res_test9_crop.jpg")

