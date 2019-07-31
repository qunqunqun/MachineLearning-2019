import pandas as pd
import numpy as np
from numpy import *
from sklearn.metrics import classification_report


def mysigmod(x):
    #print(x)
    return 1.0 / (1 + np.exp(-x))

class Solution:
    Trainset = []
    dataMat  = []
    labelMat = []
    Testset = []
    #变成每个class类
    NumOfOneClass = []
    resMat =mat( zeros((26,17)))
    #生成26个特征矩阵
    #读取文件
    #LR模型

    def readTrain_set(self):
        #读入数据
        self.Trainset =  pd.read_csv('train_set.csv')
        #print(self.Trainset)
        #转化为矩阵
        self.dataMat = mat(self.Trainset)
        self.labelMat = self.dataMat[:,16]
        self.dataMat = delete(self.dataMat,16,axis=1)
        self.dataMat = np.insert(self.dataMat,0,values=1,axis=1)
        print("改变后的数据矩阵",self.dataMat)
        self.dataMat = np.double(self.dataMat)
        self.normalize(self.dataMat)
        print(self.dataMat)

    def readTest_set(self):
        self.Testset = pd.read_csv('test_set.csv')

    def readRes_set(self,type):
        if type == 0:
            self.resMat = np.loadtxt('res_gra.txt')
        else:
            self.resMat = np.loadtxt('res_newton.txt')

    #将矩阵进行归一化
    def normalize(self,tempDataMat):
        max = []
        for i in range(shape(tempDataMat)[1]):
            tempMax = tempDataMat[0,i]
            for j in range(shape(tempDataMat)[0]):
                #print(tempDataMat[j,i])
                if(tempMax < tempDataMat[j,i]):
                    tempMax = tempDataMat[j,i]
            #放入最大值
            max.append(tempMax)
        #print(max)
        for i in range(shape(tempDataMat)[0]):
            for j in range(shape(tempDataMat)[1]):
                t = double(tempDataMat[i,j]) / double(max[j])
                #print(t)
                tempDataMat[i, j] = t

        #print(tempDataMat)


    #OvR进行矩阵分析
    def OvR(self,type):
        for i in range(26):
            #print(self.labelMat)
            if type == 0:
                temp = self.gradAscent(self.dataMat, self.labelMat, i + 1)
            else :
                temp = self.newton_method(self.dataMat, self.labelMat, i + 1)
            for index in range(17):
                self.resMat[i,index] = temp[index]
        print(self.resMat)
        return 0

    #梯度下降进行求解
    def gradAscent(self, dataMatrix, labelsMatrix ,inX):
        #对于label矩阵进行0-1转换
        print("第",inX,"开始！")
        temp_data =  mat(dataMatrix)
        temp_label = zeros(shape(labelsMatrix))
        for i in range(shape(labelsMatrix)[0]):
            temp_label[i][0] = labelsMatrix[i][0]
            #print("before",temp_label[i][0])
            if(temp_label[i][0] == inX):
                temp_label[i][0] = 1
            else:
                temp_label[i][0] = 0
            #print("after",temp_label[i][0])

        print(temp_label.transpose())
        m, n = shape(temp_data)
        alpha = 0.001
        maxCycles = 5000
        weights = ones((n, 1))  # n为矩阵dataMatIn的列数，也就是变量的个数
        for k in range(maxCycles):  # heavy on matrix operations
            h = mysigmod(dataMatrix * weights)  # matrix mult,计算z值
            error = (temp_label - h)  # vector subtraction，计算预测值域实际值的偏差
            weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult  #梯度下降算法，找出最佳的参数
        print(weights)
        return weights

    # 牛顿法进行求解
    def newton_method(self,dataMatrix, labelsMatrix,inX):
        #对于label矩阵进行0-1转换
        print("第",inX,"开始！")
        temp_data =  mat(dataMatrix)
        temp_label = zeros(shape(labelsMatrix))
        for i in range(shape(labelsMatrix)[0]):
            temp_label[i][0] = labelsMatrix[i][0]
            #print("before",temp_label[i][0])
            if(temp_label[i][0] == inX):
                temp_label[i][0] = 1
            else:
                temp_label[i][0] = 0
            #print(temp_label.transpose())
            #迭代次数
        count = 10
        xlen, ylen = shape(temp_data)
        # theta矩阵
        theta = zeros((ylen, 1))
        for i in range(count):
            print(i,"轮开始！")
            #计算预测值
            h = mysigmod((temp_data * theta))
            # .T 矩阵的转置
            # 一阶导数矩阵算法 (1 / m) * x.T * (h-y)
            J_theta =  (1.0 / xlen) * temp_data.transpose() * (h - temp_label)
            # 获取Hession矩阵
            # getA() 矩阵转换为数组
            # diag(x) 生成对角线为x其余为0的矩阵
            # Hession矩阵算法(1 / m) * x.T * U * x   U表示用(h * (1 - h))构成对角，其余为0的矩阵
            H = (1.0 / xlen) * (temp_data.T * diag(multiply(h, (1 - h)).T.getA() [0]) * temp_data)
            #.I 矩阵的逆
            theta = theta - H.I * J_theta
        print(theta)
        return theta

    def showResMatrix(self):
        print(self.resMat)

    def predictOneData(self,feature):
        typeOfClass = 0
        #print(self.resMat)
        p = 0
        res = 0
        for i in range(26):
            if  p < mysigmod(self.resMat[i,:] * feature.transpose()):
                p =  mysigmod(self.resMat[i,:] * feature.transpose())
                res = i + 1
        print(res)
        return typeOfClass

    def predictTestData(self):
        TP_26 = zeros(26)
        FN_26 = zeros(26)
        FP_26 = zeros(26)
        TN_26 = zeros(26)
        P = zeros(26)
        R = zeros(26)
        #开始测试
        TestMat =  mat(self.Testset)
        res_Of_test = TestMat[:,16]
        print("结果矩阵",res_Of_test)
        TestMat = delete(TestMat, 16, axis=1)
        data_Of_test = np.double(np.insert(TestMat, 0, values=1, axis=1))
        self.normalize(data_Of_test)
        print("数据矩阵", data_Of_test)
        res_Of_pre = zeros((shape(res_Of_test)))

        for i in range(shape(data_Of_test)[0]):
            #获取行数
            temp_data = mat(data_Of_test[i,:])
            #print(temp_data)
            r = 0
            num = 0
            for j in range(26):
                t = mysigmod(self.resMat[j,:] * temp_data.transpose())
                #进行数据的结果,分别进行预测
                #如果j预测为正例
                if t >= 0.5 :
                    if res_Of_test[i] == j + 1:
                        TP_26[j] += 1 #真正例
                    else :
                        FP_26[j] += 1 #假正例
                else :
                    if res_Of_test[i] == j + 1:
                        FN_26[j] += 1 #假反例
                    else:
                        TN_26[j] += 1 #假正例

                if r < t:
                  r = t
                  num = j + 1

            res_Of_pre[i] = num
            #print("单次结果",num)
        print(res_Of_pre)
        count_of_correct = 0
        for  i in range(shape(data_Of_test)[0]):
            if res_Of_pre[i] == res_Of_test[i]:
                count_of_correct += 1

        print("正确率为",count_of_correct/shape(data_Of_test)[0])
        #计算查全率和查准率
        macro_P = 0
        macro_R = 0
        for j in range(26):
            if TP_26[j] == 0:
                P[j] = 0
                R[j] = 0
            else:
                P[j] = double(TP_26[j]) / double(TP_26[j] + FP_26[j])
                R[j] = double(TP_26[j]) / double(TP_26[j] + FN_26[j])
            macro_P += P[j]
            macro_R += R[j]
        #总查全率和查准率
        macro_R = macro_R / 26
        macro_P = macro_P / 26
        print("查准率 macro_P：",macro_P," 查全率 macro_R：",macro_R," macro_F1：", (2 * macro_P * macro_R )/(macro_R + macro_P))
        TP_total = 0
        FP_total = 0
        FN_total = 0
        for j in range(26):
            TP_total += TP_26[j]
            FP_total += FP_26[j]
            FN_total += FN_26[j]
        TP_total = TP_total / 26
        FP_total = FP_total / 26
        FN_total = FN_total / 26
        micro_P = double(TP_total) / double(TP_total + FP_total)
        micro_R = double(TP_total) / double(TP_total + FN_total)
        print("查准率 micro_P：", micro_P, " 查全率 micro_R：", micro_R," micro_F1：", (2 * micro_P * micro_R )/(micro_P + micro_R))

        #print(classification_report(res_Of_test,res_Of_pre))

    def saveRes(self,type):
        if type == 0:
            np.savetxt("res_gra.txt",self.resMat,fmt='%.4f')
        else:
            np.savetxt("res_newton.txt", self.resMat, fmt='%.4f')



if __name__ == '__main__':
    solve = Solution()
    solve.readTrain_set()
    solve.readTest_set()
    #solve.saveRes()
    solve.OvR(1)
    solve.saveRes(1)
    solve.readRes_set(1)
    #solve.showResMatrix()
    solve.predictTestData()
