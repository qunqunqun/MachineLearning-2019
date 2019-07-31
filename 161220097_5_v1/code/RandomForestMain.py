import numpy as np
from sklearn import tree #决策树
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


class RandomForest:
    dataSet = list()      #表示数据集
    dataMat = None
    trainMat = None
    resultMat = None
    testData = list()
    testTrainMat = None
    testResultMat = None
    NumOfDataSet = 0    #表示训练集的大小
    countOfFutrue = 0
    Dict = []           #字典集

    def __init__(self,num):
        self.countOfFutrue = num

    def clearData(self):
        newDataSet = self.dataSet #数据拷贝
        newTestDate = self.testData
        for i in range(0,15):
            dic = dict()
            self.Dict.append(dic)
        for i in  {1,3,5,6,7,8,9,13}:
            Count = (1)
            for j in range(0,len(self.dataSet)): #进行数据处理
                if self.dataSet[j][i] == '?' : #缺失值处理
                    continue
                if self.dataSet[j][i] not in self.Dict[i]: #字典没有，则加入
                    self.Dict[i][self.dataSet[j][i]] = Count
                    Count += 1
        #更新返回集
        for i in range(0,15):
            for j in range(0,len(self.dataSet)):
                if i in {1,3,5,6,7,8,9,13}:
                    newDataSet[j][i] = self.Dict[i][newDataSet[j][i]]
                elif i != 14:
                    newDataSet[j][i] = int(newDataSet[j][i])

            for j in range(0,len(self.testData)):
                if i in {1,3,5,6,7,8,9,13}:
                    newTestDate[j][i] = self.Dict[i][newTestDate[j][i]]
                elif i != 14:
                    newTestDate[j][i] = int(newTestDate[j][i])
        return newDataSet,newTestDate

    def readData(self,filenamse):   #读取数据
        read = open(filenamse)
        for line in read:
            if line.__contains__("?"): #缺失值处理，缺失值直接去除
                continue
            line = line.strip('\n')
            temp = line.split(", ")
            self.dataSet.append(temp)
        read.close()
        # print(self.dataSet)
        print(self.dataSet.__len__())
        #测试集
        read = open("./data/adult.test")
        for line in read:
            if line.__contains__("?"): #缺失值处理，缺失值直接去除
                continue
            line = line.strip('\n')
            temp = line.split(", ")
            self.testData.append(temp)
        read.close()
        #数据修改
        for i in range(0,self.dataSet.__len__()):
            if self.dataSet[i][14] == '>50K':
                self.dataSet[i][14] = 1
            else:
                self.dataSet[i][14] = -1
        for i in range(0, self.testData.__len__()):
            if self.testData[i][14] == '>50K.':
                self.testData[i][14] = 1
            else:
                self.testData[i][14] = -1
        self.dataSet,self.testData = self.clearData()                             #数据清洗
        self.dataMat = np.mat(self.dataSet)         #训练集为矩阵
        self.trainMat = self.dataMat[:,0:14]
        self.resultMat = self.dataMat[:,14]

        self.testTrainMat = np.mat(self.testData)[:,0:14]
        self.testResultMat = np.mat(self.testData)[:,14]

    #TODO:实现Bootstrap抽样
    def BootStrap(self,DataSet):
        ResDataSet = []
        for i in range(len(DataSet)): #每次都生成一个样本
            Index  = random.randint(0, len(DataSet)-1) #每次从DataSet中采取一些
            ResDataSet.append(DataSet[Index])
        ResDataMat = np.mat(ResDataSet)     #变成矩阵
        return ResDataMat                   #返回结果

    #TODO:实现特征随机选取
    def ChooseFuture(self,NumOfFuture,NumOfRest):
        #选取NumOfRest个特征
        resultList = random.sample(range(0, NumOfFuture), NumOfRest)
        return resultList

    #TODO：随机森林
    def randomForest(self,DataSet,num):             #进行训练
        trees = []                                  #决策树的集合
        FeatureList = []                            #训练的特征值的
        # 将数据集中的字符串转化为代表类别的数字。因为sklearn的决策树只识别数字
        for i in range(num): #构建训练树
            # 每次都随机选取8个特征，且数据集为放回抽样
            BootstrapMat =  self.BootStrap(DataSet)    #获得结果矩阵
            tempX = BootstrapMat[:,0:14]    #采样后的数据矩阵
            tempY = BootstrapMat[:,14]      #采样后的结果矩阵
            #选取特征行
            resultFutureList = self.ChooseFuture(14,10)  #从14个特征选个特征得到的列表
            resultFutureList.sort()
            #print("选择的特征值",resultFutureList)
            FeatureList.append(resultFutureList)        #放入
            #构建决策树并加入
            tempXextra = tempX[: ,[i for i in resultFutureList]]
            stump = DecisionTreeClassifier(random_state=5)
            resTree = stump.fit(tempXextra, tempY)
            trees.append(resTree)   #放入

        return trees,FeatureList

    #TODO：随机森林算法的AUC计算
    def AUC_of_Random(self,Data,trees,FeatureList):
        #print(Data)
        length = len(trees)
        X = Data[:,0:14]            #数据矩阵
        y = Data[:,14]              #标签矩阵
        resLabel = []
        for i in range(0,length):
            tempTree = trees[i] #决策树
            tempList = FeatureList[i]
            tempX = X[:,[i for i in tempList]] #数据矩阵
            tempResLabel = tempTree.predict(tempX)  #计算
            resLabel.append(tempResLabel)
        ProbLabel = [0.0 for i in range(len(Data))] #概率矩阵
        #进行投票
        for i in range(len(Data)):
            for j in range(length):
                ProbLabel[i] += resLabel[j][i]
            ProbLabel[i] = 1 / (1 + np.exp(-ProbLabel[i]))
        print(ProbLabel)
        fpr, tpr, thresholds = roc_curve(y, np.array(ProbLabel), pos_label=1)
        auc_score = auc(fpr, tpr)
        return auc_score

    #TODO：测试集上的验证
    # TODO：随机森林算法的AUC计算
    def ACURACY_of_Random_test(self, Data, trees, FeatureList):
        # print(Data)
        length = len(trees)
        X = Data[:, 0:14]  # 数据矩阵
        y = Data[:, 14]  # 标签矩阵
        resLabel = []
        for i in range(0, length):
            tempTree = trees[i]  # 决策树
            tempList = FeatureList[i]
            tempX = X[:, [i for i in tempList]]  # 数据矩阵
            tempResLabel = tempTree.predict(tempX)  # 计算
            resLabel.append(tempResLabel)
        ProbLabel = [0.0 for i in range(len(Data))]  # 概率矩阵
        # 进行投票
        for i in range(len(Data)):
            for j in range(length):
                ProbLabel[i] += resLabel[j][i]
        tempPro = 1 / (1 + np.exp(-np.array(ProbLabel)))

        fpr, tpr, thresholds = roc_curve(y, tempPro, pos_label=1)
        auc_score = auc(fpr, tpr)
        print("测试集AUC为:", auc_score)

        for i in range(len(Data)):
                if ProbLabel[i] > 0:
                    ProbLabel[i] = 1
                else:
                    ProbLabel[i] = -1
                #print(ProbLabel[i])
        acr = accuracy_score(y,np.array(ProbLabel))
        print("测试集准确度为:",acr)
        return 0

    # 交叉验证
    def cross_validation(self,times):
        auc_v = 0
        skf = KFold(n_splits=5)
        count = 0
        X = self.dataSet
        trees = []
        FeatureList = []
        for train_index, test_index in skf.split(X):
            #划分验证
            count += 1
            print("划分",train_index,test_index,count)
            train_Data_set, test_Data_set = np.array(X)[train_index], np.array(X)[test_index]
            trees , FeatureList = self.randomForest(train_Data_set,times)   #训练测试集,获得训练树
            auc_v += self.AUC_of_Random(test_Data_set,trees,FeatureList)                  #计算AUC

        auc_v = auc_v / 5
        print("训练集auc_v",auc_v)

        self.ACURACY_of_Random_test(np.mat(self.testData),trees,FeatureList)
        return auc_v

if __name__ == '__main__':
    filename = './data/adult.data'
    countOfFutureChoosen = 5 #每次只选择五个特征值
    randomForest = RandomForest(countOfFutureChoosen) #构造函数
    randomForest.readData(filename)     #读取数据
    res_AUC = []
    for NumT in  {75}:                    #测试，i为基学习器的数量
        print("Num:",NumT)
        res_AUC.append( randomForest.cross_validation(NumT))
    # plt.plot([i for i in range(5, 101, 5)], res_AUC)
    # plt.show()