import numpy as np
from sklearn import tree #决策树
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

class AdaBoost:
    dataSet = list()      #表示数据集
    dataMat = None
    trainMat = None
    resultMat = None
    testData = list()
    testTrainMat = None
    testResultMat = None
    NumOfDataSet = 0    #表示训练集的大小
    Dict = []           #字典集

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
        #print(newDataSet)
        #print(self.Dict)
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


    def Adaboost(self,X,y,num):            #进行训练

        trees = []                     #结果树
        treeWeights = []               #每棵树的权重
        Dweights = [] #D的比重,首先全是
        #print(weights)
        # 构造数据集成pandas结构
        attr_names = ['age', 'workclass', 'fnlwgt', 'education','education-num','marital-status','occupation','relationship'
            ,'race','sex','capital-gain','capital-loss','hours-per-week','native-country']  # 特征属性的名称
        attr_pd = pd.DataFrame(data=X, columns=attr_names)  # 每行为一个对象，每列为一种属性，最后一个为结果值
        # 将数据集中的字符串转化为代表类别的数字。因为sklearn的决策树只识别数字
        #print(attr_pd)
        #######构建树#######
        for i in range(num): #构建训练树
            if i == 0:
                tempWeight = [1.0/len(y) for k in range(len(y))]
            else:
                tempWeight = Dweights[i-1]
            # print("error!!",tempWeight[0])
            stump = tree.DecisionTreeClassifier(max_depth=6)
            stump.fit(attr_pd, y, sample_weight = tempWeight) #决策树进行训练
            predictions = stump.predict(X)  #预测结果

            #计算错误率
            error = sum([tempWeight[i] for i in range(len(predictions)) if predictions[i] != y[i][0]])
            #print("错误率",error)
            if error > 0.5 :
                break
            #计算Alpha
            alpha = 0.5 * np.log((1-error)/max(error,1e-16))
            print("错误率",error,"权重",alpha)
            trees.append(stump)        #增加树
            treeWeights.append(alpha)  #增加树的权重
            #更新D值
            temp_Sum = 0.0
            for j in range(0,len(y)):
                # print(predictions[j],y[j][0])
                temp_Sum += tempWeight[j] * np.exp(-alpha * y[j][0] * predictions[j])
                if predictions[j] == y[j][0]: #相同
                    tempWeight[j] = tempWeight[j] * np.exp(-alpha)
                else:
                    tempWeight[j] = tempWeight[j] * np.exp(alpha)

            for j in range(0,len(y)):
                tempWeight[j] = tempWeight[j]/ temp_Sum
            Dweights.append(tempWeight)

        return trees,treeWeights

    def adaboost_predict(self,X, y, trees, trees_weights):
        weighted_predictions = np.zeros(y.shape,dtype=float)
        Prob_Of_PredTrue = np.zeros(y.shape,dtype=float)
        #print("len:",len(weighted_predictions))
        #print(X)
        #print(y)
        for m in range(len(trees)):
            temptree = trees[m] #第M个树的决策树
            predictions = temptree.predict(X)   #其预测
            temp_tree_weight = trees_weights[m]      #其权重
            #print("第",m,"预测结果",predictions,"权重为",temp_tree_weight)
            res_predictions = [0.0 for i in range(0, len(X))]
            for i in range(len(X)):
                res_predictions[i] = float(predictions[i] * temp_tree_weight)  # 乘以权重
                weighted_predictions[i][0] += res_predictions[i]  # 加上权重

        for i in range(len(X)):
            Prob_Of_PredTrue[i][0] = 1 / (1 + np.exp(-weighted_predictions[i][0]))

        fpr, tpr, thresholds = roc_curve(y, Prob_Of_PredTrue, pos_label=1)
        auc_score = auc(fpr, tpr)
        return auc_score

    def adaboost_predict_2(self, X, y, trees, trees_weights):
        weighted_predictions = [0.0 for i in range(0, len(X))]

        for m in range(len(trees)):
            temptree = trees[m]  # 第M个树的决策树
            predictions = temptree.predict(X)  # 其预测
            temp_tree_weight = trees_weights[m]  # 其权重
            # print("第",m,"预测结果",predictions,"权重为",temp_tree_weight)
            # for i in range(len(y)):
            #     print(i,"结果为",predictions[i])
            res_predictions = [0.0 for i in range(0, len(X))]
            for i in range(len(X)):
                res_predictions[i] = float(predictions[i] * temp_tree_weight)  # 乘以权重
                weighted_predictions[i] += res_predictions[i]  # 加上权重

        #print(weighted_predictions)
        for i in range(len(X)):
            if weighted_predictions[i] < 0:
                weighted_predictions[i] = -1
            else:
                weighted_predictions[i] = 1

        #print(weighted_predictions)
        Acuarrcy = accuracy_score(y,np.array(weighted_predictions))
        print("准确率:",Acuarrcy)

    def cross_validation(self,times): #交叉验证
        auc_v = 0
        skf = KFold(n_splits=5) #五折交叉验证
        X = self.trainMat
        y = self.resultMat
        #print(X)
        #print(y)
        count = 0
        trees = []
        trees_weights = []
        for train_index, test_index in skf.split(X):
            #划分验证
            count += 1
            print("划分",train_index,test_index,count)
            train_feature_set, test_feature_set = np.array(X)[train_index], np.array(X)[test_index]
            train_label_set, test_label_set = np.array(y)[train_index],np.array(y)[test_index]
            #Adaboost算法
            trees, trees_weights = self.Adaboost(train_feature_set, train_label_set,times)
            auc_v += self.adaboost_predict(test_feature_set,test_label_set,trees,trees_weights)
        print("AUC_aver:",auc_v/5)
        #最后验证测试集
        self.adaboost_predict_2(self.testTrainMat, self.testResultMat, trees, trees_weights)
        return auc_v/5

if __name__ == '__main__':
    filename = './data/adult.data'
    adaboost = AdaBoost()
    adaboost.readData(filename)
    res_AUC = []
    for NumT in {45}:
        print("Num:",NumT)
        res_AUC.append(adaboost.cross_validation(NumT))
    # plt.plot([i for i in range(5,100,5)], res_AUC)
    # plt.show()