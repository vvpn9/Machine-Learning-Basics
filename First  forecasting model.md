from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

## 数据预处理
1. 读取原始数据
2. 各种数据预处理
3. 构造一张集成好的数据大宽表

## 模型训练的三点

# 1. 数据格式
X_train是一个二维矩阵，存放m行样本，拥有n行特征
y_train是一个长度为m的一维数组
X_train和y_train的长度必须一致

# 2. 数据集拆分
训练模型用X_train和y_train，评估模型用X_test和y_test。
train和test是通过数据集的随机拆分完成的 —> test_size = (0 < something < 1)

# 3. 模型的评估
性能需要被评估，但模型评估不能使用训练集，必须在绝对没有参与模型训练的测试集上进行评估测试。
最常用的评估指标是正确率，但有时候只看正确率无法对模型进行全面评估。还有诸如以下的评估标准：
1. 误分类矩阵
2. 准确率，召回率，F1 Score等
3. ROC曲线，AUC值
4. Lorenz曲线
5. KS曲线，KS值

## 对新数据做出预测
1. 对新数据进行同样的数据预处理工作
2. 利用训练好的模型对新数据进行预测
3. 预测结果保存为文件