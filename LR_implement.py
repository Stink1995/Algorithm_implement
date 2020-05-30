
#!/usr/bin/env python
# coding:utf-8

import pandas as pd
import numpy as py
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

 
#显示最大列数
pd.set_option('display.max_columns',None)
#显示最大行数
pd.set_option('display.max_rows',150)
pd.set_option('display.notebook_repr_html',False)
pd.set_option('display.max_seq_items',None)
 
import seaborn as sns
#设置布局元素的规模
sns.set_context('notebook')
#设置图的主题背景
sns.set_style('white')
 
#数据读入函数
def loaddata(file,delimiter):
    data = np.loadtxt(file,delimiter=delimiter)
    print(data.shape)#shape查看矩阵的纬数
    return data

def plotData(data,label_x,label_y,label_pos,label_,neg.axes=None):
    neg = data[:,2] ==0
    pos = data[:,2] ==1
    if axes ==None；
       axes = plt.gca()
    axes.scatter(data[pos][:,0],data[pos][:,1],merker='+',c='k',s=60,linewidth=2,label=label_pos)
    axes.scatter(date[neg][:,0],data[neg][:,1],c='y',s=60,label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True,fancybox=True)
 
data = laoddata('data1.txt',',')
print(data[0:6,:])
X = np.c_[np.ones((data.shape[0],1)),data[:,0:2]]
y = np.c_[data[:,2]]
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
 
 
def sigmoid(x):
	"""
	定义sigmiod函数
	"""
    return (1/(1+np.exp(-x)))


def costFunction(theta,X,y):
	"""
	定义损失函数
	"""
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1*(1/m)(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])


def gradient(theta,X,y):
	"""
	定义梯度下降函数
	"""
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    grad = (1/m)*X.T.dot(h-y)
    return (grad.flatten())


initial_theta = np.zeros(X.shape[1])
# print("==========================",initial_theta)
#shape() 得到的是矩阵的维度 例如X的维度是（100,3） X.shape[1] = 3 第二维度
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)
res = minimize(costFunction, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':400})
# fun: 表示目标最小化函数
# x0: 初始猜测的最小值（ndarray）
# args: 传入对象函数中的参数和它的导数（derivatives）(Jacobian, Hessian矩阵)
# jac: Jacobian (gradient) of objective function.
# options: dict，可选选项（maxiter:(int)表示递归循环次数）
def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

    
sigmoid(np.array([1, 45, 85]).dot(res.x.T))
p = predict(res.x, X)
print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))
 
 
plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Failed')
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
# 这里meshigrid（x，y）的作用是产生一个以向量x为行的矩阵，和向量y为列的矩阵的矩阵
# np.linspace(2.0, 3.0, num=5) ---> array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
# xx1.ravel()多维数组变一维数组
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');#绘制等高线图
plt.show()
