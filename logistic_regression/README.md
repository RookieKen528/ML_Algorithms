# 逻辑回归

## Description

This is a simple implementation of Logistics Regression for **learning purpose**.

- using Batch or Stochastic Gradient Descent (SGD).
- using breast cancer toy dataset from sklearn to train the model.

## File:

- logistic_regression.ipynb : jupyter notebook version with explanation of the code.
- logistic_regression.py : python file that only contains the logistic regression class.

## Interface:

- logistic_regression(): create instance of logistic regression class.
- fit(): train model with input data.
- predict(): used to calculate prediction or probabilistic values.

## metric functions:

- precision()
- misclassification rate()
- recall()
- f1_score()

## 原理

- 如果下面的数学公式无法正确显示，请在chrome浏览器安装插件：MathJax Plugin for Github

### 1. 线性回归模型
- x：特征矩阵

- w：权值向量，同线性模型的参数向量

- b：截距
  $$
  y=x\cdot w+b
  $$

### 2. 二分类逻辑回归模型
- Y：标签

- 左侧为Y=1的条件概率
  $$
  P(Y=1|x)=\frac{1}{1+e^{-(x\cdot w+b)}}
  $$

### 3. 极大似然与损失函数
- 设：
  $$
  P(Y=1|x)=p(x)
  $$

  $$
  P(Y=0|x)=1-p(x)
  $$

  

- 通过极大似然使得概率值逼近标签值
    - xi:一个样本
    
    - yi：i样本对应的真实标签
      $$
      MLE=ArgMax \prod [p(x_{i})]^{y_{i}}*[1-p(x_{i})]^{1-y_{i}}
      $$
    
- 损失函数 logloss
    - n：样本数
      $$
      LogLoss = ArgMin-\frac{1}{n}(\sum_{i=1}^{n}(y_{i}*ln(p(x_{i}))+(1-y_{i})*ln(1-p(x_{i})))
      $$

### 4. 梯度下降
$$
g_{i} = \frac{\partial LogLoss}{\partial w}=\frac{1}{n}\sum_{i=1}^{n}x_{i}*(p(x_{i})-y_{i})
$$

- 通过迭代，更新w得到最优损失函数
    - j：第j轮迭代
    
    - lambda：学习率
      $$
      w_{i}^{j+1}=w_{i}^{j}-\lambda g_{i}
      $$

### 5. 逻辑回归最终预测模型
$$
LogisticModel=\left \lbrace
\begin{aligned}
1 \quad \frac{1}{1+e^{-(x\cdot w+b)}}>0.5 \\
-1  \quad \frac{1}{1+e^{-(x\cdot w+b)}}\leq 0.5
\end{aligned}
\right.
$$

## Contact

Email: ken1997528@hotmail.com

WeChat: 799557629
