#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from functools import  reduce

class Perceptron(object):
    '''
    感知器类
    '''

    def __init__(self,Input_Numb,activator):
        """
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        """
        self.activator = activator
        self.weights = [0.0] * Input_Numb
        self.bias=0.0

    def __str__(self):
        """
        打印学习到的权重、偏置项
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights,self.bias)

    def Pridict(self,input_vec):
        #  z=x1*w1+x2+w2+...xn*wn
        """
        将两个向量x和y按元素相乘
        """
        return ActivationFunction(sum(map(lambda x_y: x_y[0] * x_y[1], zip(input_vec, self.weights))) + self.bias)


    def Train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self.OneIteration(input_vecs, labels, rate)

    def OneIteration(self,input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            print('vect=', input_vec, '\tlabel=', label, end=" ")
            output=self.Pridict(input_vec)
            print('\tresult=', output)
            self.UpdataWeight(input_vec, output, label, rate)

    def UpdataWeight(self,input_vec, output, label, rate):
        """
        按照感知器规则更新权重
        """
        print('更新前self.weights======', self.weights)
        print('更新前self.bias======', self.bias)
        # 首先计算本次更新的delta
        # 然后把input_vec[x1,x2,x3,...]向量中的每个值乘上delta，得到每个权重更新
        # 最后再把权重更新按元素加到原先的weights[w1,w2,w3,...]上
        deltaTY = label - output
        print('deltaTY=',deltaTY)
        detaB=rate * deltaTY
        print('detaB=', detaB)
        detaW=list(map(lambda a,b:a*b,input_vec,[detaB]*len(input_vec)))
        print('detaW=', detaW)
        self.weights=list(map(lambda a,b:a+b,self.weights,detaW))
        self.bias += detaB

        print('更新后self.weights=====', self.weights)
        print('更新后self.bias====', self.bias)
        # 更新bias


def GetTrain_Data():
    '''
    获取训练数据----元组
    '''
    IntPut_Vector=[[1,1,1],[1,1,0],[0,0,1],[0,1,1],[1,0,0],[0,0,0]]
    Lables=[1,1,1,1,1,0]
    return IntPut_Vector,Lables

def ActivationFunction(z):
    '''
    感知器激活函数---阶跃函数
    '''
    return 1 if z > 0 else 0


def Tain_or_Perceptron():
    '''
    数据训练
    '''
    # 创建感知器，输入参数个数为3（因为and是二元函数），激活函数为f
    Ptron = Perceptron(3, ActivationFunction)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = GetTrain_Data()
    Ptron.Train(input_vecs, labels, 10, 0.1)
    # 返回训练好的感知器
    return Ptron


if __name__=="__main__":
    # 训练and感知器
    and_perception = Tain_or_Perceptron()
    # 打印训练获得的权重
    print(and_perception)
    # 测试
    print('1 or 1 or 1= %d' % and_perception.Pridict([1, 1, 1]))
    print('0 or 0 or 0= %d' % and_perception.Pridict([0, 0, 0]))
    print('1 or 0 or 1= %d' % and_perception.Pridict([1, 0, 1]))
    print('0 or 1 or 0= %d' % and_perception.Pridict([0, 1, 0]))
