#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @File    : Test.py
# @Author  : MoonKuma
# @Date    : 2019/4/20
# @Desc   : 

from model.model import *

#  Train
train_data_array = load_data('data/iris_training.csv')
print('data_array.shape', train_data_array.shape)
model2 = test_model2(train_data_array, min(train_data_array.shape[0], 50), 1000) # model2 works perfect with accuracy near 100%
model1 = test_model(train_data_array, train_data_array.shape[0], 1000) # don't understand why but model 1 won't work
model3 = test_model3(train_data_array, train_data_array.shape[0], 1000) # Adding L2 regularization here, however this won't help any
model4 = test_model4(train_data_array, train_data_array.shape[0], 1000) # Adding drop out strategy, won't help either, maybe the problem is not about over-fitting


#  Test
test_data_array = load_data('data/iris_test.csv')
X = test_data_array[:,0:4].T
Y = test_data_array[:,4].reshape(1,test_data_array.shape[0])
predict(model2, X, Y)
predict(model1, X, Y)
predict(model4, X, Y)
# model 2 reach 100% accuracy(train) with current setting
# model 1 may be two complicated
