#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 20:38:09 2017

@author: jjz
"""

import numpy as np
def forward_backward_prop(data,labels,params,dimensions):
    """
    只有一个输入层DX，一个隐藏层H，一个输出层DY。输出这个函数的loss和gradient。
    """
    ##设定参数W1，w2,b1,b2
    start=0
    DX,H,DY=(dimensions[0],dimensions[1],dimensions[2])
    w1=np.reshape(params[start:start+DX*H],(DX,H))
    start+=DX*H
    b1=np.reshape(params[start:start+H],(1,H))
    start+=H
    w2=np.reshape(params[start:start+H*DY],(H,DY))
    start+=H*DY
    b2=np.reshape(params[start:start+DY],(1,H))
    
    #forward net
    N=data.shape[0]
    l1=np.dot(data,w1)+b1
    h=log(l1)
    l2=np.dot(h,w2)+b2
    y_hat=softmax(l2)
    
    loss=-np.sum(labels*np.log(y_hat))/N#computer loss
    
    #computer gradient
    dl2=y_hat-labels
    dw2=np.dot(h.T,dl2)
    db2=np.sum(dl2,axis=0)
    
    dh=np.dot(dl2,w2.T)
    
    dl1=dh*sigmod_grad(h)
    dw1=np.dot(data.T,dl1)
    db1=np.sum(dl1,axis=0)
    
    gradw1=dw1/N
    gradb1=db1/N
    gradw2=dw2/N
    gradb2=db2/N
    #stack gradient
    grad=np.concatenate((gradw1.flatten(),gradb1.flatten(),gradw2.flatten(),gradb2.flatten()))
    return loss,grad
