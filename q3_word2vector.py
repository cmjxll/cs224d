#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:18:24 2017

@author: jjz
"""
import numpy as np
def normalizeRows(x):
    """
    normalize rows
    """
    x=(x.T/(np.sqrt(np.sum(x.T**2,axis=0))+1e-30)).T
    return x


def softmaxCostAndGradient(predicted,target,outputvectors,dataset):
     """
     computer gradient and cost of word2vector assuming the softmax prediction
     function and cross entropy loss
     
     #Inputs:
     #-predicted:numpy ndarray, predicted word vector
     #-target:integer,the index of the target word
     #-outputVectors:"output"vectors(as row )for all tokens
     #-dataset: needed for negative sampling
     
     #outputs:
     #-cost: cross entropy cost for softmax word prediction
     #-gradPred: the gradient with respect to predicted word vector
     #grad: the gradient with respect to all the other word vectors
     """
     N, D=outputvectors.shape
     
     cita=np.dot(outputvectors,predicted.T).T #computer cita dimension 1*N
     probs=softmax(cita)

    
     #computer cost
     cost=-np.log(probs[target])

     #computer gradient
     probs[target]-=1#dimension N*1
     grad=probs.T*predictd
     gradPred=(probs.shape(1,N)*outputvectots).flatten()
     return  cost,gradPred,grad
 
    
def negSamplingCostAndGradient(predicted,target,outputvectors,dataset,K=10):
    N,D =outputvectors.shape
    gradPred=np.zeros_like(predicted)
    grad=np.zeros_like(outputvectors)
    
    
    #negtive-sample
    neg_sample=[]
    for i in range(K):
        neg_idx=dataset.sampleTokenIdx()
        while neg_ix==target:#remove target
            neg_idx=dataset.sampleTokenIdx()
        neg_sample+=[neg_idx]
    index[0]=target
    index+=neg_sample
    labels=np.array([1]+[-1 for i in range(K)])
    
    neg_vec=outputvectors[index]
    #transform matrix multiply
    muti=(neg_vec.reshape(K+1,D).dot(predicted.reshape(D,1)))*labels
    probs=sigmoid(muti)
    
    #cost
    cost=-np.sum(np.log(probs))
    
    #computer gradient
    gradPred=((probs-1)*labels).reshape(1,K+1).dot(neg_vec.reshape(K+1,D)).flatten()
    gradtemp=(probs-1).reshape(K+1,1).dot(predicted.reshape(1,D))
    for i in range(K+1):
        grad[index[i],:]=gradtemp[i,:]
    return cost,gradPred,grad


def skipgram(currentWord,C,contextWords,tokens,inputvectors,outputvectors,
             dataset,word2vecCostAndGradient=softmaxCostAndGradient):
    """
    #implement the skip-gram model in this function
    #Inputs:
    #- currentWord:a string of the current center word
    #- C:integer,context size
    #- contextWords: list of no more than 2*C strings, the context words
    #-tokens: a dictionary that maps words to their indeices 
    #         in the word vector list
    #-inputvectors: "input" word vectors(as rows) for all tokens
    #-outputvectors: "output" word vectors(as rows) for all tokens
    #-word2vecCostAndGradient: the cost and gradient function for a prediction
    #
    #outputs:
    # - cost: the cost function value for the skip-gram model
    # - grad: the gradient with respect to the word vectors
    """
    gradIn=np.zeros()
    gradOut=np.zeros(outputvectors)
    cost=0
    c_idx=tokens[currentWord]
    predicted=inputvectors[c_idx]
    
    for i in contextWords:
        target=tokens[i]
        c_cost,gradPred,grad=word2vecCostAndGradient(predicted,target,outputvectors,dataset)
        cost+=c_cost
        gradIn[c_idx,:]+=gradPred
        gradOut+=grad
        
    return cost,gradIn,gradOut


def word2vec_sgd_wrapper(word2vecModel,tokens,wordVectors,dataset,C,
                         word2vectorCostAndGradient=softmaxCostAndGradient):
    """
    implement word2vector with sgd
    """
    bachsize=50
    cost=0.0
    grad=np.zeros_like(wordVectors)
    N=wordVectors.shape[0]
    inputvectors=wordVectors[:N/2,:]
    outputvectors=wordVectors[N/2:,:]
    #sgd
    for i in range(bachsize):
        C1=random.randint(1,C)
        currentWord, contextWord=dataset.getRandomContext(C1)
        if word2vecModel==skipgram:
            denom=1
        else:
            denom=1
            
        c, gin, gout=word2vecModel(currentWord, C1,contextWord, tokens, inputvectors,
                                      outputvectors,dataset)
        cost+=c/batchsize/demo
        grad[:N/2,:]+=gin/batchsize/demo
        grad[N/2,:]+=gout/batchszie/demo
        
    return cost, grad
