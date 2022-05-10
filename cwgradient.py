# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:16:39 2022

@author: Gregor
"""

import numpy as np
import pandas as pd
from pspline import PSplineRegressor

class CWGradientBoosting(object):
    
    def __init__(self,X_train,X_test,Y_train,Y_test,v,M,Lambda):
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        self.v=v
        self.M=M
        self.train_estimator=np.zeros(Y_train.size)
        self.test_estimator=np.zeros(Y_test.size)
        self.step=0
        self.psp=PSplineRegressor(Lambda=Lambda)
    
    def train(self,x):
        u=self.Y_train-self.train_estimator
        a,pred,t=self.psp.fit(x,u)
        train_error=np.sum((u-pred)*(u-pred))
        return a,pred,train_error,t
    
    def run_step(self):
        best_a,best_pred,best_train_error,best_t,best_i=None,None,1e100,None,None
        for i in range(self.X_train.shape[1]):
            x=self.X_train[:,i]
            a,pred,train_error,t=self.train(x)
            if train_error<best_train_error:
                best_a,best_pred,best_train_error,best_t,best_i=a,pred,train_error,t,i
        if self.step==0:
            self.train_estimator+=best_pred
            self.test_estimator+=self.psp.predict(self.X_test[:,best_i],best_t,best_a)
        else:
            self.train_estimator+=self.v*best_pred
            self.test_estimator+=self.v*self.psp.predict(self.X_test[:,best_i],best_t,best_a)
        
        if self.step%10==0:
            self.print_error()
        self.step+=1
        
    
    def gradient_boosting(self):
        for m in range(self.M+1):
            self.run_step()
        print("Done!")
        self.print_error()
        
        
    def print_error(self):
        train_error=np.sqrt(np.mean((self.Y_train-self.train_estimator)**2))
        test_error=np.sqrt(np.mean((self.Y_test-self.test_estimator)**2))
        R2_train=1-np.mean((self.Y_train-self.train_estimator)**2)/np.std(Y_train)**2
        R2_test=1-np.mean((self.Y_test-self.test_estimator)**2)/np.std(Y_test)**2
        print("######\nStep:{}\nTrain error:{:.2f}\nTest error:{:.2f}\nR2_train:{:.2f}\nR2_test:{:.2f}".format(self.step,train_error,test_error,R2_train,R2_test))
    


if __name__=="__main__":
    data=pd.read_csv("formatted_data/formatted_data_2_5_21.csv")
    Y=data.load.values
    data.drop(columns=["Unnamed: 0","zone_id","station_id","hour","date","load"],inplace=True)
    data["One"]=np.ones(Y.size)
    X=data.values
    Y_train,Y_test=Y[:-168],Y[-168:]
    X_train,X_test=X[:-168],X[-168:]
    print(np.mean(Y_test),np.mean(Y_train))
    cwg=CWGradientBoosting(X_train,X_test,Y_train,Y_test,0.15,500,10)
    cwg.gradient_boosting()
    
    
    
    