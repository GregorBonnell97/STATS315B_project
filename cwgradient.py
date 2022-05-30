import numpy as np
import pandas as pd
from pspline import PSplineRegressor
from multiprocessing import Pool

class CWGradientBoosting(object):
    
    def __init__(self,X_train,X_val,X_test,Y_train,Y_val,Y_test,v,M,Lambda,verbose=False,features_names="features.txt"):
        self.X_train=X_train
        self.X_val=X_val
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_val=Y_val
        self.Y_test=Y_test
        self.v=v
        self.M=M
        self.train_estimator=np.zeros(Y_train.size)
        self.val_estimator=np.zeros(Y_val.size)
        self.test_estimator=np.zeros(Y_test.size)
        self.step=0
        self.psp=PSplineRegressor(Lambda=Lambda) # deg = 3 by default
        self.verbose=verbose
        self.features_count,self.features_list=self.init_variable_count(features_names)
        
    
    def init_variable_count(self,features_names):
        f=open(features_names,"r+")
        l=f.readlines()
        l=[s.strip() for s in l]
        output={}
        for x in l:
            output[x]=0
        return output,l
    

    def train(self,x):
        u=self.Y_train-self.train_estimator
        a,pred,t=self.psp.fit(x,u)
        train_error=np.sum((u-pred)*(u-pred))
        return a,pred,train_error,t
    

    def run_step(self):
        best_a,best_pred,best_train_error,best_t,best_i=None,None,1e100,None,None
        # DOES NOT work. AssertionError: daemonic processes are not allowed to have children
        #l = [self.X_train[:,i] for i in range(self.X_train.shape[1])]
        #with Pool(len(l)+1) as p:
        #    U = p.map(self.train, l)
        #print("U =", U)
        #V = [U[i][2] for i in range(len(l))]
        #index = np.argmin(V)
        # best_a,best_pred,best_train_error,best_t,best_i=U[index],index

        for i in range(self.X_train.shape[1]): # goal: get best?
            # print("here ! self.X_train.shape[1] =", self.X_train.shape[1]) --> 47
            x=self.X_train[:,i]
            a,pred,train_error,t=self.train(x)
            if train_error<best_train_error:
                best_a,best_pred,best_train_error,best_t,best_i=a,pred,train_error,t,i
        if self.step==0:
            self.train_estimator+=best_pred
            self.val_estimator+=self.psp.predict(self.X_val[:,best_i],best_t,best_a)
            self.test_estimator+=self.psp.predict(self.X_test[:,best_i],best_t,best_a)
        else:
            self.train_estimator+=self.v*best_pred
            self.val_estimator+=self.v*self.psp.predict(self.X_val[:,best_i],best_t,best_a)
            self.test_estimator+=self.v*self.psp.predict(self.X_test[:,best_i],best_t,best_a)
        if self.step%10==0 and self.verbose:
            self.print_error()
        self.step+=1
        self.features_count[self.features_list[best_i]]+=1
        
    
    def gradient_boosting(self):   # cannot be parallelized
        for _ in range(self.M+1):  # sequential steps
            self.run_step()
        print("gradient boosting in cwgradient.py done!")
        
        
    def print_error(self):
        train_error=np.sqrt(np.mean((self.Y_train-self.train_estimator)**2))
        val_error=np.sqrt(np.mean((self.Y_val-self.val_estimator)**2))
        test_error=np.sqrt(np.mean((self.Y_test-self.test_estimator)**2))
        R2_train=1-np.mean((self.Y_train-self.train_estimator)**2)/np.std(self.Y_train)**2
        R2_val=1-np.mean((self.Y_val-self.val_estimator)**2)/np.std(self.Y_val)**2
        R2_test=1-np.mean((self.Y_test-self.test_estimator)**2)/np.std(self.Y_test)**2
        print("######\nStep:{}\nTrain error:{:.2f}\nVal error:{:.2f}\nTest error:{:.2f}\nR2_train:{:.2f}\nR2_val:{:.2f}\nR2_test:{:.2f}".format(self.step,train_error,val_error,test_error,R2_train,R2_val,R2_test))
        


if __name__=="__main__":

    data=pd.read_csv("formatted_data/formatted_data_2_5_21.csv")
    Y=data.load.values
    data.drop(columns=["Unnamed: 0","zone_id","station_id","hour","date","load","temp_max_24_48.1","temp_avg_24.1"],inplace=True)
    data["One"]=np.ones(Y.size)
    X=data.values
    Y_train,Y_val,Y_test=Y[:-140],Y[-140:-70],Y[-70:]
    X_train,X_val,X_test=X[:-140],X[-140:-70],X[-70:]
    print(np.mean(Y_test),np.mean(Y_val),np.mean(Y_train))
    cwg=CWGradientBoosting(X_train,X_val,X_test,Y_train,Y_val,Y_test,0.15,500,10,True)
    cwg.gradient_boosting()

