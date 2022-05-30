# -*- coding: utf-8 -*-
"""
Script that computes a simple autoregressive model as benchmark and report its performance
"""

import pandas as pd
import numpy as np
import pylab as plt
import os

load=pd.read_csv("Data/Load_history.csv",parse_dates=[["year","month","day"]],thousands=",")

ts_load={"zone_id":[],"date":[],"hour":[],"load":[]}
for i in range(load.shape[0]):
    for hour in range(1,25):
        ts_load["zone_id"].append(load.at[i,"zone_id"])
        ts_load["date"].append(load.at[i,"year_month_day"])
        ts_load["hour"].append(hour)
        ts_load["load"].append(load.at[i,"h{}".format(hour)])

load_ts=pd.DataFrame(ts_load)

for k in range(1,3):
    load_ts["load_{}".format(k)]=load_ts.groupby("zone_id").shift(k).load
    
load_ts.dropna(inplace=True)

if not os.path.isdir("./plots_simple_data"):
    os.mkdir("./plots_simple_data")

f=open("plots_simple_data/R2.txt","w+")
for zone in range(1,21):
    print("Now processing zone {}...".format(zone))
    loc_data=load_ts[load_ts.zone_id==zone]
    Y=loc_data.load.values
    X=loc_data[["load_1","load_2"]].values
    Y_train,Y_test=Y[:-1680],Y[-1680:]
    X_train,X_test=X[:-1680],X[-1680:]
    X_inv=np.linalg.inv(X_train.T.dot(X_train))
    beta=X_inv.dot(X_train.T.dot(Y_train))
    Y_pred=X_test.dot(beta)
    plt.hist(np.abs(Y_pred-Y_test)/Y_test,bins=50)
    plt.grid()
    plt.title("Empirical distribution of relative residuals for simple model, zone={}".format(zone))
    plt.savefig("plots_simple_data/hist_simple_{}.png".format(zone))
    plt.clf()
    R2=1-np.mean((Y_test-Y_pred)**2)/np.std(Y_test)**2
    f.write(str(R2)+'\n')

f.close()
    