# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:03:31 2022

@author: Gregor
"""

import sys
import pandas as pd
import numpy as np
import pylab as plt

if len(sys.argv)!=2:
    print("Please run scripts with following arguments:\n-zone")
    
zone=int(sys.argv[1])

val_errors=[0 for i in range(11)]

for station in range(1,12):
    for hour in range(1,24):
        f=open("val_errors/{}_{}_{}.txt".format(zone,station,hour),"r")
        val_errors[station-1]+=float(f.read())
        f.close()

best_station=np.argmax(val_errors)+1

dfs_val=[]
dfs_test=[]

for hour in range(1,24):
    df_val=pd.read_csv("val_pred/pred_{}_{}_{}.csv".format(zone,best_station,hour),parse_dates=["date"])
    df_test=pd.read_csv("test_pred/pred_{}_{}_{}.csv".format(zone,best_station,hour),parse_dates=["date"])
    dfs_val.append(df_val)
    dfs_test.append(df_test)


df_val=pd.concat(dfs_val)
df_test=pd.concat(dfs_test)

df_val["datetime"]=df_test["date"]+pd.to_timedelta(df_test.hour-1, unit='h')
df_test["datetime"]=df_test["date"]+pd.to_timedelta(df_test.hour-1, unit='h')

df_val.sort_values("datetime",inplace=True)
df_test.sort_values("datetime",inplace=True)
        
plt.scatter(df_val.datetime.values,df_val["values"].values,color="b")
plt.plot(df_val.datetime.values,df_val.pred.values,color="r")
plt.grid()
plt.title("pred, real values vs time, zone={}, validation set".format(zone))
#plt.locator_params(axis="x",nbins=5)
plt.xlim((df_val.datetime.values[0],df_val.datetime.values[-1]))
plt.savefig("val_plots/{}.png".format(zone))
plt.clf()

plt.hist(np.minimum(2,np.abs(df_val["values"]-df_val.pred)/df_val["values"]),bins=50)
plt.grid()
plt.title("Distribution of relative error for validation set,zone={}".format(zone))
plt.savefig("val_plots/residual_{}.png".format(zone))
plt.clf()


plt.scatter(df_test.datetime.values,df_test["values"].values,color="b")
plt.plot(df_test.datetime.values,df_test.pred.values,color="r")
plt.grid()
plt.title("pred, real values vs time, zone={}, test set".format(zone))
#plt.locator_params(axis="x",nbins=5)
plt.xlim((df_test.datetime.values[0],df_test.datetime.values[-1]))
plt.savefig("test_plots/{}.png".format(zone))
plt.clf()

plt.hist(np.minimum(2,np.abs(df_test["values"]-df_test.pred)/df_test["values"]),bins=50)
plt.grid()
plt.title("Distribution of relative error for test set,zone={}".format(zone))
plt.savefig("test_plots/residual_{}.png".format(zone))
plt.clf()

R2_val=1-np.mean((df_val["values"]-df_val.pred)**2)/np.std(df_val["values"])**2
R2_test=1-np.mean((df_test["values"]-df_test.pred)**2)/np.std(df_test["values"])**2

print("R2_val:{:.2f}\nR2_test:{:.2f}".format(R2_val, R2_test))

f=open("R2/R2_{}.txt".format(zone),"w")

f.write("{:.2f}".format(R2_val))
f.write("\n{:.2f}".format(R2_test))
f.close()