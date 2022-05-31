import sys
import numpy as np
import pandas as pd
from cwgradient import CWGradientBoosting
from multiprocessing import Pool
import os

#if len(sys.argv)!=3:
#    print("Please run scripts with following arguments:\n-zone\n-M")
    
#zone=int(sys.argv[1])
#M=int(sys.argv[2])

if not os.path.isdir("./important_features"):
    os.mkdir("./important_features")

def processing(spacetime):
    M, zone, Lambda, station, hour = spacetime[0], spacetime[1], spacetime[2], spacetime[3], spacetime[4]
    print("Now processing station {} and hour {} ...".format(station,hour))
    data=pd.read_csv("formatted_data/formatted_data_{}_{}_{}.csv".format(zone,station,hour))
    Y=data.load.values
    dates=data.date.values
    hours=data.hour.values
    zones=data.zone_id.values
    data.drop(columns=["Unnamed: 0","zone_id","station_id","hour","date","load"],inplace=True)
    # "temp_max_24_48.1","temp_avg_24.1" are removed
    data["One"]=np.ones(Y.size)
    X=data.values
    Y_train,Y_val,Y_test=Y[:-140],Y[-140:-70],Y[-70:]
    X_train,X_val,X_test=X[:-140],X[-140:-70],X[-70:]
    cwg=CWGradientBoosting(X_train,X_val,X_test,Y_train,Y_val,Y_test,0.15,M,Lambda,False)
    cwg.gradient_boosting()
    val=pd.DataFrame(data={"date":dates[-140:-70],"hour":hours[-140:-70],"pred":cwg.val_estimator,"values":Y_val})
    test=pd.DataFrame(data={"date":dates[-70:], "hour":hours[-70:],"pred":cwg.test_estimator, "values":Y_test})
    val_error=np.sum((cwg.val_estimator-Y_val)**2)/1e06

    f=open("val_errors/{}_{}_{}.txt".format(zone,station,hour), "w+") 
    # creates file if it doesn t already exist, otherwise overwrite 
    f.write(str(val_error))
    f.close()
    val.to_csv("val_pred/pred_{}_{}_{}.csv".format(zone,station,hour))
    test.to_csv("test_pred/pred_{}_{}_{}.csv".format(zone,station,hour))
    
    f=open("important_features/{}_{}_{}.txt".format(zone,station,hour),"w+")
    for x in cwg.features_count:
        f.write("{}:{}\n".format(x,cwg.features_count[x]))
    f.close()


if __name__ == "__main__":

    l = []
    # don t ask too much to your RAM
    for M in [20,50,100,200]:
        for zone in [4,9,13]:
            for Lambda in [0, 0.1, 1, 5]:
                for station in range(1,12):  # 1-> 12
                    for hour in range(1,25):  # 1 -> 24
                        l.append([M, zone, Lambda, station, hour])

    if not os.path.exists("val_errors"):
        os.mkdir("val_errors")
    if not os.path.exists("val_pred"):
        os.mkdir("val_pred")
    if not os.path.exists("test_pred"):
        os.mkdir("test_pred")

    print("HERE", len(l))

    # with Pool(len(l)+1) as p:
    with Pool(5) as p:  # 100? 1000?
        print(p.map(processing, l))

