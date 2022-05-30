import sys
import numpy as np
import pandas as pd
from cwgradient import CWGradientBoosting
from multiprocessing import Pool

if len(sys.argv)!=3:
    print("Please run scripts with following arguments:\n-zone\n-M")
    
zone=int(sys.argv[1])
M=int(sys.argv[2])

def processing(spacetime):
    station, hour = spacetime[0], spacetime[1]
    print("Now processing station {} and hour {} ...".format(station,hour))
    data=pd.read_csv("formatted_data/formatted_data_{}_{}_{}.csv".format(zone,station,hour))
    Y=data.load.values
    dates=data.date.values
    hours=data.hour.values
    zones=data.zone_id.values
    data.drop(columns=["Unnamed: 0","zone_id","station_id","hour","date","load","temp_max_24_48.1","temp_avg_24.1"],inplace=True)
    data["One"]=np.ones(Y.size)
    X=data.values
    Y_train,Y_val,Y_test=Y[:-140],Y[-140:-70],Y[-70:]
    X_train,X_val,X_test=X[:-140],X[-140:-70],X[-70:]
    cwg=CWGradientBoosting(X_train,X_val,X_test,Y_train,Y_val,Y_test,0.15,M,10,False)
    cwg.gradient_boosting()
    val=pd.DataFrame(data={"date":dates[-140:-70],"hour":hours[-140:-70],"pred":cwg.val_estimator,"values":Y_val})
    test=pd.DataFrame(data={"date":dates[-70:], "hour":hours[-70:],"pred":cwg.test_estimator, "values":Y_test})
    val_error=np.sum((cwg.val_estimator-Y_val)**2)/1e06
    f=open("val_errors/{}_{}_{}.txt".format(zone,station,hour),"w")
    f.write(str(val_error))
    f.close()
    val.to_csv("val_pred/pred_{}_{}_{}.csv".format(zone,station,hour))
    test.to_csv("test_pred/pred_{}_{}_{}.csv".format(zone,station,hour))


if __name__ == "__main__":

    l = []
    for station in range(1,12):
        for hour in range(1,24):
            l.append([station, hour])
            # processing(station, hour)

    print("hello !!!")

    with Pool(len(l)+1) as p:
        print(p.map(processing, l))
        