"""
First script to be run.

- creates a subfolder called formatted_data
- load the original data from subfolder Data and process it
- append the formatted data - used later in the project - in formatted_data
"""

import numpy as np
import pandas as pd
import os

temperature=pd.read_csv("Data/temperature_history.csv",parse_dates=[["year","month","day"]])
load=pd.read_csv("Data/Load_history.csv",parse_dates=[["year","month","day"]],thousands=",")


#Formatting load data
print("Formatting load data...")
ts_load={"zone_id":[],"date":[],"hour":[],"load":[]}
for i in range(load.shape[0]):
    for hour in range(1,25):
        ts_load["zone_id"].append(load.at[i,"zone_id"])
        ts_load["date"].append(load.at[i,"year_month_day"])
        ts_load["hour"].append(hour)
        ts_load["load"].append(load.at[i,"h{}".format(hour)])

load_ts=pd.DataFrame(ts_load)

#Formatting temp data
print("Formatting temperature data...")
ts_temp={"station_id":[],"date":[],"hour":[],"temperature":[]}
for i in range(temperature.shape[0]):
    for hour in range(1,25):
        ts_temp["station_id"].append(temperature.at[i,"station_id"])
        ts_temp["date"].append(temperature.at[i,"year_month_day"])
        ts_temp["hour"].append(hour)
        ts_temp["temperature"].append(temperature.at[i,"h{}".format(hour)])

temp_ts=pd.DataFrame(ts_temp) 

load_ts["date_time"]= load_ts["date"]+pd.to_timedelta(load_ts.hour-1, unit='h')
temp_ts["date_time"]= temp_ts["date"]+pd.to_timedelta(temp_ts.hour-1, unit='h')

#Adding previous load/temperature effects
print("Adding previous load/temperature effects...")
for k in range(1,13):
    temp_ts["temperature_{}".format(k)]=temp_ts.groupby("station_id").shift(k).temperature
    load_ts["load_{}".format(k)]=load_ts.groupby("zone_id").shift(k).load
    
for k in [24,48]:
    load_ts["load_{}".format(k)]=load_ts.groupby("zone_id").shift(k).load    

load_ts["load_max_24"]=load_ts.groupby("zone_id").shift(1).load.rolling(24).max()
load_ts["load_min_24"]=load_ts.groupby("zone_id").shift(1).load.rolling(24).min()
load_ts["load_avg_168"]=load_ts.groupby("zone_id").shift(1).load.rolling(168).mean()
    
temp_ts["temp_max_24"]=temp_ts.groupby("station_id").shift(1).temperature.rolling(24).max()
temp_ts["temp_min_24"]=temp_ts.groupby("station_id").shift(1).temperature.rolling(24).min()
temp_ts["temp_max_24_48"]=temp_ts.groupby("station_id").shift(24).temp_max_24
temp_ts["temp_min_24_48"]=temp_ts.groupby("station_id").shift(24).temp_min_24
temp_ts["temp_avg_24"]=temp_ts.groupby("station_id").shift(1).temperature.rolling(24).mean()
temp_ts["temp_avg_24_48"]=temp_ts.groupby("station_id").shift(24).temp_avg_24
temp_ts["temp_avg_168"]=temp_ts.groupby("station_id").shift(1).temperature.rolling(168).mean()

#Adding calendar effects

print("Adding calendar effects...")
load_ts[["Monday","Tuesday","Wedesday","Thursday","Friday","Saturday","Sunday"]]=pd.get_dummies(load_ts.date.apply(lambda x:x.weekday()))

load_ts["previous_year_date"]=[pd.Timestamp("{}-12-31".format(x)) for x in load_ts.date.dt.year-1]
load_ts["days_delta"]=(load_ts.date-load_ts.previous_year_date).dt.days

load_ts["cos_time"]=np.cos(load_ts.days_delta*2*np.pi/365)
load_ts["sin_time"]=np.cos(load_ts.days_delta*2*np.pi/365)


print("Merging temp and load data...")
keys=["zone_id","station_id","hour","date"]
target=["load"]
calendar_effect=["cos_time","sin_time","Monday","Tuesday","Wedesday","Thursday","Friday","Saturday","Sunday"]
previous_demand_effect=["load_{}".format(i) for i in range(1,13)]+["load_24","load_48","load_max_24","load_min_24","load_avg_168"]
temperature_effect=["temperature"]+["temperature_{}".format(i) for i in range(1,13)]+["temp_max_24","temp_min_24","temp_max_24_48","temp_min_24_48","temp_avg_24","temp_avg_24_48","temp_avg_168"]

columns=keys+target+calendar_effect+previous_demand_effect+temperature_effect

os.mkdir("./formatted_data")

for zone in range(1,21):
    for station in range(1,12):
        for hour in range(1,25):
            local_load_ts=load_ts[(load_ts.zone_id==zone)&(load_ts.hour==hour)]
            local_temp_ts=temp_ts[(temp_ts.station_id==station)&(temp_ts.hour==hour)].groupby(["date","hour"]).first()
            local_temp_ts.drop(axis=1,columns=["date_time"],inplace=True)
            local_output=local_load_ts.join(other=local_temp_ts,how="left",on=["date","hour"])
            local_output=local_output[columns]
            local_output.dropna(inplace=True)

            local_output.to_csv("formatted_data/formatted_data_{}_{}_{}.csv".format(zone,station,hour))
            