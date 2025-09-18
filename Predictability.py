import numpy as np
from minepy import MINE
from math import log2

def cal_predictability_PIE(data,t_res=1,row=100):
    # Function to calculate the predictability of time series in PIE.
    # data: a time series list. E.g. data=[0,1,2,3,4,5,6,7,8,9].
    # t_res: the temporal resolution of the data time series, unit: hour. E.g. t_res=0.25 for a temporal resolution of 15-minute.
    # row: the level number (row number) of frequency matrix.
    if t_res>24:
        print('Temporal resolution not supported.')
        return None
    NperDay=round(24/t_res)
    data=np.array(data).reshape(-1,NperDay)
    days=len(data)
    N=[]
    for sub in data:
        mv=max(sub)
        temp=np.zeros(row)
        for val in sub:
            i=0
            while i<row:
                if val/mv<=(i+1)/row:
                    temp[i] += 1
                    break
                i+=1
        N.append(temp)
    X2Z=1/days
    HX=-days*X2Z*log2(X2Z)
    HXY=0
    Z = np.sum(N)
    N=np.array(N).ravel()/Z
    for i in range(len(N)):
        if N[i]:
            HXY-=N[i]*log2(N[i])
    return 1-(HXY-HX)/log2(row)

def cal_predictability_MIC(data,t_res=1):
    # Function to calculate the predictability of time series in MIC.
    # data: a time series list. E.g. data=[0,1,2,3,4,5,6,7,8,9].
    # t_res: the temporal resolution of the data time series, unit: hour. E.g. t_res=0.25 for a temporal resolution of 15-minute.
    mine = MINE()
    if t_res>24:
        print('Temporal resolution not supported.')
        return None
    NperDay=round(24/t_res)
    x=np.array(data[:-NperDay])
    y=np.array(data[NperDay :])
    mine.compute_score(x, y)
    return mine.mic()
