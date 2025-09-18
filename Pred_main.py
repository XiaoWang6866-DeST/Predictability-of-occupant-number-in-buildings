import numpy as np
import pandas as pd
from AIpred import Model
import time

def predict(data,t_res=1):
    # data: a time series list. E.g. data = pd.read_excel('raw_data.xlsx'). The data example can be found in https://doi.org/10.5281/zenodo.17151005.
    # t_res: the temporal resolution of the data time series, unit: hour. E.g. t_res=0.25 for a temporal resolution of 15-minute.
    if t_res>24:
        print('Temporal resolution not supported.')
        return None
    steps = round(24 / t_res)
    CaseModel = Model(n_per_day=steps,method='EN')
    for i in ['L2','L3','L4','L5','L6']:
        raw=data[i]
        new=[raw[steps*8:]]
        temp=[]
        for j in range(steps*8,len(raw)):
            temp.append([raw[j-1],raw[j-steps],raw[j-steps-1],raw[j-steps*7],raw[j-steps*7-1]])
        new.extend(np.array(temp).T)
        df = pd.DataFrame(data=np.array(new).T)
        df.to_excel(i+'input.xlsx')
        time.sleep(5)
        CaseModel.load_from_file(i+'input.xlsx')
        CaseModel.eval_get_result(to_file=i)
    return 0