# coding=utf-8
# Scripts for AI-based load prediction

import os
import xlrd
import xlsxwriter
import numpy as np
import pandas as pd
from itertools import combinations
import warnings
import datetime

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

def mean_biased_error(real,pred):
    real=np.array(real)
    pred=np.array(pred)
    return np.mean(pred-real)

def _flatten(l):
    for ele in l:
        if not isinstance(ele, (list, tuple)):
            yield ele
        else:
            yield from _flatten(ele)


class _CombOut(object):
    def __init__(self, keys):
        self.combs = []
        for i in range(len(keys)):
            iter_comb = combinations(keys, i+1)
            for obj in iter_comb:
                self.combs.append(obj)
        self.nt = len(self.combs)
        self.n = -1

    def gen(self):
        self.n += 1
        if self.n < self.nt:
            return self.combs[self.n]
        else:
            return None


class _DatLoad(object):
    def __init__(self, input_data, n_per_day=24, trainday=30, keys=None):
        self.n_per_day = n_per_day
        self.trainday = trainday

        self.timestamp = input_data["timestamp"]
        self.loads = input_data["loads"]
        self.vars = input_data["in_vars"]

        self.ndays = len(self.timestamp)/n_per_day
        self.tvars = []
        if not keys:
            keys = self.vars.keys()
        self.key_comb = _CombOut(keys)

    def GetParams(self):
        r = self.key_comb.gen()
        # Check task complete signal
        if r is None:
            return None
        # Update vars
        t_keys = r
        #print("Current input parameters: %s" % str(r))
        keys = _flatten(t_keys)
        vars_temp = []
        key_tup = tuple(keys)
        for key in key_tup:
            vars_temp.append(self.vars[key])
        self.tvars = np.column_stack(vars_temp)
        return key_tup

    def GetData(self, start_day, pred_days):
        if start_day + pred_days > self.ndays:
            return None
        start_ind = int(round(start_day * self.n_per_day, 0))
        end_ind = int(round(start_ind + pred_days * self.n_per_day, 0))

        test_vars = self.tvars[start_ind: end_ind]
        test_loads = self.loads[start_ind: end_ind]
        test_time = self.timestamp[start_ind: end_ind]

        if start_day < self.trainday:
            train_ind = int((start_day + self.ndays - self.trainday) * self.n_per_day)
            train_loads = np.hstack((self.loads[train_ind:], self.loads[:start_ind]))
            train_vars = np.vstack((self.tvars[train_ind:], self.tvars[: start_ind]))
        else:
            train_ind = (start_day - self.trainday) * self.n_per_day
            train_loads = self.loads[train_ind: start_ind]
            train_vars = self.tvars[train_ind: start_ind]

        return train_vars, train_loads, test_vars, test_loads, test_time

# Main class for AI-based prediction
class Model(object):
    def __init__(self, n_per_day=24, method="EN",  train_days=30, update_days=1):
        self.n_per_day = n_per_day
        self.method = method
        self.train_days = train_days
        self.update_days = update_days
        self.para_comb = None
        self._raw_data = None
        self._preded_days = 0
        self.predictor = None

    # return models based on selections
    @staticmethod
    def _model_select(method):
        if method == "RF":
            model = RandomForestRegressor()
            tuned_parameters = {"max_depth": [6, 10, 15],
                                "n_estimators": [50, 100, 200],
                                "min_samples_split": [5, 20, 50],
                                "min_samples_leaf": [2, 5, 10]}
        elif method == "MLP":
            model = MLPRegressor()
            tuned_parameters = {"hidden_layer_sizes": [  (200,)*1,
                                                       (100,)*2,
                                                       (10,)*3,  ]}
        elif method == "EN":
            model = ElasticNet()
            tuned_parameters = {"alpha": [0.1, 1,2],
                                "l1_ratio": [0.1, 0.5,0.9],
                                "tol": [1E-3, 1E-4,1E-5],}
        else:
            raise Exception("Method not supported")
        return model, tuned_parameters

    @staticmethod
    def _get_vars_by_keys(in_vars, keys):
        keys = _flatten(keys)
        vars_temp = []
        for key in keys:
            vars_temp.append(in_vars[key])
        vars_array = np.column_stack(vars_temp)
        return vars_array

    # Base function for prediction
    def _predict(self, dLoader,parallel=-1):
        step=0
        n_day = 0
        pred_result = []
        real_result = []
        time_stamps = []
        comp_time=[]
        model_hypers=[]
        train_result=[]
        train_real=[]
        # Predict by day
        while True:
            data = dLoader.GetData(n_day + self.train_days, self.update_days)
            if not data:
                break
            train_vars, train_loads, test_vars, test_loads, test_time = data
            AI_model, AI_para = self._model_select(self.method)

            print('pred',step)
            step=step+1
            t1 = datetime.datetime.now()
            clf = GridSearchCV(AI_model, AI_para, scoring="neg_mean_squared_error", n_jobs=parallel, cv=5)
            clf.fit(train_vars, train_loads)
            model_hypers.append(clf.best_estimator_)
            pred_loads = clf.predict(test_vars)
            t1 = datetime.datetime.now() - t1
            comp_time.append(t1.seconds + t1.microseconds / 1000000)
            train_result.append(clf.predict(train_vars))
            train_real.append(train_loads)
            pred_result += pred_loads.tolist()
            real_result += test_loads.tolist()
            time_stamps += test_time
            n_day += self.update_days

        #err_rmse = np.sqrt(mse(real_result, pred_result))
        err_mae = mae(real_result, pred_result)
        return err_mae, pred_result, real_result, time_stamps, comp_time, model_hypers, train_result, train_real

    # Based function to get prediction results
    def _predict_get_result(self, dLoader,parallel=-1):
        _, pred_result, real_result, time_stamp, comp_time, model_hypers, train_result, train_real = self._predict(dLoader, parallel=parallel)
        return time_stamp, real_result, pred_result, comp_time, model_hypers, train_result, train_real

    # Load raw_data from an excel template file (.xls)
    def load_from_file(self, fpath):
        if os.path.exists(fpath):
            data=pd.read_excel(fpath)
            keys=data.columns
            timestamp = list(data[keys[0]])
            loads = np.array(data[keys[1]])
            in_vars = data[keys[2:]]
            self._raw_data = {
                "timestamp": timestamp,
                "loads": loads,
                "in_vars": in_vars
            }
            if self.para_comb==None:
                self.para_comb = list(keys[2:])
            return data
        else:
            raise Exception("File not exists")

    def eval_get_result(self, to_file=None, parallel=-1):
        dLoader = _DatLoad(
            input_data=self._raw_data,
            n_per_day=self.n_per_day,
            trainday=self.train_days,
            keys=(self.para_comb,),
        )
        dLoader.GetParams()
        timestamp, real_result, pred_result,comp_time, model_hypers, train_result, train_real = self._predict_get_result(dLoader,parallel=parallel)
        train_mbe=[]
        train_mape=[]
        train_mae=[]
        train_rmse=[]
        for i in range(len(train_real)):
            train_mbe.append(mean_biased_error(train_real[i],train_result[i])/np.mean(train_real[i]))
            train_mape.append(mape(train_real[i],train_result[i]))
            train_mae.append(mae(train_real[i],train_result[i])/np.mean(train_real[i]))
            train_rmse.append(mse(train_real[i],train_result[i])**0.5/np.mean(train_real[i]))
        if to_file:
            for k in range(len(pred_result)):
                pred_result[k] = max(0, round(pred_result[k]))
            writer = pd.ExcelWriter(str(to_file)+'.xlsx')
            df = pd.DataFrame(data={'time':timestamp,'real': real_result, 'pred': pred_result})
            df.to_excel(writer,index=False, sheet_name='pred_result')
            df = pd.DataFrame(data={'train_CVMBE': train_mbe,'train_MAPE': train_mape,  'train_MAE': train_mae, 'train_CVRMSE': train_rmse, 'comp_time': comp_time, 'model': model_hypers})
            df.to_excel(writer,index=False, sheet_name='train_info')
            writer._save()
        return None
