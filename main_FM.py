import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from fastFM import als
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error

def read_excel(filePath):
    df = pd.read_excel(filePath, sheet_name='Sheet1_user_dt')
    df_1 = df.dropna()
    drop_colume = ['userid',
                   'email',
                   'sn',
                   'username',
                   'reg_time',
                   'birthday',
                   'reg_type',
                   'reg_state',
                   'country',
                   'city',
                   'area',
                   'num_country',
                   'num_city',
                   'num_area',
                   'sell_type',
                   'sell_reason',
                   'sell_target']

    df_2 = df_1.drop(drop_colume, axis=1)
    return df_2

def binary_encoding(df, header_list, target_header):
    for header in header_list:
        if header != target_header:
            criteria = np.mean(df[header])
            df[header] = np.where(df[header] > criteria, 1, 0)
        else:
            pass
    return df

def onehot_encoding(df, header_list):
    oh_list_all = []
    for header in header_list:
        oh_list_0 = np.where(df[header] == 1, 1, 0)
        oh_list_1 = oh_list_0.copy()
        for i in range(len(oh_list_1)):
            if oh_list_1[i] == 0:
                oh_list_1[i] = 1
            else:
                oh_list_1[i] = 0
        oh_list_0 = np.vstack((oh_list_0))
        oh_list_1 = np.vstack((oh_list_1))
        oh_list = np.hstack((oh_list_0, oh_list_1))
        oh_list_all.append(oh_list)
    oh_list_all = np.hstack((oh_list_all))
    return oh_list_all

def split_data(df):
    train, test = train_test_split(df, test_size=0.3)
    target_factor = 'sc_day_month'
    drop_factor = ['used_day_month', 'used_freq_day', 'used_day_month',
                   'sc_times', 'no_sc_times', 'sc_days', 'no_sc_days', 'sc_freq_day']
    train_1 = train.drop(drop_factor, axis=1)
    test_1 = test.drop(drop_factor, axis=1)
    y_train = train_1[target_factor]
    x_train = train_1.drop(target_factor, axis=1)
    y_test = test_1[target_factor]
    x_test = test_1.drop(target_factor, axis=1)

    header_list = x_train.columns
    oh_x_train = onehot_encoding(x_train, header_list)
    oh_x_test = onehot_encoding(x_test, header_list)
    return x_train, y_train, x_test, y_test, oh_x_train, oh_x_test

def split_data_2(df):
    train, test = train_test_split(df, test_size=0.1)
    #print("train = ", train)
    target_factor = 'sc_day_month'
    drop_factor = ['used_day_month', 'used_freq_day', 'used_day_month',
                   'sc_times', 'no_sc_times', 'sc_days', 'no_sc_days', 'sc_freq_day']
    train_1 = train.drop(drop_factor, axis=1)

    y_train = train_1[target_factor]
    x_train = train_1.drop(target_factor, axis=1)
    return x_train, y_train

def saveModel(model_name, model_fit):
    with open(str(model_name)+'.pickle', 'wb') as model:
        pickle.dump(model_fit, model)

def loadModel(model_name):
    with open(model_name+'.pickle', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

if __name__ == "__main__":
    filePath = 'user_data.xlsx' # replace with your file
    df = read_excel(filePath)
    header_list = df.columns
    target_header = 'sc_day_month'
    category_df = binary_encoding(df, header_list, target_header)

    x_train, y_train, x_test, y_test, oh_x_train, oh_x_test = split_data(category_df)

    # convert to sparse matrix representation
    Sparse_oh_x_train = csc_matrix(oh_x_train)
    Sparse_oh_x_test = csc_matrix(oh_x_test)

    # Build and train a Factorization Machine
    fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=5, l2_reg_w=0.1, l2_reg_V=0.5)
    fm.fit(Sparse_oh_x_train, y_train)
    y_pred = fm.predict(Sparse_oh_x_test)

    mse = mean_squared_error(np.asarray(y_test).flatten(), y_pred)
    rmse = np.sqrt(mse)
    print("rmse = ", rmse)
    print("fm.V_ = ", fm.V_)

    '''
    # Compare the performance with Linear Regression
    clf = LinearRegression().fit(Sparse_oh_x_train, np.vstack((y_train)))
    y_pred_linReg = clf.predict(Sparse_oh_x_test)

    mse_linReg = mean_squared_error(np.asarray(y_test).flatten(), y_pred_linReg)
    rmse_linReg = np.sqrt(mse_linReg)

    print("mse_linReg = ", mse_linReg)
    print("rmse_linReg = ", rmse_linReg)

    print("y_test = ", np.asarray(y_test).flatten())
    print("y_pred = ", y_pred)
    print("y_pred_linReg = ", np.asarray(y_pred_linReg).flatten())
    '''

