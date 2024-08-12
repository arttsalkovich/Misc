import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import models
import pytorch_models
import util
import tensorflow as tf
import keras

# List of files to read from or to write to
save_path_stationary = 'data/data_stationary.csv'
save_path_stationary_normalized = 'data/data_stationary_normalized.csv'
save_no_FRED_path_stationary  = 'data/data_no_FRED_stationary.csv'
save_no_FRED_path_stationary_normalized  = 'data/data_no_FRED_stationary_normalized.csv'
save_graph1_path ='yields_time_series.png' 
save_graph2_path = 'diff_yields_time_series.png' 

# Global parameters for training/validation/test sets proportions
training_proportion = 0.70
validation_proportion = 0.15
test_proportion = 0.15
y_types = ['5_YR_R', '7_YR_R', '10_YR_R'] #list of names for predicted parameters 
SEED_ID = 42 # Random state

def load_dataset (GENERATE_NEW_DATA=True, FRED_MD = True):
    if (GENERATE_NEW_DATA == True):       
        # Creates new database using the input data from diferent .csv files. The output data is stationary, but not normalized
        df = util.create_dataset(FRED_MD) 

        # Addition of time lags
        columns_to_lag = [ ] #  Add if needed (e.g.'5_YR_R', '7_YR_R', '10_YR_R', '3_MO', '6_MO', '1_YR', '2_YR', '3_YR', '5_YR', '10_YR', '20_YR')
        lags_to_apply = 1 
        print(f'Adding lags {lags_to_apply} for the columns {columns_to_lag}')
        df = util.add_lags(df, columns_to_lag, lags_to_apply) # we use lag 1 as there is very low autocorrelation
        df = df.iloc[lags_to_apply:] # deleting the first lines with NaN data 
        
        # Normalizaiton of dataset 
        data_mean = df.mean()
        data_std = df.std()
        df_normalized = (df - data_mean) / data_std

        if FRED_MD == True:
            df.to_csv(save_path_stationary, index=True)
            df_normalized.to_csv(save_path_stationary_normalized, index=True) 
        else:
            df.to_csv(save_no_FRED_path_stationary, index=True)
            df_normalized.to_csv(save_no_FRED_path_stationary_normalized, index=True) 
    else:
        # Read from the CSV file, where the data was previously saved
        if FRED_MD == True:
            df = pd.read_csv(save_path_stationary_normalized) 
        else:
            df = pd.read_csv(save_no_FRED_path_stationary_normalized) 

    return df

def split_data(df):
    # Extracting the array of features x_data and the array of target data y_data (real yields)
    x_data = df[df.columns.difference(['Date', '5_YR_R', '7_YR_R', '10_YR_R'])].apply(pd.to_numeric, errors='coerce').values
    y_data = df[y_types].values

    # #Shuffled data - BE CAREFUL
    # permutation = np.random.permutation(len(x_data))
    # x_data = x_data[permutation, :]
    # y_data = y_data[permutation, :]

    # Splitting the data into 3 sub-sets as per the global parameters 
    total_samples = len(x_data) 
    split1 = int(training_proportion * total_samples)
    split2 = int((training_proportion + validation_proportion ) * total_samples)

    x_train = x_data[:split1,:]
    x_valid = x_data[split1:split2,:]
    x_test = x_data[split2:,:] 

    y_train = y_data[:split1,:]
    y_valid = y_data[split1:split2,:]
    y_test = y_data[split2:,:]   
    
    x_train_valid = np.concatenate((x_train, x_valid), axis=0)
    y_train_valid = np.concatenate((y_train, y_valid), axis=0) 

    return  x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test


def main(selected_models):
    # Parameter controlling whether the new data is being generated or not 
    GENERATE_NEW_DATA = False
    FRED_MD = False
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Controlling randomness 
    # np.random.seed(SEED_ID)
    # tf.random.set_seed(SEED_ID)
    # keras.utils.set_random_seed(SEED_ID)"Dense_NN_3layers_model_optimization"

    # Load dataset (stationary and normalized)
    df = load_dataset(GENERATE_NEW_DATA, FRED_MD) # the flag showing whether FRED_MD data is included 

    # Splitting the data into three subsets
    x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test = split_data(df)

    # Grouping arguments to reduce redundancy
    data_splits = (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test)

    ### SUPERVISED LEARNING METHODS ###
    all_models = {
        "PCA_regression": lambda: models.PCA_regression(*data_splits),
        "OLS_regression": lambda: models.OLS_regression(*data_splits),
        "Ridge_regression": lambda: models.Ridge_regression(*data_splits),
        "Lasso_regression": lambda: models.Lasso_regression(*data_splits),
        "Elastic_Lasso_regression": lambda: models.Elastic_Lasso_regression(*data_splits),
        "AdaBoost": lambda: models.AdaBoost(*data_splits, FRED_MD),
        "GradientBoost": lambda: models.GradientBoost(*data_splits, FRED_MD),
        "XGBoost_reg": lambda: models.XGBoost_reg(*data_splits, FRED_MD),
        "RandomForest": lambda: models.RandomForest(*data_splits, FRED_MD),
        "ExtraTrees": lambda: models.ExtraTrees(*data_splits, FRED_MD),
        "Dense_NN_1layer": lambda: models.Dense_NN_1layer(*data_splits, FRED_MD), 
        "Dense_NN_2layers": lambda: models.Dense_NN_2layers(*data_splits, FRED_MD),
        "Dense_NN_3layers": lambda: models.Dense_NN_3layers(*data_splits, FRED_MD),
        "Dense_NN_3layers_model_search": lambda: pytorch_models.Dense_NN_3layers_model_search(*data_splits, ['5_YR_R']),
        "Dense_NN_3layers_model_optimization": lambda: pytorch_models.Dense_NN_3layers_model_optimization(*data_splits, ['10_YR_R']),
        "LSTM_model_optimization": lambda: pytorch_models.LSTM_model_optimization(*data_splits, ['5_YR_R'], 5)
    }

    # Execute the selected models
    for model_name in selected_models:
        if model_name in all_models:
            all_models[model_name]()
        else:
            print(f"Model {model_name} not recognized.")


    ### Additional functions ###
    if False:
        # Plot graphs for real yield timeseries 
        util.real_yield_plot(save_graph1_path, save_graph2_path)

        # Autocorrelation analysis for time series. Switch to True, if needed
        df = pd.read_csv(save_path_stationary) 
        time_series_code = '5_YR_R'
        util.autocorrelation_analysis(df[time_series_code], 10, time_series_code+'_autocorrelation.png')

        # Heatmap for the correlation matrix
        util.correlation_heatmap( )
     
        # Convergence graphs 
        GradientBoost_plot(*data_splits)
        XGBoost_reg_plot(*data_splits)

    print ('-=END=-')   


if __name__ == '__main__': 
    selected_models = [
    "RandomForest"
    ]

    # "PCA_regression"
    # "OLS_regression"
    # "Ridge_regression"
    # "Lasso_regression"
    # "Elastic_Lasso_regression"
    # "AdaBoost"
    # "GradientBoost"
    # "XGBoost_reg"
    # "RandomForest"
    # "ExtraTrees"
    # "Dense_NN_1layer"
    # "Dense_NN_2layers"
    # "Dense_NN_3layers"
    # "Dense_NN_3layers_model_search"
    # "Dense_NN_3layers_model_optimization"
    # "LSTM_model_optimization"

    main(selected_models)

