import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf
import keras
import xgboost as xgb


from tensorflow.keras import layers
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable


y_types = ['5_YR_R', '7_YR_R', '10_YR_R'] #list of names for predicted parameters 
SEED_ID = 42
tf.random.set_seed(SEED_ID)
keras.utils.set_random_seed(SEED_ID)
np.random.seed(SEED_ID)


@ignore_warnings(category=ConvergenceWarning)

def PCA_regression (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test):
    # PCA regression
    print('-=PCA regression=-')
    for i, y_i in enumerate(y_types):
        # Create and fit the PCA regression model
        pca = PCA(n_components=8)
        x_pca = pca.fit_transform(np.vstack((x_train_valid, x_test)))

        # Splitting the dataset 
        x_train_valid_pca = x_pca[:len(x_train_valid),:]
        x_test_pca = x_pca[len(x_train_valid):,:]

        # Explained variance per principal component
        mask = np.ones((len(pca.explained_variance_ratio_),len(pca.explained_variance_ratio_)))
        mask_lower = np.tril(mask,0)
        sum_pca = np.einsum('ij,j->i',mask_lower, pca.explained_variance_ratio_)
        #print(f'Explained variance ratio {pca.explained_variance_ratio_}')
        print(f'Explained variance ratio (sum of components) {sum_pca}')

        # Run OLS
        reg_PCA_OLS = linear_model.LinearRegression()
        reg_PCA_OLS.fit(x_train_valid_pca, y_train_valid[:,i])
        #print(reg_PCA_OLS.coef_)

        # Predict on the train+validation and test sets
        y_pred_train_valid = reg_PCA_OLS.predict(x_train_valid_pca)
        y_pred_test = reg_PCA_OLS.predict(x_test_pca)

        # Output data
        print(
            f'PCA regression ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train_valid[:,i], y_pred_train_valid):.3f} and '
            f'R-squared = {r2_score(y_train_valid[:,i], y_pred_train_valid):.3f}')
        
        print(
            f'PCA regression ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}') 

def OLS_regression (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test):
    # OLS regression
    print('-=OLS regression=-')
    for i, y_i in enumerate(y_types):
        # Create and fit the OLS model
        reg_OLS = linear_model.LinearRegression()
        reg_OLS.fit(x_train_valid, y_train_valid[:,i])

        # Predict on the train+validation and test sets
        y_pred_train_valid = reg_OLS.predict(x_train_valid)
        y_pred_test = reg_OLS.predict(x_test)
        #print(reg_OLS.coef_)

        # Output data
        print(
            f'OLS regression ({y_i}) for train set: MSE ='
            f' {mean_squared_error(y_train_valid[:,i], y_pred_train_valid):.3f} and '
            f'R-squared = {r2_score(y_train_valid[:,i], y_pred_train_valid):.3f}') 
        print(
            f'OLS regression ({y_i}) for test set: MSE = '
            f'{mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}') 

def Ridge_regression (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test):
    # Ridge regression  
    print('-=Ridge regression=-')
    for i, y_i in enumerate(y_types):
        mse_scores = []
        r2_scores = []
        alphas = np.logspace(-6, 6, 13)

        for alpha in alphas:
            # Create and fit the Ridge model
            reg_ridge = linear_model.Ridge(alpha=alpha)
            reg_ridge.fit(x_train, y_train[:,i]) 

            # Predict on the validation set
            y_pred = reg_ridge.predict(x_valid)

            # Calculate MSE and R-squared
            mse = mean_squared_error(y_valid[:,i], y_pred)
            r2 = r2_score(y_valid[:,i], y_pred)

            # Append scores to the lists
            mse_scores.append(mse)
            r2_scores.append(r2)

        #for alpha, mse, r2 in zip(alphas, mse_scores, r2_scores):
            #print(f"Alpha = {alpha:.5f}: MSE = {mse:.3f}, R-squared = {r2:.3f}")

        # Fitting the model with the alpha parameter producing the highest r2_score on the validation set
        print(f'Alpha with the highest R-squared = {alphas[np.argmax(r2_scores)]:.3f}')
        reg_ridge = linear_model.Ridge(alpha=alphas[np.argmax(r2_scores)])
        reg_ridge.fit(x_train, y_train[:,i]) 

        # Predict on the train+validation and test sets
        y_pred_train = reg_ridge.predict(x_train)
        y_pred_valid = reg_ridge.predict(x_valid)
        y_pred_test = reg_ridge.predict(x_test)
        print(
            f'Ridge regression ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}')
        print(
            f'Ridge regression ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}')
        print(
            f'Ridge regression ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}')

def Lasso_regression (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test):
    # Lasso regression
    print('-=Lasso regression=-')
    for i, y_i in enumerate(y_types):
        mse_scores = []
        r2_scores = []
        alphas = np.logspace(-4, 4, 9)

        for alpha in alphas:
            # Create and fit the Lasso model
            reg_lasso = linear_model.Lasso(alpha=alpha)
            reg_lasso.fit(x_train, y_train[:,i]) 

            # Predict on the validation set
            y_pred = reg_lasso.predict(x_valid)

            # Calculate MSE and R-squared
            mse = mean_squared_error(y_valid[:,i], y_pred)
            r2 = r2_score(y_valid[:,i], y_pred)

            # Append scores to the lists
            mse_scores.append(mse)
            r2_scores.append(r2)

        # for alpha, mse, r2 in zip(alphas, mse_scores, r2_scores):
        #     print(f"Alpha = {alpha:.5f}: MSE = {mse:.3f}, R-squared = {r2:.3f}")

        # Fitting the model with the alpha parameter producing the highest r2_score on the validation set
        print(f'Alpha with the highest R-squared = {alphas[np.argmax(r2_scores)]:.3f}')
        reg_lasso = linear_model.Lasso(alpha=alphas[np.argmax(r2_scores)])
        reg_lasso.fit(x_train, y_train[:,i]) 

        # Predict on the train+validation and test sets
        y_pred_train = reg_lasso.predict(x_train)
        y_pred_valid = reg_lasso.predict(x_valid)
        y_pred_test = reg_lasso.predict(x_test)

        print(
            f'Lasso regression ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}')

        print(
            f'Lasso regression ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}')

        print(
            f'Lasso regression ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}')

def Elastic_Lasso_regression (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test):
    # Elastic Lasso regression
    print('-=Elastic Lasso regression =-')
    for i, y_i in enumerate(y_types):
        alphas = np.logspace(-5, 5, 11)
        l1_ratios = np.linspace(0.00, 0.50, 25, endpoint=False)
        mse_scores = np.zeros((len(l1_ratios), len(alphas)))
        r2_scores = np.zeros((len(l1_ratios), len(alphas)))

        for l,l1_ratio in enumerate(l1_ratios):
            for a,alpha in enumerate(alphas):
                # Create and fit the Ridge model
                reg_elasticnet = linear_model.ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
                reg_elasticnet.fit(x_train, y_train[:,i]) 

                # Predict on the validation set
                y_pred = reg_elasticnet.predict(x_valid)

                # Calculate MSE and R-squared
                mse = mean_squared_error(y_valid[:,i], y_pred)
                r2 = r2_score(y_valid[:,i], y_pred)

                # Append scores to the lists
                mse_scores[l,a] = mse
                r2_scores[l,a] = r2

                #print(f"Alpha = {alpha:.5f}, l1_ratio = {l1_ratio:.5f} : MSE = {mse:.3f}, R-squared = {r2:.3f}")

        # for l1_ratio, alpha, mse, r2 in zip(l1_ratios, alphas, mse_scores, r2_scores):
        #     print(f"Alpha = {alpha:.5f}, l1_ratio = {l1_ratio:.5f} : MSE = {mse:.3f}, R-squared = {r2:.3f}")
        
        # Fitting the model with the alpha parameter producing the highest r2_score on the validation set
        l1_ratio_max_index, alpha_max_index = np.unravel_index(np.argmax(r2_scores), r2_scores.shape)
        print(f'Alpha = {alphas[alpha_max_index]:.3f}, l1_ratio = {l1_ratios[l1_ratio_max_index]:.3f}')
        reg_elasticnet = linear_model.ElasticNet(alpha=alphas[alpha_max_index], l1_ratio=l1_ratios[l1_ratio_max_index])
        reg_elasticnet.fit(x_train, y_train[:,i]) 

        # Predict on the train+validation and test sets
        y_pred_train = reg_elasticnet.predict(x_train)
        y_pred_valid = reg_elasticnet.predict(x_valid)
        y_pred_test = reg_elasticnet.predict(x_test)

        print(
            f'Elastic Lasso regression ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}')

        print(
            f'Elastic Lasso regression ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}')

        print(
            f'Elastic Lasso regression ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}')

def AdaBoost (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test, FRED_MD):
    # ADABoost
    print('-=AdaBoosting=-')
    for i, y_i in enumerate(y_types):

        # Create an array to specify which samples are in the training set (0) and which are in the validation set (1)
        train_valid_split = [0] * len(x_train) + [1] * len(x_valid)

        # Create PredefinedSplit object to use custom validation set
        custom_cv = PredefinedSplit(test_fold=train_valid_split)

        # Define the parameter grid for grid search
        param_grid = {
        'n_estimators': [25, 50, 100, 200, 300, 500, 700], 
        'learning_rate': [0.01, 0.1, 1.0, 10.0]} # Default: 1.0

        # Create the AdaBoosting Regressor model
        base_model = AdaBoostRegressor(
            random_state = SEED_ID, 
            loss = 'square')

        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator = base_model, 
            param_grid = param_grid, 
            n_jobs = -1, 
            verbose = 1, 
            scoring = 'r2', 
            cv = custom_cv)

        # Fit the grid search to your training data
        grid_search.fit(x_train_valid, y_train_valid[:, i])

        # Get the best parameters
        best_n_estimators = grid_search.best_params_['n_estimators']
        best_learning_rate = grid_search.best_params_['learning_rate']

        # Create the final model with the best parameters
        final_model = AdaBoostRegressor(
            n_estimators = best_n_estimators, 
            learning_rate = best_learning_rate, 
            random_state = SEED_ID, 
            loss = 'square')

        # Fit the final model to the training data
        final_model.fit(x_train, y_train[:, i])

        # Predict on the train+validation and test sets
        y_pred_train = final_model.predict(x_train)
        y_pred_valid = final_model.predict(x_valid)
        y_pred_test = final_model.predict(x_test)

        print(
            f'AdaBoost ({y_i}): '
            f'best n_estimators: {best_n_estimators} and '
            f'best learning_rate:{best_learning_rate}')
        print(
            f'AdaBoost ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}')
        print(
            f'AdaBoost ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}') 
        print(
            f'AdaBoost ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}') 

        # Write data in the file
        if i==0:
            with open(f'logs/AdaBoost_best_params(FRED_MD={FRED_MD}).txt', 'w') as file:
                pass
        with open(f'logs/AdaBoost_best_params(FRED_MD={FRED_MD}).txt', 'a') as file:
            file.write(
                f'AdaBoost ({y_i}): best n_estimators = {best_n_estimators} '
                f'and best learning_rate = {best_learning_rate}\n')
            file.write(
                f'AdaBoost ({y_i}) for train set: '
                f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
                f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}\n')
            file.write(
                f'AdaBoost ({y_i}) for valid set: '
                f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
                f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}\n') 
            file.write(
                f'AdaBoost ({y_i}) for test set: '
                f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
                f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}\n')


def GradientBoost (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test, FRED_MD = True):
    # GradientBoost
    print('-=Gradient boosting=-')
    for i, y_i in enumerate(y_types):
        # Create an array to specify which samples are in the training set (0) and which are in the validation set (1)
        train_valid_split = [0] * len(x_train) + [1] * len(x_valid)

        # Create PredefinedSplit object to use custom validation set
        custom_cv = PredefinedSplit(test_fold=train_valid_split)

        # Define the parameter grid for grid search
        param_grid = {
        'n_estimators': [10, 25, 50, 100, 200],  #[10, 25, 50, 100, 200, 300, 500]
        'learning_rate': [0.2], # Default: 0.1,  [0.01, 0.1, 0.2]
        'max_depth': [2, 3, 4, 5, 6, 7, 8],# [2, 3, 4, 5, 6, 7, 8, 9, 10]
        'max_features' : [1.0] # Default: 1.0
        } 

        # Create the Gradient Boosting Regressor model
        base_model = GradientBoostingRegressor(
            random_state = SEED_ID,
            loss='squared_error')

        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator = base_model, 
            param_grid = param_grid, 
            n_jobs = -1, 
            verbose = 1, 
            scoring = 'r2', 
            cv = custom_cv)

        # Fit the grid search to your training data
        grid_search.fit(x_train_valid, y_train_valid[:, i])

        # Get the best parameters
        best_n_estimators = grid_search.best_params_['n_estimators']
        best_learning_rate = grid_search.best_params_['learning_rate']
        best_max_depth = grid_search.best_params_['max_depth']
        best_max_features = grid_search.best_params_['max_features']

        # Create the final model with the best parameters
        final_model = GradientBoostingRegressor(
            n_estimators=best_n_estimators, 
            learning_rate=best_learning_rate, 
            max_depth=best_max_depth,
            max_features=best_max_features,
            random_state = SEED_ID,
            loss='squared_error')

        # Fit the final model to the training data
        final_model.fit(x_train, y_train[:, i])

        # Predict on the train+validation and test sets
        y_pred_train = final_model.predict(x_train)
        y_pred_valid = final_model.predict(x_valid)
        y_pred_test = final_model.predict(x_test)

        print(
            f'Gboost ({y_i}): '
            f'best n_estimators = {best_n_estimators}, '
            f'best learning_rate = {best_learning_rate}, '
            f'best max_depth = {best_max_depth} and '
            f'best max_features = {best_max_features}')

        print(
            f'GradientBoost ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}') 
        print(
            f'GradientBoost ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}')
        print(
            f'GradientBoost ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}')

        # Write data in the file
        if i==0:
            with open(f'logs/Gboost_best_params(FRED_MD ={FRED_MD}).txt', 'w') as file:
                pass
        with open(f'logs/Gboost_best_params(FRED_MD ={FRED_MD}).txt', 'a') as file:
            file.write(
                f'Gboost ({y_i}): '
                f'best n_estimators = {best_n_estimators}, '
                f'best learning_rate = {best_learning_rate}, '
                f'best max_depth = {best_max_depth} and '
                f'best max_features = {best_max_features}\n')
            file.write(
                f'GBoost ({y_i}) for train set: '
                f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
                f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}\n')
            file.write(
                f'GBoost ({y_i}) for valid set: '
                f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
                f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}\n') 
            file.write(
                f'GBoost ({y_i}) for test set: '
                f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
                f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}\n')

 
def GradientBoost_plot(x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test):
    #Boosting convergence graph
    y_i = 0 
    params = {
    "n_estimators" : 50, 
    "learning_rate" : 0.2, 
    "max_depth" : 3,
    #"max_features" : 0.8,
    "random_state" : SEED_ID, 
    'verbose' : 1, 
    "loss" : 'squared_error'}

    Gboost = GradientBoostingRegressor(**params)
    Gboost.fit(x_train, y_train[:, y_i])

    # Predict on the train+validation and test sets
    y_pred_train = Gboost.predict(x_train)
    y_pred_test = Gboost.predict(x_test)
    print(
        f'Gboost ({y_types[y_i]}) for train set: '
        f'MSE = {mean_squared_error(y_train[:, y_i], y_pred_train):.3f} and '
        f'R-squared = {r2_score(y_train[:, y_i], y_pred_train):.3f}') 
    print(
        f'Gboost ({y_types[y_i]}) for test set: '
        f'MSE = {mean_squared_error(y_test[:, y_i], y_pred_test):.3f} and '
        f'R-squared = {r2_score(y_test[:, y_i], y_pred_test):.3f}')  
        
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred_test in enumerate(Gboost.staged_predict(x_test)):
        test_score[i] = mean_squared_error(y_test[:, y_i], y_pred_test)

    
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    #plt.title("Deviance")
    plt.plot(np.arange(params["n_estimators"]) + 1, Gboost.train_score_, "b-", label="Training Set Deviance")  
    plt.plot(np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance")
    # Formatting 
    font_size = 14
    plt.legend(loc="upper right", fontsize=font_size)
    plt.xlabel("Boosting Iterations",fontsize=font_size)
    plt.ylabel("Deviance",fontsize=font_size)
    plt.xticks(fontsize=font_size)  
    plt.yticks(fontsize=font_size)
    plt.grid(True)
    fig.tight_layout()
    plt.show()


def XGBoost_reg (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test, FRED_MD):
    # XGBoost
    print('-=XGBoost=-')
    for i, y_i in enumerate(y_types):      
        #i = 2 
        #y_i = '10_YR_R'

        # Create an array to specify which samples are in the training set (0) and which are in the validation set (1)
        train_valid_split = [0] * len(x_train) + [1] * len(x_valid)

        # Create PredefinedSplit object to use custom validation set
        custom_cv = PredefinedSplit(test_fold=train_valid_split)

        # Define the parameter grid for grid search
        param_grid = {
        'n_estimators': [10, 25, 50, 100, 200, 300], 
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 
        'eta' : [0.30],  #[default=0.3] [0.01, 0.10, 0.20, 0.30, 0.40] Step size shrinkage used in update to prevents overfitting. 
        'reg_lambda' : [1.0, 1.5, 2.0, 10, 100, 1000] #[default=1.0] L2 regularization term on weights
        } 

        # Create the XGradient Boosting Regressor model
        reg_alpha = 0.0 #[default=0.0] L1 regularization term on weights
        subsample = 1.0 #[default=1.0] Subsample ratio of the training instances.Subsampling will occur once in every boosting iteration.
        colsample_bytree = 1.0 #[default=1.0]  Subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.

        base_model = xgb.XGBRegressor(
            nthread = -1, 
            seed = SEED_ID,
            reg_alpha=reg_alpha,
            subsample=subsample, 
            colsample_bytree=colsample_bytree)

        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=base_model, 
            param_grid=param_grid, 
            n_jobs=-1, 
            scoring = 'r2', 
            cv=custom_cv)

        # Fit the grid search to your training data
        grid_search.fit(x_train_valid, y_train_valid[:, i])

        # Get the best parameters
        best_n_estimators = grid_search.best_params_['n_estimators']
        best_max_depth = grid_search.best_params_['max_depth']
        best_eta  = grid_search.best_params_['eta']
        best_lambda = grid_search.best_params_['reg_lambda']

        # Create the final model with the best parameters
        final_model = xgb.XGBRegressor(
            nthread = -1,
            seed = SEED_ID,
            n_estimators=best_n_estimators, 
            max_depth=best_max_depth, 
            eta=best_eta, 
            reg_lambda=best_lambda, 
            reg_alpha=reg_alpha,
            subsample=subsample, 
            colsample_bytree=colsample_bytree)

        # Fit the final model to the training data
        final_model.fit(x_train, y_train[:, i])

        # Predict on the train+validation and test sets
        y_pred_train = final_model.predict(x_train)
        y_pred_valid = final_model.predict(x_valid)
        y_pred_test = final_model.predict(x_test)

        print(
            f'XGBoost ({y_i}): '
            f'best n_estimators = {best_n_estimators}, '
            f'best max_depth = {best_max_depth}, '
            f'best lambda = {best_lambda}')
        print(
            f'XGBoost ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}')
        print(
            f'XGBoost ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}') 
        print(
            f'XGBoost ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}') 

        # Write data in the file
        if i==0:
            with open(f'logs/XGBoost_best_params(FRED_MD={FRED_MD}).txt', 'w') as file:
                pass

        with open(f'logs/XGBoost_best_params(FRED_MD={FRED_MD}).txt', 'a') as file:
            file.write(
                f'XGBoost ({y_i}): '
                f'best n_estimators = {best_n_estimators}, '
                f'best max_depth = {best_max_depth}, '
                f'best lambda = {best_lambda}\n')
            file.write(
                f'XGBoost ({y_i}) for train set: '
                f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
                f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}\n')
            file.write(
                f'XGBoost ({y_i}) for valid set: '
                f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
                f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}\n') 
            file.write(
                f'XGBoost ({y_i}) for test set: '
                f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
                f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}\n')

        
        # # Non-optimized on the validation set model with default parameters 
        # xgb_reg_simple = xgb.XGBRegressor( )
        # xgb_reg_simple.fit(x_train, y_train[:,i])

        # y_pred_train = xgb_reg_simple.predict(x_train)
        # y_pred_valid = xgb_reg_simple.predict(x_valid)
        # y_pred_test = xgb_reg_simple.predict(x_test)

        # print(
        #     f'XGBoost_simple ({y_i}) for train set: '
        #     f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
        #     f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}')
        # print(
        #     f'XGBoost_simple ({y_i}) for valid set: '
        #     f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
        #     f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}') 
        # print(
        #     f'XGBoost_simple ({y_i}) for test set: '
        #     f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
        #     f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}') 

        #break

def XGBoost_reg_plot(x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test):
    #XGBoosting convergence graph
    y_i = 0 # 0: 5_YR_R, 1: 7_YR_R, 2: 10_YR_R
    params = {
    'nthread' : 4, 
    'n_estimators': 50, 
    'max_depth': 2, 
    'eta': 0.30, 
    'reg_lambda': 10,
    'subsample': 1.0, 
    'colsample_bytree': 1.0, 
    'reg_alpha': 0.0
    }
    XGBoost = xgb.XGBRegressor(**params)
    XGBoost.fit(x_train, y_train[:, y_i])

    # Predict on the train+validation and test sets
    y_pred_train = XGBoost.predict(x_train)
    y_pred_test = XGBoost.predict(x_test)
    print(
        f'XGBoost ({y_types[y_i]}) for train set: '
        f'MSE = {mean_squared_error(y_train[:, y_i], y_pred_train):.3f} and '
        f'R-squared = {r2_score(y_train[:, y_i], y_pred_train):.3f}') 

    print(
        f'XGBoost ({y_types[y_i]}) for test set: '
        f'MSE = {mean_squared_error(y_test[:, y_i], y_pred_test):.3f} and '
        f'R-squared = {r2_score(y_test[:, y_i], y_pred_test):.3f}')  
    
    n_estimators = params['n_estimators']
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    train_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i in range(1, n_estimators+1):
        params["n_estimators"] = i
        XGBoost = xgb.XGBRegressor(**params)
        XGBoost.fit(x_train, y_train[:, y_i])
        y_pred_test = XGBoost.predict(x_test)
        train_score[i-1] = mean_squared_error(y_train[:, y_i], XGBoost.predict(x_train))  
        test_score[i-1] = mean_squared_error(y_test[:, y_i], y_pred_test)    

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    #plt.title("Deviance")
    plt.plot(np.arange(params["n_estimators"]) + 1, train_score, "b-", label="Training Set Deviance")  
    plt.plot(np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance")
    # Formatting 
    font_size = 14
    plt.legend(loc="upper right", fontsize=font_size)
    plt.xlabel("Boosting Iterations",fontsize=font_size)
    plt.ylabel("Deviance",fontsize=font_size)
    plt.xticks(fontsize=font_size)  
    plt.yticks(fontsize=font_size)
    plt.grid(True)
    fig.tight_layout()
    plt.show()

def RandomForest (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test, FRED_MD):
    # RandomForest
    print('-=RandomForest=-')
    for i, y_i in enumerate(y_types):

        # Create an array to specify which samples are in the training set (0) and which are in the validation set (1)
        train_valid_split = [0] * len(x_train) + [1] * len(x_valid)
        #print(train_valid_split)

        # Create PredefinedSplit object to use custom validation set
        custom_cv = PredefinedSplit(test_fold=train_valid_split)

        # Define the parameter grid for grid search
        param_grid = {
        'n_estimators': [10, 25, 50, 100, 200, 300],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], #[2, 3, 4, 5, 6, 7, 8, 9, 10]
        'max_features' : [1.0], # grid [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] does not provide an improvement
        'max_samples': [1.0]} # grid [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] does not provide an improvement

        # Create the RandomForest Regressor model
        base_model = RandomForestRegressor(
            n_jobs=-1, 
            bootstrap=True,
            random_state = SEED_ID,
            criterion='squared_error')

        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=base_model, 
            param_grid=param_grid, 
            n_jobs=-1, 
            verbose = 1,  # 3 if more details needed
            scoring = 'r2', 
            cv=custom_cv)

        # Fit the grid search to your training data
        grid_search.fit(x_train_valid, y_train_valid[:, i])

        # Get the best parameters
        best_n_estimators = grid_search.best_params_['n_estimators']
        best_max_depth = grid_search.best_params_['max_depth']
        best_max_features = grid_search.best_params_['max_features']
        best_max_samples = grid_search.best_params_['max_samples']

        # Create the final model with the best parameters
        final_model = RandomForestRegressor(
            n_estimators=best_n_estimators, 
            max_depth=best_max_depth,  
            n_jobs=-1, 
            bootstrap=True,
            random_state = SEED_ID,
            max_features = best_max_features,
            max_samples = best_max_samples,
            criterion='squared_error')

        # Fit the final model to the training data
        final_model.fit(x_train, y_train[:, i])

        # Predict on the train+validation and test sets
        y_pred_train = final_model.predict(x_train)
        y_pred_valid = final_model.predict(x_valid)
        y_pred_test = final_model.predict(x_test)

        print(
            f'RandomForest ({y_i}): '
            f'best n_estimators = {best_n_estimators}, '
            f'best max_depth = {best_max_depth}, '
            f'best max_features = {best_max_features} and '
            f'best max_samples = {best_max_samples}')
        print(
            f'RandomForest ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}')
        print(
            f'RandomForest ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}') 
        print(
            f'RandomForest ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}') 

        # Write data in the file
        if i==0:
            with open(f'logs/RandomForest_best_params(FRED_MD={FRED_MD}).txt', 'w') as file:
                pass

        with open(f'logs/RandomForest_best_params(FRED_MD={FRED_MD}).txt', 'a') as file:
            file.write(
                f'RandomForest ({y_i}): '
                f'best n_estimators = {best_n_estimators}, '
                f'best max_depth = {best_max_depth}, '
                f'best max_features = {best_max_features} and '
                f'best max_samples = {best_max_samples}\n')
            file.write(
                f'RandomForest ({y_i}) for train set: '
                f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
                f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}\n')
            file.write(
                f'RandomForest ({y_i}) for valid set: '
                f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
                f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}\n') 
            file.write(
                f'RandomForest ({y_i}) for test set: '
                f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
                f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}\n')
        
        #break

def ExtraTrees (x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test, FRED_MD):
    # Extra Trees Regressor
    print('-=Extra Trees Regressor=-')
    for i, y_i in enumerate(y_types):

        #i = 2
        #y_i = y_types[i]

        # Create an array to specify which samples are in the training set (0) and which are in the validation set (1)
        train_valid_split = [0] * len(x_train) + [1] * len(x_valid)
        #print(train_valid_split)

        # Create PredefinedSplit object to use custom validation set
        custom_cv = PredefinedSplit(test_fold=train_valid_split)

        # Define the parameter grid for grid search
        param_grid = {
        'n_estimators': [10, 25, 50, 100, 200, 300, 500],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'max_features' : [1.0], # [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        'max_samples': [1.0]} # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Create the Extra Trees Regressor model 
        base_model = ExtraTreesRegressor(
            n_jobs=-1, 
            bootstrap=True,
            random_state = SEED_ID,
            criterion='squared_error')

        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=base_model, 
            param_grid=param_grid, 
            n_jobs=-1, 
            verbose = 1, # 3 if more details needed
            scoring = 'r2', 
            cv=custom_cv)

        # Fit the grid search to your training data
        grid_search.fit(x_train_valid, y_train_valid[:, i])

        # Get the best parameters
        best_n_estimators = grid_search.best_params_['n_estimators']
        best_max_depth = grid_search.best_params_['max_depth']
        best_max_features = grid_search.best_params_['max_features']
        best_max_samples = grid_search.best_params_['max_samples']

        # Create the final model with the best parameters
        final_model = ExtraTreesRegressor(
            n_estimators=best_n_estimators, 
            max_depth=best_max_depth,
            max_features = best_max_features,
            max_samples = best_max_samples,
            n_jobs=-1, 
            bootstrap=True,
            random_state = SEED_ID,
            criterion='squared_error')

        # Fit the final model to the training data
        final_model.fit(x_train, y_train[:, i])

        # Predict on the train+validation and test sets
        y_pred_train = final_model.predict(x_train)
        y_pred_valid = final_model.predict(x_valid)
        y_pred_test = final_model.predict(x_test)

        print(
            f'ExtraTrees ({y_i}): '
            f'best n_estimators = {best_n_estimators}, '
            f'best max_depth = {best_max_depth}, '
            f'best max_features = {best_max_features} and '
            f'best max_samples = {best_max_samples}')
        print(
            f'ExtraTrees ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}')
        print(
            f'ExtraTrees ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}') 
        print(
            f'ExtraTrees ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}') 

        # Write data in the file
        if i==0:
            with open(f'logs/ExtraTrees_best_params(FRED_MD={FRED_MD}).txt', 'w') as file:
                pass

        with open(f'logs/ExtraTrees_best_params(FRED_MD={FRED_MD}).txt', 'a') as file:
            file.write(
                f'ExtraTrees ({y_i}): '
                f'best n_estimators = {best_n_estimators}, '
                f'best max_depth = {best_max_depth}, '
                f'best max_features = {best_max_features} and '
                f'best max_samples = {best_max_samples}\n')
            file.write(
                f'ExtraTrees ({y_i}) for train set: '
                f'MSE = {mean_squared_error(y_train[:,i], y_pred_train):.3f} and '
                f'R-squared = {r2_score(y_train[:,i], y_pred_train):.3f}\n')
            file.write(
                f'ExtraTrees ({y_i}) for valid set: '
                f'MSE = {mean_squared_error(y_valid[:,i], y_pred_valid):.3f} and '
                f'R-squared = {r2_score(y_valid[:,i], y_pred_valid):.3f}\n') 
            file.write(
                f'ExtraTrees ({y_i}) for test set: '
                f'MSE = {mean_squared_error(y_test[:,i], y_pred_test):.3f} and '
                f'R-squared = {r2_score(y_test[:,i], y_pred_test):.3f}\n') 
            
        #break

def Dense_NN_1layer(x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test, FRED_MD):
    print('-=Dense NN (1 layer)=-')

    for i, y_i in enumerate(y_types):
    
        layer1_dims = [5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700]  # Example dimensions for the first layer
        best_val_loss = np.inf
        best_config = None

        # Early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0,
            restore_best_weights=True
            )

        for layer1_dim in layer1_dims:
            print(f"Training model with layer1_dim: {layer1_dim} ")
            model = keras.Sequential([
                layers.Dense(layer1_dim, activation='relu', input_shape=(x_train.shape[1],)),
                layers.Dense(1)
                ])

            # Compile model
            model.compile(optimizer='adam', loss='mse')
            #model.summary()

            # Fit the model
            model_fitted = model.fit(
                x_train,y_train[:, i], 
                validation_data=(x_valid, y_valid[:, i]), 
                epochs = 1000, 
                verbose = 0,
                callbacks=[early_stopping_callback],
                )

            # Check if this model has the best validation loss
            val_loss = min(model_fitted.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = (layer1_dim)
        

        # Re-train the best model
        model = keras.Sequential([
                    layers.Dense(best_config, activation='relu', input_shape=(x_train.shape[1],)),
                    layers.Dense(1)
                    ])

        model.compile(optimizer='adam', loss='mse')

        model_fitted = model.fit(
                    x_train,y_train[:, i], 
                    validation_data=(x_valid, y_valid[:, i]), 
                    epochs=1000, 
                    verbose = 0,
                    callbacks=[early_stopping_callback],
                    )

        # Predict on test data
        y_pred_train = model.predict(x_train, verbose = 0)
        y_pred_valid = model.predict(x_valid, verbose = 0)
        y_pred_test = model.predict(x_test, verbose = 0)

        print(
            f'Best configuration: '
            f'layer1_dim={best_config}')
        print(
            f'Dense NN ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:, i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:, i], y_pred_train):.3f}')
        print(
            f'Dense NN ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:, i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:, i], y_pred_valid):.3f}') 
        print(
            f'Dense NN ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:, i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:, i], y_pred_test):.3f}') 

        # Write data in the file
        if i==0:
            with open(f'logs/Dense_NN_1L_best_params(FRED_MD={FRED_MD}).txt', 'w') as file:
                pass

        with open(f'logs/Dense_NN_1L_best_params(FRED_MD={FRED_MD}).txt', 'a') as file:
            file.write(
                f'Dense NN with 1 layer ({y_i}): '
                f'layer1_dim={best_config}\n')
            file.write(
                f'Dense NN ({y_i}) for train set: '
                f'MSE = {mean_squared_error(y_train[:, i], y_pred_train):.3f} and '
                f'R-squared = {r2_score(y_train[:, i], y_pred_train):.3f}\n')
            file.write(
                f'Dense NN ({y_i}) for valid set: '
                f'MSE = {mean_squared_error(y_valid[:, i], y_pred_valid):.3f} and '
                f'R-squared = {r2_score(y_valid[:, i], y_pred_valid):.3f}\n') 
            file.write(
                f'Dense NN ({y_i}) for test set: '
                f'MSE = {mean_squared_error(y_test[:, i], y_pred_test):.3f} and '
                f'R-squared = {r2_score(y_test[:, i], y_pred_test):.3f}\n') 

        # Plotting
        training_loss = model_fitted.history['loss']
        validation_loss = model_fitted.history['val_loss']
        epochs = range(1, len(training_loss) + 1)

        plt.figure()
        plt.plot(epochs, training_loss, 'bo-', label='Training Loss')
        plt.plot(epochs, validation_loss, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('graphs/' + y_i + f'_NN_1L(FRED_MD={FRED_MD}).png')


def Dense_NN_2layers(x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test, FRED_MD):
    print('-=Dense NN (2 layers) =-')
    for i, y_i in enumerate(y_types):

        layer1_dims = [25, 50, 100, 200, 300, 400, 500]  # Example dimensions for the first layer
        layer2_dims = [5, 25, 50, 100, 150, 200, 300]    # Example dimensions for the second layer
     
        best_val_loss = np.inf
        best_config = None

        # Early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0,
            restore_best_weights=True
            )

        for layer1_dim in layer1_dims:
            for layer2_dim in layer2_dims:
                print(f"Training model with layer1_dim: {layer1_dim}, layer2_dim: {layer2_dim} ")
                model = keras.Sequential([
                    layers.Dense(layer1_dim, activation='relu', input_shape=(x_train.shape[1],)),
                    layers.Dense(layer2_dim, activation='relu'),
                    layers.Dense(1)
                    ])

                # Compile model
                model.compile(optimizer='adam', loss='mse')
                #model.summary()

                # Fit the model
                model_fitted = model.fit(
                    x_train,y_train[:, i], 
                    validation_data=(x_valid, y_valid[:, i]), 
                    epochs=1000, 
                    verbose = 0,
                    callbacks=[early_stopping_callback],
                    )

                # Check if this model has the best validation loss
                val_loss = min(model_fitted.history['val_loss'])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = (layer1_dim, layer2_dim)
            

        # Re-train the best model
        model = keras.Sequential([
                    layers.Dense(best_config[0], activation='relu', input_shape=(x_train.shape[1],)),
                    layers.Dense(best_config[1], activation='relu'),
                    layers.Dense(1)
                    ])

        model.compile(optimizer='adam', loss='mse')

        model_fitted = model.fit(
                    x_train,y_train[:, i], 
                    validation_data=(x_valid, y_valid[:, i]), 
                    epochs=1000, 
                    verbose = 0,
                    callbacks=[early_stopping_callback],
                    )

        # Predict on test data
        y_pred_train = model.predict(x_train, verbose = 0)
        y_pred_valid = model.predict(x_valid, verbose = 0)
        y_pred_test = model.predict(x_test, verbose = 0)

        print(
            f'Best configuration: '
            f'layer1_dim={best_config[0]}, '
            f'layer2_dim={best_config[1]}')
        print(
            f'Dense NN ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:, i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:, i], y_pred_train):.3f}')
        print(
            f'Dense NN ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:, i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:, i], y_pred_valid):.3f}') 
        print(
            f'Dense NN ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:, i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:, i], y_pred_test):.3f}') 

        # Write data in the file
        if i==0:
            with open(f'logs/Dense_NN_2L_best_params(FRED_MD={FRED_MD}).txt', 'w') as file:
                pass

        with open(f'logs/Dense_NN_2L_best_params(FRED_MD={FRED_MD}).txt', 'a') as file:
            file.write(
                f'Dense NN with 2 layers ({y_i}): '
                f'layer1_dim={best_config[0]}, '
                f'layer2_dim={best_config[1]}\n')
            file.write(
                f'Dense NN ({y_i}) for train set: '
                f'MSE = {mean_squared_error(y_train[:, i], y_pred_train):.3f} and '
                f'R-squared = {r2_score(y_train[:, i], y_pred_train):.3f}\n')
            file.write(
                f'Dense NN ({y_i}) for valid set: '
                f'MSE = {mean_squared_error(y_valid[:, i], y_pred_valid):.3f} and '
                f'R-squared = {r2_score(y_valid[:, i], y_pred_valid):.3f}\n') 
            file.write(
                f'Dense NN ({y_i}) for test set: '
                f'MSE = {mean_squared_error(y_test[:, i], y_pred_test):.3f} and '
                f'R-squared = {r2_score(y_test[:, i], y_pred_test):.3f}\n') 

        # Plotting
        training_loss = model_fitted.history['loss']
        validation_loss = model_fitted.history['val_loss']
        epochs = range(1, len(training_loss) + 1)

        plt.figure()
        plt.plot(epochs, training_loss, 'bo-', label='Training Loss')
        plt.plot(epochs, validation_loss, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('graphs/' + y_i + f'_NN_2L(FRED_MD={FRED_MD}).png')


def Dense_NN_3layers(x_train, x_valid, x_train_valid, x_test, y_train, y_valid, y_train_valid, y_test, FRED_MD):
    print('-=Dense NN (3 layers)=-')
    for i, y_i in enumerate(y_types):

        layer1_dims = [50, 100, 200, 300, 400, 500]  # Example dimensions for the first layer
        layer2_dims = [25, 50, 100, 150, 200, 300]    # Example dimensions for the second layer
        layer3_dims = [5, 10, 25, 50, 100]    # Example dimensions for the third layer

        best_val_loss = np.inf
        best_config = None

        # Early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0,
            restore_best_weights=True
            )

        for layer1_dim in layer1_dims:
            for layer2_dim in layer2_dims:
                for layer3_dim in layer3_dims:
                    print(f"Training model with layer1_dim: {layer1_dim}, layer2_dim: {layer2_dim}, layer3_dim: {layer3_dim}")

                    model = keras.Sequential([
                        layers.Dense(layer1_dim, activation='relu', input_shape=(x_train.shape[1],)),
                        layers.Dense(layer2_dim, activation='relu'),
                        layers.Dense(layer3_dim, activation='relu'),
                        layers.Dense(1)
                        ])

                    # Compile model
                    model.compile(optimizer='adam', loss='mse')
                    #model.summary()

                    # Fit the model
                    model_fitted = model.fit(
                        x_train,y_train[:, i], 
                        validation_data=(x_valid, y_valid[:, i]), 
                        epochs=1000, 
                        verbose=0,
                        callbacks=[early_stopping_callback],
                        )

                    # Check if this model has the best validation loss
                    val_loss = min(model_fitted.history['val_loss'])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_config = (layer1_dim, layer2_dim, layer3_dim)
            

        # Re-train the best model
        model = keras.Sequential([
                    layers.Dense(best_config[0], activation='relu', input_shape=(x_train.shape[1],)),
                    layers.Dense(best_config[1], activation='relu'),
                    layers.Dense(best_config[2], activation='relu'),
                    layers.Dense(1)
                    ])

        model.compile(optimizer='adam', loss='mse')

        model_fitted = model.fit(
                    x_train,y_train[:, i], 
                    validation_data=(x_valid, y_valid[:, i]), 
                    epochs=1000, 
                    verbose=0,
                    callbacks=[early_stopping_callback],
                    )

        # Predict on test data
        y_pred_train = model.predict(x_train, verbose = 0)
        y_pred_valid = model.predict(x_valid, verbose = 0)
        y_pred_test = model.predict(x_test, verbose = 0)

        print(
            f'Best configuration: '
            f'layer1_dim={best_config[0]}, '
            f'layer2_dim={best_config[1]}, '
            f'layer3_dim={best_config[2]}')
        print(
            f'Dense NN ({y_i}) for train set: '
            f'MSE = {mean_squared_error(y_train[:, i], y_pred_train):.3f} and '
            f'R-squared = {r2_score(y_train[:, i], y_pred_train):.3f}')
        print(
            f'Dense NN ({y_i}) for valid set: '
            f'MSE = {mean_squared_error(y_valid[:, i], y_pred_valid):.3f} and '
            f'R-squared = {r2_score(y_valid[:, i], y_pred_valid):.3f}') 
        print(
            f'Dense NN ({y_i}) for test set: '
            f'MSE = {mean_squared_error(y_test[:, i], y_pred_test):.3f} and '
            f'R-squared = {r2_score(y_test[:, i], y_pred_test):.3f}') 

        # Write data in the file
        if i==0:
            with open(f'logs/Dense_NN_3L_best_params(FRED_MD={FRED_MD}).txt', 'w') as file:
                pass

        with open(f'logs/Dense_NN_3L_best_params(FRED_MD={FRED_MD}).txt', 'a') as file:
            file.write(
                f'Dense NN with 3 layers ({y_i}): '
                f'layer1_dim={best_config[0]}, '
                f'layer2_dim={best_config[1]}, '
                f'layer3_dim={best_config[2]}\n')
            file.write(
                f'Dense NN ({y_i}) for train set: '
                f'MSE = {mean_squared_error(y_train[:, i], y_pred_train):.3f} and '
                f'R-squared = {r2_score(y_train[:, i], y_pred_train):.3f}\n')
            file.write(
                f'Dense NN ({y_i}) for valid set: '
                f'MSE = {mean_squared_error(y_valid[:, i], y_pred_valid):.3f} and '
                f'R-squared = {r2_score(y_valid[:, i], y_pred_valid):.3f}\n') 
            file.write(
                f'Dense NN ({y_i}) for test set: '
                f'MSE = {mean_squared_error(y_test[:, i], y_pred_test):.3f} and '
                f'R-squared = {r2_score(y_test[:, i], y_pred_test):.3f}\n') 

        # Plotting
        training_loss = model_fitted.history['loss']
        validation_loss = model_fitted.history['val_loss']
        epochs = range(1, len(training_loss) + 1)

        plt.figure()
        plt.plot(epochs, training_loss, 'bo-', label='Training Loss')
        plt.plot(epochs, validation_loss, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('graphs/' + y_i + f'_NN_3L(FRED_MD={FRED_MD}).png')