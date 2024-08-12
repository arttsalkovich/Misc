### util.py is a library for main.py containing auxiliary functions 
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


from datetime import datetime

# List of files to read from or to write to
bbg_market_data_path = 'data/bbg_market_data.csv'
bbg_moodys_corp_yield_path = 'data/bbg_moodys_corp_yield.csv'
bbg_CP3M_path = 'data/bbg_CP3M.csv'
real_yield_path = 'data/par_real_yield_curve_rates_2003_2023.csv'
UST_curve_path = 'data/yield-curve-rates-1990-2023.csv'
FRED_MD_path = 'data/FRED_MD.csv' # change to 'data/FRED_MD_ADJ.csv' if you want to use the data up to Sep
features_path = 'data/data_no_FRED_stationary_normalized_features.csv'
heatmap_file_path = 'graphs/correlation_matrix_heatmap.png'

def create_dataset(FRED_MD = True):
    ## WEEKLY BLOOMBERG DATA ###
    # Load market data (FX rates, commodity prices and equity market metrics) - weekly data from Bloomberg (19 in total)
    print('Processing Bloomberg data for FX, EQ and commodities (weekly values)')
    df_bbg_market_data = pd.read_csv(bbg_market_data_path, parse_dates=['Date'])
    df_bbg_market_data.set_index('Date', inplace=True)
    df_bbg_market_data.sort_index(inplace=True)
    df_bbg_data_weekly = df_bbg_market_data.resample('W').mean()
    missing_data_fill(df_bbg_data_weekly)
   
    # Load market data for Moody's Corporate Bond Yield Aaa and Baa - weekly data from Bloomberg (2 in total)
    print('Processing Bloomberg data for Moody\'s Corporate Bond Yield Aaa and Baa (weekly values)')
    df_bbg_moodys_corp_yield = pd.read_csv(bbg_moodys_corp_yield_path, parse_dates=['Date'])
    df_bbg_moodys_corp_yield.set_index('Date', inplace=True)
    df_bbg_moodys_corp_yield.sort_index(inplace=True)
    df_bbg_moodys_corp_yield_weekly = df_bbg_moodys_corp_yield.resample('W').mean()
    missing_data_fill(df_bbg_moodys_corp_yield_weekly)

    # Merging df_bbg_data_weekly and df_bbg_moodys_corp_yield_weekly (21 in total)
    merged_df = pd.merge(df_bbg_data_weekly, df_bbg_moodys_corp_yield_weekly, on='Date', how='inner')
    merged_df[merged_df.columns.difference(['Date'])] = merged_df[merged_df.columns.difference(['Date'])].apply(pd.to_numeric, errors='coerce')

    # Load market data for 90d financial commercial paper yield - weekly data from Bloomberg (1 in total)
    print('Processing Bloomberg data for 90d financial commercial paper yield (weekly values)')
    df_bbg_bbg_CP3M = pd.read_csv(bbg_CP3M_path, parse_dates=['Date'])
    df_bbg_bbg_CP3M.set_index('Date', inplace=True)
    df_bbg_bbg_CP3M.sort_index(inplace=True)
    df_bbg_bbg_CP3M_weekly = df_bbg_bbg_CP3M.resample('W').mean()
    missing_data_fill(df_bbg_bbg_CP3M_weekly)

    # Merging df_bbg_data_weekly, df_bbg_moodys_corp_yield_weekly and df_bbg_bbg_CP3M_weekly (22 in total)
    df_bbg_data = pd.merge(merged_df, df_bbg_bbg_CP3M_weekly, on='Date', how='inner')
    df_bbg_data[df_bbg_data.columns.difference(['Date'])] = df_bbg_data[df_bbg_data.columns.difference(['Date'])].apply(pd.to_numeric, errors='coerce')

    ### DAILY RATES DATA ####
    # Load TIPS real yield curves dataset (5, 7, 10, 20 and 30 years tenors) - daily data from FRED StLouis
    print('Processing TIPS real yield curves dataset (daily values)')
    df_real_yield = pd.read_csv(real_yield_path, parse_dates=['Date'])
    df_real_yield.set_index('Date', inplace=True)
    df_real_yield.sort_index(inplace=True)

    # Load UST yield curves dataset (1, 2, 3, 5, 7, 10, 20 and 30 years tenors) - daily data from FRED StLouis
    print('Processing UST yield curves dataset (daily values)')
    df_UST_curve = pd.read_csv(UST_curve_path, parse_dates=['Date'])
    df_UST_curve.set_index('Date', inplace=True)
    df_UST_curve.sort_index(inplace=True)  

    # Load FedFunds Effective Rate - daily data from FRED StLouis
    print('Processing FedFunds Effective Rate (daily values)')
    df_fedfunds = pd.read_csv('data/bbg_fedfunds.csv', parse_dates=['Date'])
    df_fedfunds.set_index('Date', inplace=True)
    df_fedfunds.sort_index(inplace=True)  

    # Merging TIPS real yield curves and UST yield curves datasets. The new dataframe includes weekly average values
    df_rates_merged = pd.merge(df_real_yield, df_UST_curve, on='Date', how='inner')
    df_rates_ff_merged = pd.merge(df_rates_merged, df_fedfunds, on='Date', how='inner')
    df_rates_ff_merged[df_rates_ff_merged.columns.difference(['Date'])] = df_rates_ff_merged[df_rates_ff_merged.columns.difference(['Date'])].apply(pd.to_numeric, errors='coerce')
    df_rates_ff_merged = df_rates_ff_merged.resample('W').mean()

    # # Merging rates and BBG data
    df_rates_bbg_data = pd.merge(df_rates_ff_merged, df_bbg_data, on='Date', how='inner')

    # Cleaning up the dataframe by deleting the columns with missing data
    columns_with_missing_data = df_rates_bbg_data.columns[df_rates_bbg_data.isna().any()].tolist() # ['20_YR_R', '30_YR_R', '2_MO', '4_MO', '30_YR'] dropped
    print(f'The columns with missing data (dropped): {columns_with_missing_data}')
    df_rates_bbg_data = df_rates_bbg_data.drop(columns=columns_with_missing_data) # 14 in total

    # Adding new columns with spreads between the rates and FedFudns Effective Rate
    df_rates_bbg_data['COMPAPFFx'] = df_rates_bbg_data['CP3M'] - df_rates_bbg_data['DFF']   # 3-Month Commercial Paper Minus FEDFUNDS
    df_rates_bbg_data['TB3SMFFM'] = df_rates_bbg_data['3_MO'] - df_rates_bbg_data['DFF']    # 3-Month Treasury C Minus FEDFUNDS
    df_rates_bbg_data['TB6SMFFM'] = df_rates_bbg_data['6_MO'] - df_rates_bbg_data['DFF']    # 6-Month Treasury C Minus FEDFUNDS
    df_rates_bbg_data['T1YFFM'] = df_rates_bbg_data['1_YR'] - df_rates_bbg_data['DFF']      # 1-Year Treasury C Minus FEDFUNDS
    df_rates_bbg_data['T5YFFM'] = df_rates_bbg_data['5_YR'] - df_rates_bbg_data['DFF']      # 5-Year Treasury C Minus FEDFUNDS
    df_rates_bbg_data['T10YFFM'] = df_rates_bbg_data['10_YR'] - df_rates_bbg_data['DFF']    # 10-Year Treasury C Minus FEDFUNDS
    df_rates_bbg_data['AAAFFM'] = df_rates_bbg_data['AAA'] - df_rates_bbg_data['DFF']       # Moody’s Aaa Corporate Bond Minus FEDFUNDS
    df_rates_bbg_data['BAAFFM'] = df_rates_bbg_data['BAA'] - df_rates_bbg_data['DFF']       # Moody’s Baa Corporate Bond Minus FEDFUNDS
          
    # Adjusting bbg_data to make it stationary
    print('Adjusting the dataset to make it stationary')
    df_rates_bbg_data_stationary = pd.DataFrame()
    headers_one_lag_diff = ['5_YR_R', '7_YR_R', '10_YR_R', '1_MO', '3_MO', '6_MO', '1_YR', '2_YR','3_YR', 
                            '5_YR', '7_YR', '10_YR', '20_YR', 'DFF', 'CP3M', 'AAA', 'BAA', 'VIX', 'SPX_DIV',
                            'COMPAPFFx','TB3SMFFM', 'TB6SMFFM','T1YFFM','T5YFFM','T10YFFM','AAAFFM','BAAFFM']
    df_rates_bbg_data_stationary[headers_one_lag_diff] = df_rates_bbg_data[headers_one_lag_diff].diff(periods=1)    # Simple lag for rates data

    headers_one_lag_log_diff = ['S1','W1','C1','XAUUSD','LA1','LN1','CL1','HG1','SPX','SPX_IND','SPX_PE',
                                'USDEUR','USDJPY','USDGBP','USDCHF','USDCAD','USDWBGD']
    df_rates_bbg_data_stationary[headers_one_lag_log_diff] = df_rates_bbg_data[headers_one_lag_log_diff].apply(np.log).diff(periods=1) # Log difference for commodities, EQ and FX
    
    df_rates_bbg_data_stationary = df_rates_bbg_data_stationary.iloc[1:] # deleting the first row with NaN values

    if FRED_MD == False:
        df_rates_bbg_data_stationary = df_rates_bbg_data_stationary.iloc[:-3] # !!! DATASET SPECIFIC, PLEASE DELETE IF NOT APPLICABLE !!!
        return df_rates_bbg_data_stationary # 44 in total (3 real yield + 11 nominal yields + 30 bbg market data)

    ### FRED MD DATA ####
    # Load the RED-MD macro dataset CSV file, skipping the first row as it contains the transformation codes - monthly data
    print('Loading FRED-MD dataset (monthly data)')
    df = pd.read_csv(FRED_MD_path, skiprows=[1]) 

    # Read the transformation codes separately
    with open(FRED_MD_path, 'r') as file:
        next(file)  # Skip the header row
        transform_line = next(file)  # Read the transformation codes line

    # Split the transformation codes line by comma and remove the 'Transform:' label
    tcode = np.array([int(x) for x in transform_line.split(',')[1:]])

    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    print('Stationary transformation for FRED-MD dataset')
    for i, col in enumerate(df.columns):
        print(f'Column: {col}, tcode: {tcode[i]}')
        df[col] = FRED_MD_make_stationary(df[col], tcode[i])
        
    df_FRED_MD = pd.DataFrame()
    df_FRED_MD = df.iloc[2:]

    # Dropping the columns which will be overriden with Bloomberg weekly data 
    columns_to_drop = ['FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA', 'COMPAPFFx',
                        'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 'AAAFFM', 'BAAFFM', 'TWEXAFEGSMTHx',
                        'EXSZUSx', 'EXJPUSx', 'EXUSUKx', 'EXCAUSx', 'OILPRICEx', 'S&P 500', 'S&P: indust', 
                        'S&P div yield', 'S&P PE ratio', 'VIXCLSx']  # 28 in total, 99 features left in FRED-MD (127 originally)
    print(f'The following columns are being dropped from FRED-MD dataset \n {columns_to_drop}')
    df_FRED_MD = df_FRED_MD.drop(columns=columns_to_drop)

    # Resampling FRED-MD dataset from monthly observations to weekly, missed data is added using linear interpolation
    print('Resampling FRED-MD macro dataset to weekly data (linear interpolation)')
    df_FRED_MD_weekly = df_FRED_MD.resample('W').first()
    df_FRED_MD_weekly_interpolated = df_FRED_MD_weekly.interpolate(method='linear')

    #df_FRED_MD_weekly_interpolated.to_csv('data/FRED_MD_weekly_interpolated.csv', index=True)

    # Mergining FRED-MD data with weekly yields curves 
    final_merge = pd.merge(df_rates_bbg_data_stationary, df_FRED_MD_weekly_interpolated, on='Date', how='inner')
 
    # Cleaning up the dataframe by deleting the columns with missing data.
    columns_with_missing_data = final_merge.columns[final_merge.isna().any()].tolist()
    if columns_with_missing_data != []:
        print(f'Dropping the columns with missing data: {columns_with_missing_data}')
        final_merge = final_merge.drop(columns=columns_with_missing_data)
        
    print(final_merge)

    return final_merge


def missing_data_fill (df):
    # The function checks on whether there is missing data in the dataframe df and fills it with linearly interpolated approximation
    nan_exists = df.isna().any().any()
    print(f"Are there NaN values in the data? {nan_exists}")
    nan_counts = df.isna().sum()
    print(f'Number of missing elements per each column \n {nan_counts}')
    if nan_exists:
        df.interpolate(method='linear', inplace=True)

def FRED_MD_make_stationary(data, tcode):
    # Preallocate the output with NaNs
    y = pd.Series(np.nan, index=data.index)

    if tcode == 1:  # Level
        y = data

    elif tcode == 2:  # First difference
        y = data.diff()

    elif tcode == 3:  # Second difference
        y = data.diff().diff()

    elif tcode == 4:  # Natural log
        y = np.where(data > 0, np.log(data), np.nan)

    elif tcode == 5:  # First difference of natural log
        y = np.where(data > 0, np.log(data).diff(), np.nan)

    elif tcode == 6:  # Second difference of natural log
        y = np.where(data > 0, np.log(data).diff().diff(), np.nan)

    elif tcode == 7:  # First difference of percent change
        y1 = data.pct_change()
        y = y1.diff()
    return y


def add_lags (df, columns_to_lag, lags):
    # adds the lags for the columns specified in the list columns_to_lag
    for col in columns_to_lag:
        for i in range(1,lags+1):
            df[f'{col}_LAG{i}'] = df[col].shift(i)
    return df


def autocorrelation_analysis (data, n_lags, save_path):
    # Compute ACF and PACF
    acf = sm.tsa.acf(data, fft=False)
    print(f'ACF coefficients {acf}')
    pacf = sm.tsa.pacf(data)
    title_font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16}
    label_font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 14}

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Plot ACF with gridlines
    sm.graphics.tsa.plot_acf(data, lags=n_lags, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)', fontdict=title_font)
    ax1.grid(which='both')

    # Plot PACF with gridlines
    sm.graphics.tsa.plot_pacf(data, lags=n_lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)', fontdict=title_font)
    ax2.grid(which='both')  # Add gridlines

    num_ticks = 10
    ax1.set_xticks(np.arange(0, n_lags + 1, n_lags // num_ticks))
    ax2.set_xticks(np.arange(0, n_lags + 1, n_lags // num_ticks))

    ax1.set_xlabel('Lags', fontdict=label_font)
    ax2.set_xlabel('Lags', fontdict=label_font)
    ax1.set_ylabel('ACF', fontdict=label_font)
    ax2.set_ylabel('PACF', fontdict=label_font)

    plt.tight_layout()
    plt.savefig(save_path)


def real_yield_plot(save_graph1_path, save_graph2_path):
    ### Plot Real Yields ###
    df_real_yield = pd.read_csv(real_yield_path, parse_dates=['Date'])
    df_real_yield.set_index('Date', inplace=True)
    df_real_yield.sort_index(inplace=True)
    df_real_yield[df_real_yield.columns.difference(['Date'])] = df_real_yield[df_real_yield.columns.difference(['Date'])].apply(pd.to_numeric, errors='coerce')
    df_real_yield = df_real_yield.resample('W').mean()
    plot_labels = {'5_YR_R' : '5y TIPS', '7_YR_R' : '7y TIPS', '10_YR_R': '10y TIPS'}
    
    plt.figure(figsize=(12, 6))
    for column in y_types:
        plt.plot(df_real_yield.index, df_real_yield[column], label=plot_labels[column],linewidth=1.5)
    
    #plt.title(f'Time Series Plot for 5y, 7y and 10y Real Yields')
    font_size = 14
    plt.xlabel('Date', fontsize=font_size)
    plt.ylabel('Real Yield',fontsize=font_size)
    plt.grid(True)
    plt.xticks(fontsize=font_size)  # Adjust rotation and fontsize for better readability
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size)
    #plt.show()
    plt.savefig(save_graph1_path)

    ### Plot for Differences ###
    df_real_yield_diff = df_real_yield  - df_real_yield.shift(1)
    df_real_yield_diff = df_real_yield_diff.iloc[1:] 

    plt.figure(figsize=(12, 6))
    plt.plot(df_real_yield_diff.index, df_real_yield_diff['10_YR_R'], label=plot_labels['10_YR_R'],linewidth=1.0)

    # Set the y-axis label
    font_size = 14
    plt.xlabel('Date', fontsize=font_size)
    plt.ylabel('Real Yield Differences',fontsize=font_size)
    plt.grid(True)
    plt.xticks(fontsize=font_size)  # Adjust rotation and fontsize for better readability
    plt.yticks(fontsize=font_size)
    plt.legend(loc='upper left',fontsize=font_size)

    # Create a new axes on the right for the histogram
    divider = make_axes_locatable(plt.gca())
    ax_histy = divider.append_axes("right", 1.5, pad=0.1, sharey=plt.gca())

    # Plot a histogram using another column (e.g., 'HistogramData') from your DataFrame
    ax_histy.hist(df_real_yield_diff['10_YR_R'], bins=100, orientation='horizontal', color='green')
    ax_histy.get_xaxis().set_visible(False)
    ax_histy.grid(True)

    plt.savefig(save_graph2_path)


def correlation_heatmap ( ):
    # Load the data from the provided file
    data = pd.read_csv(features_path)
    
    # Convert the 'Date' column to datetime and set it as the index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Calculate the correlation matrix
    correlation_matrix = data.corr( )

    # Set the font scale (adjust 1.5 to your preference)
    sns.set(font_scale=1.3)

    # Setting the size of the plot
    plt.figure(figsize=(20, 15))

    # Creating a heatmap
    heatmap = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)

    # Setting the title
    plt.title('Heatmap of the Correlation Matrix of the Market Data Features')

    # Adjust layout to fit
    plt.tight_layout()

    # Rotate x-axis labels
    plt.xticks(rotation=65)

    # Save the heatmap as an image file
    plt.show()
    plt.savefig(heatmap_file_path)






