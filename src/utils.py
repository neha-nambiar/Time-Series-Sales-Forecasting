import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet


# Define Ecuador holidays to be used in model building
ecuador_holidays = pd.DataFrame({
    'holiday': 'ecuador_holidays',
    'ds': pd.to_datetime([
        '2013-01-01', '2013-02-04', '2013-02-05', '2013-03-29', '2013-04-01',
        '2013-05-01', '2013-05-24', '2013-08-10', '2013-10-09', '2013-11-02',
        '2013-11-03', '2013-12-25',
        '2014-01-01', '2014-02-03', '2014-02-04','2014-03-28', '2014-04-01',
        '2014-05-01', '2014-05-24', '2014-08-10','2014-10-09', '2014-11-02',
        '2014-11-03', '2014-12-25',
        '2015-01-01', '2015-02-04', '2015-02-05', '2015-03-29', '2015-04-01',
        '2015-05-01', '2015-05-24', '2015-08-10', '2015-10-09', '2015-11-02',
        '2015-11-03', '2015-12-25',
        '2016-01-01', '2016-02-04', '2016-02-05', '2016-03-29', '2016-04-01',
        '2016-05-01', '2016-05-24', '2016-08-10', '2016-10-09', '2016-11-02',
        '2016-11-03', '2016-12-25',
        '2017-01-01', '2017-02-04', '2017-02-05', '2017-03-29', '2017-04-01',
        '2017-05-01', '2017-05-24', '2017-08-10'
    ]),
    'lower_window': -2,
    'upper_window': 1,
})


def mape_func(actual, pred):         
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    """
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5, 0.9],
    'yearly_seasonality': [1, 2, 3, 4, 5],
    'holidays_prior_scale': [0.001, 0.002, 0.003, 0.004, 0.005],
    'fourier_order_weekly': [1, 2, 3],
    'fourier_order_quarterly': [1, 2, 3, 4],
    'fourier_order_daily': [5, 10, 15, 20, 25],
    'prior_scale_weekly': [0.2, 0.4,0.6, 0.8],
    'prior_scale_quarterly': [0.002, 0.004, 0.006, 0.008, 0.01]
}


def data_transform(df):
    """
    Transform the input DataFrame as required by the Prophet model.
    
    """
    df['date'] = pd.to_datetime(df['date'])
    df['unit_sales'] = df['unit_sales'].astype(float)
    df.rename(columns={'date': 'ds', 'unit_sales': 'y'}, inplace=True)
    df = df.groupby('ds').sum()
    cleaned_df = df.reset_index()
    return cleaned_df


def hyperparameter_tuning(cleaned_df):
    """
    Performs hyperparameter tuning for the Prophet model.
    
    Parameters:
    cleaned_df (pd.DataFrame): Transformed DataFrame with columns 'ds' and 'y'
    as required by the Prophet model.
    
    Returns:
    dict: Dictionary containing the best hyperparameters.

    """
    
    train_df = cleaned_df[:-30]
    prediction_days= 30
    
    """
    Finding the value for the hyperparameter 'changepoint_prior_scale'
    that give the least mean absolute percentage error.  
    """

    best_mape = float('inf')
    best_cp_scale = None

    for cp_scale in param_grid['changepoint_prior_scale']:

        # Initialize Prophet model with current parameter
        m = Prophet(seasonality_mode='multiplicative',
                    changepoint_prior_scale=cp_scale)

        m.fit(train_df)
        future_df = m.make_future_dataframe(periods= prediction_days)
        prophet_train = m.predict(future_df)

        # Evaluate with Mean Absolute Percentage Error (MAPE)
        mape = mape_func(cleaned_df['y'], prophet_train['yhat'])

    # Update best parameter if current MAPE is lower
    if mape < best_mape:
        best_mape = mape
        best_cp_scale = cp_scale
        

    """
    Finding the value for the hyperparameter 'holidays_prior_scale'
    that give the least mean absolute percentage error.  
    """

    best_mape = float('inf')
    best_h_scale = None

    for h_scale in param_grid['holidays_prior_scale']:

    # Initialize Prophet model with current parameter
        m = Prophet(seasonality_mode='multiplicative',
                    holidays=ecuador_holidays,
                    holidays_prior_scale=h_scale)

        m.fit(train_df)
        future_df = m.make_future_dataframe(periods= prediction_days)
        prophet_train = m.predict(future_df)

        # Evaluate with Mean Absolute Percentage Error (MAPE)
        mape = mape_func(cleaned_df['y'], prophet_train['yhat'])

    # Update best parameters if current mape is lower
    if mape < best_mape:
        best_mape = mape
        best_h_scale = h_scale

    """
    Finding the value for the hyperparameter 'yearly_seasonality'
    that give the least mean absolute percentage error.  
    """

    best_mape = float('inf')
    best_y_scale = None

    for y_scale in param_grid['yearly_seasonality']:

        # Initialize Prophet model with current parameter
        m = Prophet(seasonality_mode='multiplicative',
                    yearly_seasonality=y_scale)

        m.fit(train_df)
        future_df = m.make_future_dataframe(periods= prediction_days)
        prophet_train = m.predict(future_df)

        # Evaluate with Mean Absolute Percentage Error (MAPE)
        mape = mape_func(cleaned_df['y'], prophet_train['yhat'])

    # Update best parameters if current mape is lower
    if mape < best_mape:
        best_mape = mape
        best_y_scale = y_scale

    """
    Finding the value for the hyperparameter 'fourier_order_weekly'
    that give the least mean absolute percentage error.  
    """

    best_mape = float('inf')
    best_fow = None

    for fow in param_grid['fourier_order_weekly']:

    # Initialize Prophet model with current parameters
        m = Prophet(seasonality_mode='multiplicative')
        m.add_seasonality(name='weekly', period=7, fourier_order=fow, mode='multiplicative')

        m.fit(train_df)
        future_df = m.make_future_dataframe(periods= prediction_days)
        prophet_train = m.predict(future_df)

        # Evaluate with Mean Absolute Percentage Error (MAPE)
        mape = mape_func(cleaned_df['y'], prophet_train['yhat'])

    # Update best parameters if current mape is lower
    if mape < best_mape:
        best_mape = mape
        best_fow = fow

    """
    Finding the value for the hyperparameter 'fourier_order_daily'
    that give the least mean absolute percentage error.  
    """

    best_mape = float('inf')
    best_fod = None

    for fod in param_grid['fourier_order_daily']:

        # Initialize Prophet model with current parameters
        m = Prophet(seasonality_mode='multiplicative')
        m.add_seasonality(name='daily', period=1, fourier_order=fod, mode='multiplicative')

        m.fit(train_df)
        future_df = m.make_future_dataframe(periods= prediction_days)
        prophet_train = m.predict(future_df)

        # Evaluate with Mean Absolute Percentage Error (MAPE)
        mape = mape_func(cleaned_df['y'], prophet_train['yhat'])

    # Update best parameters if current mape is lower
    if mape < best_mape:
        best_mape = mape
        best_fod = fod

    """
    Finding the value for the hyperparameter 'fourier_order_quarterly'
    that give the least mean absolute percentage error.  
    """

    best_mape = float('inf')
    best_foq = None

    for foq in param_grid['fourier_order_quarterly']:

    # Initialize Prophet model with current parameters
        m = Prophet(seasonality_mode='multiplicative')
        m.add_seasonality(name='quarterly', period=91.25, fourier_order=foq, mode='multiplicative')

        m.fit(train_df)
        future_df = m.make_future_dataframe(periods= prediction_days)
        prophet_train = m.predict(future_df)

        # Evaluate with Mean Absolute Percentage Error (MAPE)
        mape = mape_func(cleaned_df['y'], prophet_train['yhat'])

    # Update best parameters if current mape is lower
    if mape < best_mape:
        best_mape = mape
        best_foq = foq
        
    """
    Finding the value for the hyperparameter 'prior_scale_weekly'
    that give the least mean absolute percentage error.  
    """

    best_mape = float('inf')
    best_psw = None

    for psw in param_grid['prior_scale_weekly']:

        # Initialize Prophet model with current parameters
        m = Prophet(seasonality_mode='multiplicative')
        m.add_seasonality(name='weekly', period=7, fourier_order=best_fow , prior_scale=psw, mode='multiplicative')

        m.fit(train_df)
        future_df = m.make_future_dataframe(periods= prediction_days)
        prophet_train = m.predict(future_df)

        # Evaluate with Mean Absolute Percentage Error (MAPE)
        mape = mape_func(cleaned_df['y'], prophet_train['yhat'])

    # Update best parameters if current mape is lower
    if mape < best_mape:
        best_mape = mape
        best_psw = psw

    """
    Finding the value for the hyperparameter 'prior_scale_quarterly'
    that give the least mean absolute percentage error.  
    """

    best_mape = float('inf')
    best_psq = None

    for psq in param_grid['prior_scale_quarterly']:

        m = Prophet(seasonality_mode='multiplicative')
        m.add_seasonality(name='quarterly', period=91.25, fourier_order=best_foq, prior_scale=psq, mode='multiplicative')

        m.fit(train_df)
        future_df = m.make_future_dataframe(periods= prediction_days)
        prophet_train = m.predict(future_df)

        mape = mape_func(cleaned_df['y'], prophet_train['yhat'])

    # Update best parameters if current mape is lower
    if mape < best_mape:
        best_mape = mape
        best_psq = psq
        
    """
    Outputs dictionary containing the best hyperparameter values 
    for the data.
    """

    best_params = {
        'changepoint_prior_scale': best_cp_scale,
        'holidays_prior_scale': best_h_scale,
        'yearly_seasonality': best_y_scale,
        'fourier_order_daily': best_fod,
        'fourier_order_weekly': best_fow,
        'fourier_order_quarterly': best_foq,
        'prior_scale_weekly': best_psw,
        'prior_scale_quarterly': best_psq
    }

    return best_params


def make_prediction(cleaned_df, best_params):
    
    """
    Makes predictions using the Prophet model with the best hyperparameters.

    This function splits the cleaned data into training and test sets, initializes the Prophet model
    with the best hyperparameters, fits the model on the training data, and makes predictions for the
    next 30 days.

    Parameters:
    cleaned_df (pd.DataFrame): Transformed DataFrame with columns 'ds' and 'y'.
    
    best_params (dict): Dictionary containing the best hyperparameters for the model. Expected keys are:
        - 'changepoint_prior_scale': float, the scale for the changepoints
        - 'yearly_seasonality': int, the Fourier order for yearly seasonality
        - 'holidays_prior_scale': float, the scale for the holidays component
        - 'fourier_order_weekly': int, the Fourier order for weekly seasonality
        - 'fourier_order_quarterly': int, the Fourier order for quarterly seasonality
        - 'fourier_order_daily': int, the Fourier order for daily seasonality
        - 'prior_scale_weekly': float, the scale for the weekly seasonality component
        - 'prior_scale_quarterly': float, the scale for the quarterly seasonality component

    Returns:
    pd.DataFrame: DataFrame containing the predicted values ('ds' and 'yhat') for the next 30 days.

   """

    changepoint_prior_scale = best_params['changepoint_prior_scale']
    yearly_seasonality = best_params['yearly_seasonality']
    holidays_prior_scale = best_params['holidays_prior_scale']
    fourier_order_weekly = best_params['fourier_order_weekly']
    fourier_order_quarterly = best_params['fourier_order_quarterly']
    fourier_order_daily = best_params['fourier_order_daily']
    prior_scale_weekly = best_params['prior_scale_weekly']
    prior_scale_quarterly = best_params['prior_scale_quarterly']

    df_train = cleaned_df[:-30]
    df_test = cleaned_df[-30:]

    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=yearly_seasonality, holidays=ecuador_holidays,
                holidays_prior_scale=holidays_prior_scale, changepoint_prior_scale=changepoint_prior_scale)
    m.add_seasonality(name='weekly', period=7, fourier_order=fourier_order_weekly, prior_scale=prior_scale_weekly, mode='multiplicative')
    m.add_seasonality('quarterly', period=91.25, fourier_order=fourier_order_quarterly, prior_scale=prior_scale_quarterly, mode='multiplicative')
    m.add_seasonality('daily', period=1, fourier_order=fourier_order_daily, mode='multiplicative')

    m.fit(df_train)
    prediction_days= 30
    future_df = m.make_future_dataframe(periods= prediction_days)
    prophet_train = m.predict(future_df)
    predicted_df = prophet_train[['ds','yhat']]
    return predicted_df


def plot(predicted_df):
    
    """
    Plots the actual vs. predicted sales.

    This function creates a line plot comparing the actual sales to the predicted sales.

    Parameters:
    cleaned_df (pd.DataFrame): DataFrame containing the actual sales data with columns 'ds' and 'y'.
    predicted_df (pd.DataFrame): DataFrame containing the predicted sales data with columns 'ds' and 'yhat'.

    """

    plt.figure(figsize=(100, 50))
    sns.lineplot(data=cleaned_df, x="ds", y="y", label="Actual", color='blue')
    sns.lineplot(data=predicted_df, x="ds", y="yhat", label="Predicted", color='red')
    plt.title("Actual vs. Predicted Sales", fontsize=50)
    plt.xlabel("Date", fontsize=50)
    plt.ylabel("Sales", fontsize=50)
    plt.legend(fontsize=50)
    plt.show()
    
    
def plot_prediction(df):
    
        """
    Transform the data, tune hyperparameters, make predictions, and plot the results.

    This function performs the following steps:
    1. Transforms the input data using the `data_transform` function.
    2. Tunes hyperparameters using the `hyperparameter_tuning` function.
    3. Makes predictions using the `make_prediction` function.
    4. Plots the actual vs. predicted sales using the `plot` function.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the raw sales data.

    """
    
    # Step 1: Transform the data
    cleaned_df = data_transform(df)

    # Step 2: Tune hyperparameters
    best_params = hyperparameter_tuning(cleaned_df)

    # Step 3: Make predictions
    predicted_df = make_prediction(cleaned_df, best_params)

    # Step 4: Plot the results
    plot(predicted_df)
    
