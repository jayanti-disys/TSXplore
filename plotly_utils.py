import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta


def outliers_with_quantile (X):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    Xout = X[((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR)))]
    return  Xout


def get_stl_outliers (data,period):
    result = seasonal_decompose(data, model='additive', extrapolate_trend='freq', period=period)
    resid = result.resid
    Xout = outliers_with_quantile (resid)
    return data.loc[Xout.index]




def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    return dfoutput

def get_lineplot(df, column_name):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=False,
        horizontal_spacing=0.1, vertical_spacing=0.05)

    fig1 = px.line(df, 'Date', column_name)
    trace1 = fig1['data'][0]
    fig.add_trace(trace1, row=1, col=1)
    fig.update_traces(line=dict(width=4.0))

    return fig


def get_histogram(df, column_name, nbins):
    if nbins:
        nbins = nbins
    else:
        nbins = 100
    fig = make_subplots(rows=1, cols=1, shared_xaxes=False,
         horizontal_spacing=0.1, vertical_spacing=0.05)
    fig2 = px.histogram(df, x=column_name, nbins=nbins)
    trace2 = fig2['data'][0]
    fig.add_trace(trace2, row=1, col=1)

    return fig


def plot_rolling_mean(df, column_name, window_size):
    df = df.sort_values(by=['Date'])
    df.index = df['Date'].to_list()
    S = df[column_name]
    X = S.rolling(window_size).mean()
    Y = S.rolling(window_size).std()

    fig = make_subplots(rows=1, cols=1, shared_xaxes=False,
        horizontal_spacing=0.1, vertical_spacing=0.05)

    fig1 = px.line(df, 'Date', column_name)
    trace1 = fig1['data'][0]

    fig2 = px.line(X)
    trace2 = fig2['data'][0]

    fig3 = px.line(Y)
    trace3 = fig3['data'][0]

    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)
    fig.add_trace(trace3, row=1, col=1)

    colors = ['blue', 'red', 'black']
    names = ['Raw', 'Rolling Mean', 'Rolling Std']
    for i in range (0,3):
        fig['data'][i]['line']['color'] = colors[i]
        fig['data'][i]['name'] = names[i]

    return fig


def get_figure(df, column_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
        horizontal_spacing=0.1, vertical_spacing=0.05)

    fig1 = px.line(df, 'Date', column_name)
    trace1 = fig1['data'][0]
    fig.add_trace(trace1, row=1, col=1)

    fig2 = px.histogram(df, x=column_name)
    trace2 = fig2['data'][0]
    fig.add_trace(trace2, row=2, col=1)

    return fig


def get_seasonality(df, column_name):
    df = df.sort_values(by=['Date'])
    df.index = df['Date'].to_list()

    data = df[column_name]

    result = seasonal_decompose(data, model='additive', extrapolate_trend='freq', period=7)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=False,
        horizontal_spacing=0.1, vertical_spacing=0.05)

    fig1 = px.line(result.observed)
    trace1 = fig1['data'][0]
    fig.add_trace(trace1, row=1, col=1)

    fig2 = px.line(result.trend)
    trace2 = fig2['data'][0]
    fig.add_trace(trace2, row=2, col=1)

    fig3 = px.line(result.seasonal)
    trace3 = fig3['data'][0]
    fig.add_trace(trace3, row=3, col=1)

    fig4 = px.line(result.resid)
    trace4 = fig4['data'][0]
    fig.add_trace(trace4, row=4, col=1)

    labels = ['Raw', 'Trend', 'Seasonal', 'Residual']
    colors = ['blue', 'red', 'green', 'black']
    for i in range(0, 4):
        fig['data'][i]['showlegend'] = True
        fig['data'][i]['name'] = labels[i]
        fig['data'][i]['line']['color'] = colors[i]

    fig.update_traces()

    fig.update_layout(title=column_name,
        font=dict(family="Times Roman, monospace", size=12, color="Black")
    )

    fig.update_layout(showlegend=True)
    fig.update_layout({'legend_orientation': 'v'})

    return fig


def predict_arima(df, column_name,num_forecast):
    df = df.sort_values(by=['Date'])
    df.index = df['Date'].to_list()

    data = df[column_name]

    model = ARIMA(data.values, order=(3, 1, 2))
    model_fit = model.fit()

    forecast_data = model_fit.forecast(num_forecast)
    start_time = data.index[-1]
    end_time = start_time + timedelta(days=num_forecast - 1)
    forecast_index = pd.date_range(start=data.index[-1], end=end_time)
    data_forecast = pd.Series(data=forecast_data, index=forecast_index)

    fig = make_subplots(rows=1, cols=1, shared_xaxes=False,
                        horizontal_spacing=0.1, vertical_spacing=0.05)

    fig1 = px.line(df.iloc[-2*num_forecast:], 'Date', column_name,  markers=True)
    trace1 = fig1['data'][0]

    fig2 = px.line(data_forecast, markers=True)
    trace2 = fig2['data'][0]

    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)

    colors = ['blue', 'red']
    names = ['Training', 'Forecast']

    for i in range(0, 2):
        fig['data'][i]['line']['color'] = colors[i]
        fig['data'][i]['name'] = names[i]

    return fig


def get_outliers (df, column_name,  threshold):

    X = df[column_name]

    Y = get_stl_outliers (X,7)

    df_out = pd.DataFrame(columns=['Date','Value'])
    df_out['Date'] = Y.index
    df_out['Value'] = Y.values

    fig = make_subplots(rows=1, cols=1, shared_xaxes=False,
         horizontal_spacing=0.1, vertical_spacing=0.05)

    fig1 = px.line(X,  markers=False)
    trace1 = fig1['data'][0]

    fig2 = px.scatter(x=Y.index, y=Y.values)
    trace2 = fig2['data'][0]

    colors = ['blue', 'red']
    names = ['Raw', 'Outliers']
    symb = ['line','marker']
    
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)
    
    for i in range (0, 2):
        fig['data'][i][symb[i]]['color'] = colors[i]
        fig['data'][i]['name'] = names[i]
        fig['data'][i]['showlegend'] = True  
    

    
    return fig, df_out

   
