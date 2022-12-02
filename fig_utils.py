import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime 
from statsmodels.tsa.stattools import adfuller
from  statsmodels.tsa.arima.model import ARIMA 
from ts_predict import arima_predict



def check_stationarity (S):
    result = adfuller(S)
    D = {}
    D['ADF Statistic'] = '%f' % result[0]
    D['p-value'] = '%f' % result[1]
    return D  


def get_figure (df1,df2,*argv):

    rows  = argv[9]
    train_test_split = 0.85 
    time_window = argv[8]

    fig = make_subplots(rows=rows, cols=1,shared_xaxes=False,\
          horizontal_spacing=0.1, vertical_spacing=0.05)


    df1 = df1.sort_values(by=['Date'])
    df2 = df2.sort_values(by=['Date'])

    df1.index = df1['Date'].to_list()
    df2.index = df2['Date'].to_list()

    start_date,end_date  = argv[6], argv[7]

    df1 = df1.loc[(df1['Date'] >= start_date ) & (df1['Date'] < end_date )]
    df2 = df2.loc[(df2['Date'] >= start_date ) & (df2['Date'] < end_date )]
 
    S1 = df1[argv[1]]  

    D = check_stationarity (S1)


    name1 = argv[0] + "["+argv[1]+"]"

    if argv[4] == 'SMA':
       S2  = df2[argv[3]].rolling(argv[5]).mean()
       name2 = argv[2] + "["+argv[3]+","+'Moving Average'+"]"
    elif argv[4] == 'CMA':
       S2 = df2[argv[3]].expanding(argv[5]).mean()
       name2 = argv[3] + "["+argv[3]+","+'Cumulative Average'+"]"
    elif argv[4] == 'EMA':
       S2 = df2[argv[3]].ewm(argv[5]).mean()
       name2 = argv[2] + "["+argv[3]+","+'Exponetial Average'+"]"
    elif argv[4] == 'DIF':
       S2 = df2[argv[3]].diff()
       D = check_stationarity (S2)
       name2 = argv[2] + "["+argv[3]+","+'DIF'+"]"
    elif argv[4] == 'ARIMA':
       S2 = df2[argv[3]] 
       S_train = S2.iloc[:-time_window]
       S_test  = S2.iloc[-time_window:]
       P = arima_predict (S_train, S_test,argv[5])
       name2 = argv[2] + "["+argv[3]+","+'ARIMA'+"]"
    else:
       S2 = df2[argv[3]]  
       name2 = argv[2] + "["+argv[3]+"]"

    fig1 = px.line (S1)

    trace1 = fig1['data'][0]
    fig.add_trace(trace1, row=1, col=1)
    fig['data'][0]['showlegend']=True
    fig['data'][0]['name'] = name1
    fig['data'][0]['line']['color']="red"

    if len (argv) > 2:
        fig2 = px.line (S2)
        trace2 = fig2['data'][0]
        fig.add_trace(trace2, row=rows, col=1)
        fig['data'][1]['showlegend']=True
        fig['data'][1]['name'] = name2
        fig['data'][1]['line']['color']="blue"
        if  argv[4] == 'ARIMA':
            fig3 = px.line (P)
            trace3 = fig3['data'][0]
            fig.add_trace(trace3, row=rows, col=1)
            fig['data'][2]['showlegend']=True
            fig['data'][2]['name'] = 'ARIMA'
            fig['data'][2]['line']['color']="green"


    fig.update_traces()

    fig.update_layout(title="BSE Stock Prices",
       font=dict(family="Times Roman, monospace",size=22,color="Black")
    )

    fig.update_layout(showlegend = True)

    fig.update_layout({'legend_orientation':'h'})
    fig.update_layout(width=1800,height=900,margin=dict(l=10, r=10, t=100, b=40))
    return fig, D 

