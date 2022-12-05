import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

def get_figure(df, column_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
        horizontal_spacing=0.1, vertical_spacing=0.05, subplot_titles=([column_name]))

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
    fig.update_layout(width=600, height=600, margin=dict(l=10, r=10, t=100, b=40))

    return fig

