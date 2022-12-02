import sys
import os
import base64
import argparse
import pandas as pd
import numpy as np
from datetime import date, datetime
import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from fig_utils import get_figure

global input_data_dir, df 
import glob 

input_data_dir="data/"
all_files = glob.glob(input_data_dir+os.sep+"*.csv")

all_files = [os.path.basename(a).replace(".csv","") for a in all_files]

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

start_date = datetime.strptime('2020-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2023-12-31', '%Y-%m-%d')

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    #'position': 'fixed',
    'top': 20,
    'left': 25,
    'bottom':50 ,
    'height': '20%',
    'width': '62%',
    'font-size':'22px',
    'padding': '20px 10px',
    'background-color': '#3CB371'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'width': '100%',
    'top': 200,
    'padding': '20px 10px'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#000000',
    'font-size':'40px',
    'font-color':'#FFBF00'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}


controls = dbc.FormGroup(
    [
       html.Div(className="col",children=[
       html.Table([
          html.Tr([
             html.Td(
                dcc.Dropdown(
                id='input1',value='PERSISTANT',
                options=[{'label':  i, 'value': i} for i in all_files],
             ),style={'width':'3000px','textAlign': 'center','border': '4px solid black','background-color':'#C0C0C0'}),
             html.Td(
                dcc.Dropdown(
                id='input2',value='Open Price',
               ),style={'width':'3000px','textAlign': 'center','border': '4px solid black','background-color':'#C0C0C0'}),
             html.Td(
                dcc.Dropdown(
                id='input3',value='PERSISTANT',
                options=[{'label':  i, 'value': i} for i in all_files],
               ),style={'width':'3000px','textAlign': 'center','border': '4px solid black','background-color':'#C0C0C0'}),
             html.Td(
                dcc.Dropdown(
                id='input4',value='Open Price',
               ),style={'width':'3000px','textAlign': 'center','border': '4px solid black','background-color':'#C0C0C0'}),

         ], ),

        
        html.Tr([
             html.Td(
                dcc.Dropdown(
                id='function1',value='MA',
                options=[{'label':  i, 'value': i} for i in ['SMA','CMA','EMA','DIF','ARIMA','SARIMA']],
             ),style={'width':'3000px','textAlign': 'center','border': '4px solid black','background-color':'#C0C0C0'}),
             html.Td(
                dcc.Dropdown(
                id='param1',value=2,
                options=[{'label':  i, 'value': i} for i in [1,2,5,7,10,20,30]],

               ),style={'width':'3000px','textAlign': 'center','border': '4px solid black','background-color':'#C0C0C0'}),

               html.Td(
                dcc.Dropdown(
                id='time_window',value=7,
                options=[{'label':  i, 'value': i} for i in [1,2,7,50]],
               ),style={'width':'3000px','textAlign': 'center','border': '4px solid black','background-color':'#C0C0C0'}),


               html.Td(
                  dcc.DatePickerRange(
                  id='date-picker-range',
                  start_date_placeholder_text="Start Date",
                  end_date_placeholder_text="End Date",
                  start_date=start_date,
                  min_date_allowed=date(1990, 1, 1),
                  max_date_allowed=date(2024, 12, 31),
                  end_date=end_date),
                  style={'fontSize': 14,'width':'3000px','textAlign': 'center','border': '4px solid black','background-color':'#C0C0C0'}),
         ], )
        
 

        ]),

        html.H2('Number of rows', style={'textAlign': 'left'}),
        dcc.RadioItems(
        id='nrows',
        options=[{'label': i, 'value': i} for i in [1,2]],
        value=1,
        labelStyle={'font-size':'40px', 'width': '10%','display': 'inline-block','float':'center','background-color':'#82E0AA'}),

        ]),
    ]

)
sidebar = html.Div(
    [
        html.H1('Financial Time series Analysis (FTSA) ', style=TEXT_STYLE),
        html.Hr(),
        controls,
    ],
    style=SIDEBAR_STYLE,
)



content = html.Div(
    [
       dcc.Graph(id='indicator-graphic'),

       html.H2(children='Stationarity',style={'text-align': 'left'}),
       html.Div(
            dash_table.DataTable(
                id='summary-table1',
                columns=[
                    {"name": i, "id": i} for i in ['Key','Value']],
            style_data={'whiteSpace': 'normal','height': 'auto',},
            ),
            style={'width': '30%','font-size': '1.5em'},
        ),

 
     html.Table([html.Tr([html.Td(html.H1(children= '(c) Jayanti Prasad 2022', style={'text-align': 'center'}))])],
         style={'width':'62%','float':'center','background-color':'#3CB371'}),




]) 
app.layout = html.Div([sidebar, content])


@app.callback(
    dash.dependencies.Output('input2', 'options'),
    [dash.dependencies.Input('input1', 'value')]
)
def update_dropdown_columns2(name):
    file_name = input_data_dir + os.sep + name + ".csv"
    df = pd.read_csv(file_name)
    columns = list(df.columns) 
    return [{'label': i, 'value': i} for i in columns[1:]]


@app.callback(
    dash.dependencies.Output('input4', 'options'),
    [dash.dependencies.Input('input3', 'value')]
)
def update_dropdown_columns4(name):
    #if name:
    file_name = input_data_dir + os.sep + name + ".csv"
    df = pd.read_csv(file_name)
    columns = list(df.columns)
    return [{'label': i, 'value': i} for i in columns[1:]]
    #else:
    #   return [{'None':'None'}]




def summary_table(D):
    df = pd.DataFrame()
    df['Key']  = D.keys()
    df['Value'] = D.values() 
    return df.to_dict('records')





@app.callback([Output('indicator-graphic', 'figure'),Output('summary-table1', "data")],
     [Input('input1', 'value'),Input('input2', 'value'), 
       Input('input3', 'value'),Input('input4', 'value'),
       Input('function1', 'value'),Input('param1', 'value'),
       Input('nrows', 'value'),Input('time_window', 'value'),
       Input(component_id='date-picker-range', component_property='start_date'),
       Input(component_id='date-picker-range', component_property='end_date'),])


def update_graph(input1,input2,input3,input4,function1,param1,nrows,time_window,start_date,end_date):

    file_name1 = input_data_dir + os.sep + input1 + ".csv"
    df1 = pd.read_csv(file_name1,parse_dates=['Date'])

    if input3 :
       file_name2 = input_data_dir + os.sep + input3 + ".csv"
       df2 = pd.read_csv(file_name2,parse_dates=['Date'])
    else:
       df2 = df1  


    fig,D = get_figure (df1,df2,input1,input2,input3,input4,function1,param1,start_date,end_date,time_window,nrows)

    return [fig,summary_table(D)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    app.run_server(host='0.0.0.0')


