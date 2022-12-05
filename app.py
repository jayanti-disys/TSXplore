import argparse
import pandas as pd
import dash
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import html
from dash import dcc
global input_data_dir
import io
import base64
from plotly_utils import get_figure, get_seasonality


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'top': 20,
    'left': 25,
    'bottom':50 ,
    'height': '20%',
    'width': '50%',
    'font-size':'22px',
    'padding': '20px 10px',
    'background-color': '#3498DB'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'width': '50%',
    'top': 200,
    'padding': '20px 10px'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#000000',
    'font-size':'40px',
    'font-color':'#191970'
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
                        html.Div(className="col", children=[
                            html.H2('Column', style={'textAlign': 'center'}),
                                dcc.Dropdown(
                                   id='column_name'
                                )
                        ]
                                 ),style={'width':'3000px','textAlign': 'center','border': '4px solid black','background-color':'#C0C0C0'}
                    ),

                    html.Td(
                        html.Div(className="col", children=[
                            html.H2('Options', style={'textAlign': 'center'}),
                                dcc.Dropdown(
                                     id='option',value='View',
                                     options=[{'label':  i, 'value': i} for i in ['View','STL','Rolling','anomaloy']],
                                )
                        ]
                                 ),style={'width':'3000px','textAlign': 'center','border': '4px solid black','background-color':'#C0C0C0'}
                    ),

                    html.Td(
                        html.Div(className="col", children=[
                            html.H2('Parameter', style={'textAlign': 'center'}),
                                dcc.Dropdown(
                                    id='param', value=1,
                                    options=[{'label': i, 'value': i} for i in [1,2,5,10,20]],
                                )
                        ]
                                 ), style={'width': '3000px', 'textAlign': 'center', 'border': '4px solid black',
                                  'background-color': '#C0C0C0'}
                    ),
               ]),
           ],style={'width':'70%'}),
        ]),
    ]
)

uploader = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])



sidebar = html.Div(
    [
        html.H1('Time Series Analysis', style={'font-color':'#fff004'}),
        html.Hr(),
        uploader,
        controls,
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(
    [
        dcc.Graph(id='indicator-graphic'),
        html.H2(children='Data Summary',style={'text-align': 'left','margin-left': '7%'}),
        html.Div(
            dash_table.DataTable(
                id='summary-table1',
                columns=[
                    {"name": i, "id": i} for i in ['Key','Value']],

            style_data={'whiteSpace': 'normal','height': 'auto',},

            style_data_conditional=[
                {
                  'if': {'row_index': 'odd'},
                   'backgroundColor': 'rgb(220, 220, 220)',
               }],
                          ),
            style={'width': '35%','font-size': '1.5em','margin-left': '5%'},
        ),
        html.Table([html.Tr([html.Td(html.H1(children= '(c) DISYS 2022', style={'text-align': 'center'}))])],
        style={'width':'50%','float':'center','background-color':'#3498DB'}),
    ])





app.layout = html.Div([sidebar, content])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    df = pd.DataFrame()
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        #return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('column_name', 'options'),
      [Input('upload-data', 'contents'),
      State('upload-data', 'filename'),
      State('upload-data', 'last_modified')],
      )
def update_dropdown_columns2(list_of_contents, list_of_names, list_of_dates):
    children = []
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

    try:
        df = children[0]
    except:
        df = pd.read_csv("data/HCL.csv", parse_dates=['Date'])


    columns = list(df.columns)
    return [{'label': i, 'value': i} for i in columns[1:]]


def summary_table(D):
    df = pd.DataFrame()
    df['Key'] = D.keys()
    df['Value'] = D.values() 
    return df.to_dict('records')


@app.callback([Output('indicator-graphic', 'figure'),Output('summary-table1', "data"),],
     [Input('column_name', 'value'),Input('option', 'value'),Input('param','value'),
      Input('upload-data', 'contents'),
      State('upload-data', 'filename'),
      State('upload-data', 'last_modified')
      ])
def update_graph(column_name,option,param,list_of_contents, list_of_names, list_of_dates):
    pd.options.plotting.backend = "plotly"

    children = []
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

    if len (children)  < 1:
        from plotly.subplots import make_subplots
        return [make_subplots(rows=1, cols=2), summary_table({})]
    else:
        df = children[0]
        name = list_of_names[0]

        df['Date'] = pd.to_datetime(df['Date'])
        df.index = df['Date'].to_list()

        if column_name:
            X = df[column_name]
            D = {"Number of records": len(df),
               "Start": df.index[0],
               "End": df.index[-1],
               "Minimum:": X.min(),
               "Maximum": X.max(),
               "Mean": "%.2f" % X.mean(),
               "Variance": "%.2f" % X.var()}
        else:
            D = {}
        if option == 'STL':
            fig = get_seasonality(df, column_name)
        elif option == 'Rolling':
            df1 = df.rolling(param)
            fig = get_figure(df1, column_name)
        else:
            fig = get_figure(df, column_name)
        fig.update_layout(height=1000, width=1600, title_text=name)

        return [fig, summary_table(D)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    app.run_server(host='0.0.0.0')


