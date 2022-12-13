import argparse
import pandas as pd
import dash
import dash_bootstrap_components as dbc
import dash_table
import io
import base64
from dash.dependencies import Input, Output, State
from dash import html
from dash import dcc
global input_data_dir
from plotly_utils import get_seasonality, plot_rolling_mean
from plotly_utils import get_lineplot, get_histogram, adf_test, predict_arima
from plotly.subplots import make_subplots

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

options = ['Time Series', 'Histogram', 'STL', 'ADF', 'Rolling', 'Forecast', 'Anomaloy']

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'top': 20,
    'left': 25,
    'bottom': 50,
    'height': '10%',
    'width': '100%',
    'font-size': '42px',
    'padding': '40px 40px',
    #'background-color': '#3498DE'
    'background-color':'#003366'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '5%',
    'margin-right': '5%',
    'width': '75%',
    'top': 200,
    'align':'center',
    'padding': '20px 10px'
}

TEXT_STYLE = {
    'textAlign': 'center',
    #'color': '#000000',
    'font-size': '40px',
    'font-color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}


controls = dbc.FormFloating(
    [
        html.Div(className="col", children=[
            html.Table([
                html.Tr([
                    html.Td(
                        html.Div(className="col", children=[
                            html.H2('Column', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='column_name'
                            )
                        ]
                        )
                    ),

                    html.Td(
                        html.Div(className="col", children=[
                            html.H2('Options', style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='option', value='Time Series',
                                    options=[{'label':  i, 'value': i} for i in options],
                            )
                        ]),
                    ),

                    html.Td(
                        html.Div(className="col", children=[
                            html.H2('Parameter', style={'textAlign': 'center'}),
                                dcc.Dropdown(
                                    id='param',
                                )
                        ]
                                 )
                    ),

                    html.Td(
                        html.Div(className="col", children=[
                            html.H2('Table View', style={'textAlign': 'center'}),
                            html.Div(children=[
                                dcc.RadioItems(id='table_view',
                                    options=['Summary', 'Data'],
                                    value= 'Summary',
                                    inline=True,
                                    style={"margin-left": "5px"},
                                )]
                            ,style={'font-size': '1.5em','textAlign':'center'})
                        ]
                        ,),
                        style={'width':'15%'},
                    ),




               ]),
           ],style={'width':'60%','height':'20%','font-size': '22px','border':'4px solid #000000',
                    'background-color':'#27AE60'}),
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
            'width': '60%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'font-size': '32px',
            'font-family': 'Lucida Console',
            'textAlign': 'center',
            'margin': '3px',
            'background-color':'#8798DB',
            'border':'4px solid #000000'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])



sidebar = html.Div(
    [
        html.H1('Time Series eXplorer v1.0 ', style={'color':'#FFFFFF'}),
        html.Hr(),
        uploader,
        controls,
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(
    [
        html.Div(
            dcc.Graph(id='indicator-graphic')
        , style={'width': '60%','height':'30%','font-size': '2.5em','margin-left': '5%'}
        ),
        html.H1(children='Data Summary',style={'text-align': 'left','margin-left': '5%'}),
        html.Div(
            dash_table.DataTable(
                id='summary-table1',
                style_data={'whiteSpace': 'normal','height': 'auto',},
            style_table={
                'overflowY': 'scroll'
            },
            page_size=30,
            style_data_conditional=[
                {
                  'if': {'row_index': 'odd'},
                   'backgroundColor': 'rgb(220, 220, 220)',
               }],
                          ),
            style={'width': '60%','font-size': '2.0em','margin-left': '5%'},
        ),
        html.Table([html.Tr([html.Td(html.H1(children= '(c) DISYS 2022', style={'text-align': 'center'}))])],
        style={'width':'100%','float':'center','background-color':'#003366',"color":"#FFFFFF"}),
    ])


app.layout = html.Div([sidebar, content],style={'width':'100%'})


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

    if len (children) > 0:
        df = children[0]
        columns = list (df.columns)
    else:
        columns = []
    return [{'label': i, 'value': i} for i in columns[1:]]


@app.callback(Output('param', 'options'),Input('option', 'value'))
def update_dropdown_param(option):
    if option == 'Rolling':
        values = [2, 4, 8, 16, 32, 64]
    elif option == 'Histogram':
        values = [100, 50, 20, 10, 5]
    elif option == 'Forecast':
        values = [2, 4, 8, 16]
    else:
        values = [None]

    return [{'label': i, 'value': i} for i in values]


def summary_table(D):
    df = pd.DataFrame()
    df['Key'] = D.keys()
    df['Value'] = D.values ()
    #df['Value'] = ["%.2f" %x for x in D.values()]
    df['SN'] = [int(i) for i in range (0, len(df))]
    comments = ["Number of Records",
                "Mean", "Standard Deviation","Minimum",
                "The 25% percentile","The 50% percentile",
                "The 75% percentile","Max"]
    if len(comments) == len(df):
        df['Comment'] = comments
    else:
        df['Comment'] = ["TBA"] * len (df)

    return df.to_dict('records')


@app.callback([Output('indicator-graphic', 'figure'),Output('summary-table1', "data"),],
     [Input('column_name', 'value'),Input('option', 'value'),Input('param','value'),
      Input('table_view', 'value'),Input('upload-data', 'contents'),
      State('upload-data', 'filename'),
      State('upload-data', 'last_modified')
      ])
def update_graph(column_name,option,param,table_view, list_of_contents, list_of_names, list_of_dates):
    pd.options.plotting.backend = "plotly"

    D = {}
    fig = make_subplots(rows=1, cols=2)
    fig.update_layout(
        title_text='Please upload a csv file and chose the option from drop downs'
    )
    #fig.update_layout(width=3200, height=1200, margin=dict(l=50, r=10, t=100, b=40),
    #                  font=dict(family="Courier New, monospace", size=28, color="RebeccaPurple"))

    children = []
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

    if len (children) > 0:
        df = children[0]
        df = df.dropna()
        if column_name in df.columns:
            if list_of_names and column_name:
                name = list_of_names[0].split(".")[0] + "[" + column_name + "]"
            else:
                name = "Template"

            df['Date'] = pd.to_datetime(df['Date'])
            df.index = df['Date'].to_list()

            D = dict(df[column_name].describe())

            if option == 'STL':
                fig = get_seasonality(df, column_name)
            elif option == 'Histogram':
                fig = get_histogram(df, column_name, param)
            elif option == 'Rolling' and param:
                fig = plot_rolling_mean(df, column_name, param)
            elif option == 'Forecast' and param:
                fig = predict_arima (df, column_name, param)
            elif option == 'ADF':
                fig = get_lineplot(df, column_name)
                df1 = adf_test(df[column_name])
                D = df1.to_dict()
            else:
                fig = get_lineplot(df, column_name)

            if table_view == 'Data':
                #fig = get_lineplot(df, column_name)
                D = df.to_dict('records')
                fig.update_layout(title_text=name)
                fig.update_layout(width=1800, height=600, margin=dict(l=50, r=10, t=100, b=40),
                    font=dict(family="Courier New, monospace", size=28, color="RebeccaPurple"))
                return [fig, D]

            fig.update_layout(title_text=name)

    fig.update_layout(width=1800, height=900, margin=dict(l=50, r=10, t=100, b=40),
        font=dict(family="Courier New, monospace",size=28,color="RebeccaPurple"))


    return [fig, summary_table(D)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    app.run_server(host='0.0.0.0')


