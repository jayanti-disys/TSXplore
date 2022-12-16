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
from plotly_utils import get_seasonality, plot_rolling_mean, get_outliers
from plotly_utils import get_lineplot, get_histogram, adf_test, predict_arima
from plotly.subplots import make_subplots

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

options = ['Time Series', 'Histogram', 'STL', 'ADF', 'Rolling', 'Forecast', 'Outliers']

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
           ],style={'width':'60%','height':'20%','font-size': '24px','border':'4px solid #000000',
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
        #uploader,
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
            style={'width': '60%','font-size': '2.0em','margin-left': '1%'},
        ),

    ])


app.layout = html.Div([sidebar, content,uploader,
     html.Table(
            [html.Tr([html.Td(html.H1(children='(c) DISYS 2022', style={'text-align': 'center'}))])],
                 style={'width': '100%', 'float': 'center', 'background-color': '#003366',
                        "color": "#FFFFFF"})
],style={'width':'100%'})

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


@app.callback(Output('column_name', 'options'),
      Input('upload-data', 'contents'))
def update_dropdown_columns2(list_of_contents):
    if list_of_contents is not None:
        df = parse_contents(list_of_contents[0])
        columns = list (df.columns)
    else:
        columns = ["Test"]
    return [{'label': i, 'value': i} for i in columns[1:]]


@app.callback(Output('param', 'options'),Input('option', 'value'))
def update_dropdown_param(option):
    if option == 'Rolling':
        values = [2, 4, 8, 16, 32, 64]
    elif option == 'Histogram':
        values = [100, 50, 20, 10, 5]
    elif option == 'Forecast':
        values = [2, 4, 8, 16]
    elif option == 'Outliers':
        values = [0.1,0.2,0.3,0.4,0.5]
    else:
        values = [0]
    return [{'label': i, 'value': i} for i in values]


def summary_table(D):
    df = pd.DataFrame()

    df['Key'] = D.keys()
    df['Value'] = D.values ()
    #df['SN'] = [int(i) for i in range (0, len(df))]
    comments = ["Number of Records",
                "Mean", "Standard Deviation","Minimum",
                "The 25% percentile","The 50% percentile",
                "The 75% percentile","Max"]
    if len(comments) == len(df):
        df['Comment'] = comments
    else:
        df['Comment'] = ["TBA"] * len (df)

    return df.to_dict('records')


@app.callback([Output('indicator-graphic', 'figure'),
               Output('summary-table1', "data")
               ],
     [Input('column_name', 'value'),Input('option', 'value'),Input('param','value'),
      Input('table_view', 'value'),Input('upload-data', 'contents'),
      State('upload-data', 'filename'),
      ])
def update_graph(column_name,option,param,table_view, list_of_contents, list_of_names):
    pd.options.plotting.backend = "plotly"

    D = {'Version': '1.0 (December 2022)'}

    fig = make_subplots(rows=1, cols=2)
    fig.update_layout(title_text='Please upload a csv file and chose the option from drop downs')
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(width=3200, height=10, margin=dict(l=50, r=10, t=100, b=40),
                      font=dict(family="Courier New, monospace", size=28, color="RebeccaPurple"))
    return_table =  [D]
    if list_of_contents is not None:
        df = parse_contents(list_of_contents[0])
        df = df.dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df.index = df['Date'].to_list()

        if column_name in df.columns:
            D = dict(df[column_name].describe())

        if list_of_names and column_name:
            name = list_of_names[0].split(".")[0]  + "[" + column_name + "]"
        else:
            name = ""

        if table_view == 'Data':
             return_table = df.to_dict('records')
        else:
             return_table = summary_table(D)

        if option == 'STL':
             fig = get_seasonality(df, column_name)
        elif option == 'Histogram':
             fig = get_histogram(df, column_name, param)
        elif option == 'Rolling' and param:
             fig = plot_rolling_mean(df, column_name, param)
        elif option == 'Forecast' and param:
             fig = predict_arima (df, column_name, param)
        elif option == 'Outliers':
             fig, df1 = get_outliers(df, column_name, param)
        else:
             fig = get_lineplot(df, column_name)


        fig.update_layout(title_text=name)

    fig.update_layout(width=2800, height=900, margin=dict(l=50, r=10, t=100, b=40),
        font=dict(family="Courier New, monospace",size=28,color="RebeccaPurple"))

    return [fig ,  return_table]

if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
    #app.run_server(debug=True,port=8000)


