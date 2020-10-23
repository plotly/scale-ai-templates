import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import scipy as sp
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model


def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url('dash-logo.png'),
        style={'float': 'right', 'height': 60}
    )
    link = html.A(logo, href="https://plotly.com/dash/")
    
    return dbc.Row([dbc.Col(title, md=8), dbc.Col(link, md=4)])


# CONSTANTS
numeric_cols = ['Age', 'Infrared Scan Results', 'Loading']

# LOAD DATA
spreasheets = pd.read_excel('./data.xlsx', sheet_name=list(range(5)))
df = pd.concat(spreasheets.values()).drop(columns=["ID", 'Heath Index'])
df.columns = df.columns.str.strip()

# CREATE ENCODER AND MODELS
oh_enc = OneHotEncoder(sparse=False)
models = {
    'Ridge': linear_model.RidgeClassifier(),
    'Logistic (L-BFGS)': linear_model.LogisticRegression(),
    'Logistic (SAGA)': linear_model.LogisticRegression(solver='saga'),
    'SGD': linear_model.SGDClassifier(),
}

# PREPROCESS DATASET
X = np.hstack([
    oh_enc.fit_transform(df[['Visual Conditions']]),
    df[numeric_cols].values
])
y = df['Oil Leak'].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

# train model
for name, model in models.items():
    model.fit(X_train, y_train)

# APP STARTS HERE
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE])
server = app.server

controls = [
    dbc.FormGroup([
        dbc.Label("Age"),
        dcc.Slider(id='age', min=0, max=50, step=1, marks={i: str(i) for i in range(0, 51, 10)}, value=25),
    ]),
    dbc.FormGroup([
        dbc.Label("Infrared Scan Results"),
        dcc.Slider(id='scan', min=0, max=1, step=0.01, marks={i: str(i) for i in [0, 0.5, 1]}, value=0.5)
    ]),
    dbc.FormGroup([
        dbc.Label("Visual Condition"),
        dbc.Select(
            id='visual', 
            value=df['Visual Conditions'].values[0],
            options=[{'label': x, 'value': x} for x in df['Visual Conditions'].unique()]
        ),
    ]),
    dbc.FormGroup([
        dbc.Label("Loading"),
        dcc.Slider(
            id='loading',
            value=df['Loading'].mean(),
            min=4,
            max=df['Loading'].max(),
            marks={i: str(i) for i in [4, 10, 20, 30, 40]},
            step=0.01
        )
    ]),
    dbc.FormGroup([
        dbc.Label("Model"),
        dbc.Select(
            id='model',
            value='Ridge',
            options=[{'value': x, 'label': x} for x in models]
        )
    ])

]

app.layout = dbc.Container(
    [
        Header("Transformer Maintenance Prediction", app),
        html.Hr(),
        dbc.CardDeck(
            [
                dbc.Card(controls, body=True, color="dark", outline=True),
                dbc.Card(dcc.Graph(id='graph-coef'), body=True, color="dark"),
                dbc.Card(dcc.Graph(id='graph-predict'), body=True, color="dark"),
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    Output('graph-coef', 'figure'),
    [Input('model', 'value')]
)
def update_graph_coefficient(model_name):
    model = models[model_name]

    coef_df = pd.DataFrame(
        model.coef_.T, 
        columns=model.classes_, 
        index=np.concatenate([oh_enc.categories_[0], numeric_cols])
    )

    coef_fig = px.bar(
        coef_df.T,
        title=f"{model_name} feature importance by class",
        labels={'index': 'class', 'value': 'Coefficient', 'variable': 'Feature'}
    )
    coef_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')

    return coef_fig


@app.callback(
    Output('graph-predict', 'figure'),
    [Input('age', 'value'), Input('scan', 'value'), Input('visual', 'value'), Input('loading', 'value'), Input('model', 'value')]
)
def update_graph_prediction(age, scan, visual, loading, model_name):
    model = models[model_name]

    future_ages = [age+offset for offset in np.arange(0, 10, 1)]

    samples = np.hstack([
        oh_enc.transform([[visual]]*10),
        np.array([[age, scan, loading] for age in future_ages]),
    ])

    preds = pd.DataFrame(
        sp.special.softmax(model.decision_function(samples), axis=1),
        columns=model.classes_
    )

    fig = px.area(
        preds,
        labels={'value': 'Softmax Score', 'index': 'Number of years from present', 'variable': 'Oil Leak'}, 
        title='Predicted Future Risk of Oil Leak'
    )

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
