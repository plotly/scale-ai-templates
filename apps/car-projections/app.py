import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash_table import DataTable
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE


def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("dash-logo.png"), style={"float": "right", "height": 60}
    )
    link = html.A(logo, href="https://plotly.com/dash/")

    return dbc.Row([dbc.Col(title, md=8), dbc.Col(link, md=4)])


used = pd.read_csv("vehicles_preprocessed.csv")
manufacturers = used.manufacturer.unique()
car_types = used.type.unique()

table_cols = ["Features", "Selected Values"]


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

controls = [
    dbc.FormGroup(
        [
            dbc.Label("Manufacturer"),
            dbc.Select(
                id="manufacturer",
                options=[{"label": x, "value": x} for x in manufacturers],
                value=manufacturers[0],
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Car Type"),
            dbc.Select(
                id="car-type",
                options=[{"label": x, "value": x} for x in car_types],
                value=car_types[0],
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Color By"),
            dbc.Select(
                id="color",
                options=[
                    {"label": x, "value": x}
                    for x in [
                        "state",
                        "paint_color",
                        "transmission",
                        "model",
                        "condition",
                    ]
                ],
                value="model",
            ),
        ]
    )
]

display_table = html.Div(
    [
        DataTable(
            id="table",
            columns=[{"name": i, "id": i} for i in table_cols],
            style_cell={
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "maxWidth": 0,
            }
        ),
        dbc.Alert("Click on point to learn more about a car", dismissable=True),
    ]
)

app.layout = dbc.Container(
    [
        dcc.Store(id="dataframe-store"),
        Header("Car Features Explorer with t-SNE", app),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Spinner(dcc.Graph(id="projection", style={"height": '500px'}), spinner_style={'margin': '225px auto'}),
                        dbc.Row([dbc.Col(x) for x in controls], form=True),
                        dbc.Alert("Click on point to learn more about a car", dismissable=True), 
                    ], 
                    md=7
                ),
                dbc.Col(
                    [
                        DataTable(
                            id="table",
                            columns=[{"name": i, "id": i} for i in table_cols],
                            style_cell={
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "maxWidth": 0,
                            }
                        ),
                    ],
                    md=5
                ),
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    [Output("projection", "figure"), Output("dataframe-store", "data")],
    [
        Input("manufacturer", "value"),
        Input("car-type", "value"),
        Input("color", "value"),
    ],
)
def query_and_project(manufacturer, car_type, color):
    queried = used.query(f'manufacturer == "{manufacturer}"').query(
        f'type == "{car_type}"'
    )

    if queried.shape[0] < 2:
        return px.scatter(title="Not enough datapoints"), queried.to_json()

    if queried.shape[0] > 500:
        queried = queried.sample(n=500, random_state=2020)

    vecs = pd.get_dummies(
        queried.drop(columns=["id", "vin", "url", "region_url", "image_url"])
    )

    reducer = TSNE(n_jobs=-1)
    projs = reducer.fit_transform(vecs)

    fig = px.scatter(projs, x=0, y=1, color=queried[color], title="t-SNE projections")

    return fig, queried.to_json()


@app.callback(
    Output("table", "data"),
    [Input("projection", "clickData")],
    [State("dataframe-store", "data")],
)
def update_table(click_data, df_json):
    if df_json is None or click_data is None:
        return dash.no_update

    df = pd.read_json(df_json)
    ix = click_data["points"][0]["pointIndex"]
    selected = df.iloc[ix].reset_index()
    selected.columns = table_cols

    recs = selected.to_dict("records")
    print(recs)

    return recs


if __name__ == "__main__":
    app.run_server(debug=True)
