import string
import random

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_cytoscape as cyto
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import numpy as np

from utils.graph.cytoscape import run_network_algo_cytoscape

# Graph data
x_data_graph = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-12-31"))
y_data_graph = np.random.randint(10, size=len(x_data_graph))

# App description
description = dcc.Markdown(
    """ 
This Dash application contains **2 brand new features** for the [Dash Cytoscape](https://dash.plotly.com/cytoscape) library
and **an algorithim** that predicts the connections between various components (poles, transformers, and residential users) in an voltage network.

Features:

1. Leaflet map integration for the [Dash Cytoscape](https://dash.plotly.com/cytoscape) library
2. Context menu for the [Dash Cytoscape](https://dash.plotly.com/cytoscape) library (try right clicking on the graph)
3. An algorithim that takes in voltage network data and predicts the connections between the various components (poles, transformers, and residential users).

These features were created in collaboration with [Plotly](https://plotly.com/), [SCALEAI](https://www.scaleai.ca/), and developed by [Zyphr](https://www.zyphr.ca/).
"""
)


# Cytoscape data graph 1
LEAFLET_TILE_META = {
    "tileUrl": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}{r}.png",
    "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    "maxZoom": 30,
}

ALGO_DATA = {
    "Network 1": {
        "data": pd.read_csv("data/data.csv", sep=";", index_col=0),
        "leaflet": {
            **LEAFLET_TILE_META,
            "view": [45.43331550000074, -73.53991150001814, 15],
        },
    },
    "Network 2": {
        "data": pd.read_csv("data/data2.csv", sep=";", index_col=0),
        "leaflet": {
            **LEAFLET_TILE_META,
            "view": [45.434752, -73.53325, 15],
        },
    },
    "Network 3": {
        "data": pd.read_csv("data/data3.csv", sep=";", index_col=0),
        "leaflet": {
            **LEAFLET_TILE_META,
            "view": [45.434447, -73.532785, 15],
        },
    },
}

cyto.load_extra_layouts()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# App
app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H3("Dash Cytoscape: Leaflet & Context Menu Features"),
        html.Br(),
        dbc.Row(
            children=[
                dbc.Col(
                    width=4,
                    children=dbc.Card(
                        style={"border": "0px"},
                        children=dbc.CardBody(
                            style={"padding": "0px"},
                            children=[
                                dbc.Card(
                                    children=[
                                        dbc.CardHeader("Description"),
                                        dbc.CardBody(
                                            [
                                                html.P(description),
                                            ]
                                        ),
                                    ],
                                ),
                                html.Br(),
                                dbc.Card(
                                    children=[
                                        dbc.CardHeader("Control Panel"),
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "Select Dataset",
                                                    className="card-title",
                                                ),
                                                dcc.Dropdown(
                                                    id="dataset-dropdown",
                                                    value="Network 1",
                                                    options=[
                                                        {
                                                            "label": network,
                                                            "value": network,
                                                        }
                                                        for network in [
                                                            "Network 1",
                                                            "Network 2",
                                                            "Network 3",
                                                        ]
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                ),
                dbc.Col(
                    width=8,
                    children=dbc.Card(
                        style={"width": "100%", "height": "100%"},
                        children=[
                            cyto.Cytoscape(
                                id="cytoscape",
                                boxSelectionEnabled=True,
                                responsive=True,
                                elements=run_network_algo_cytoscape(
                                    ALGO_DATA["Network 1"]["data"]
                                ),
                                layout={"name": "preset", "padding": 10},
                                stylesheet=[
                                    {
                                        "selector": "core",
                                        "style": {"active-bg-opacity": 0},
                                    },
                                    {
                                        "selector": "node",
                                        "style": {
                                            "content": "data(id)",
                                            "background-color": "blue",
                                            "width": 15,
                                            "height": 15,
                                        },
                                    },
                                    {
                                        "selector": "edge",
                                        "style": {
                                            "curve-style": "bezier",
                                            "line-color": "yellow",
                                            "target-arrow-color": "yellow",
                                        },
                                    },
                                    {
                                        "selector": ":selected",
                                        "style": {
                                            "line-color": "#0056DA",
                                            "target-arrow-color": "#0056DA",
                                            "background-color": "#0056DA",
                                        },
                                    },
                                    {
                                        "selector": "node, edge",
                                        "style": {
                                            "transition-property": "opacity",
                                            "transition-duration": "250ms",
                                            "transition-timing-function": "ease-in-out",
                                        },
                                    },
                                    {
                                        "selector": ".leaflet-viewport",
                                        "style": {
                                            "opacity": 0.333,
                                            "transition-duration": "0ms",
                                        },
                                    },
                                    {
                                        "selector": ".transformer",
                                        "style": {
                                            "background-color": "red",
                                        },
                                    },
                                    {
                                        "selector": ".pole",
                                        "style": {
                                            "background-color": "purple",
                                        },
                                    },
                                    {
                                        "selector": ".connection-point",
                                        "style": {
                                            "background-color": "green",
                                            "shape": "square",
                                        },
                                    },
                                    {
                                        "selector": ".inner-node",
                                        "style": {
                                            "background-color": "yellow",
                                            "width": 8,
                                            "height": 8,
                                        },
                                    },
                                ],
                                contextmenu=[
                                    {
                                        "selector": "node, edge, core",
                                        "content": "Add Node",
                                        "id": "AN",
                                    },
                                    {
                                        "selector": "node",
                                        "content": "Select and Connect Nodes",
                                        "id": "SCN",
                                    },
                                    {
                                        "selector": "node",
                                        "content": "Remove Node",
                                        "id": "RMN",
                                    },
                                    {
                                        "selector": "edge",
                                        "content": "Remove Edge",
                                        "id": "RME",
                                    },
                                ],
                                style={
                                    "position": "relative",
                                    "width": "100%",
                                    "height": "750px",
                                },
                                leaflet=LEAFLET_TILE_META,
                            )
                        ],
                    ),
                ),
            ]
        ),
        html.Br(),
        dbc.Card(
            [
                dbc.CardBody(
                    children=dbc.Col(
                        children=dcc.Graph(
                            figure=px.line(
                                x=x_data_graph,
                                y=y_data_graph,
                                labels={
                                    "x": "X",
                                    "y": "Y",
                                },
                            )
                        )
                    )
                )
            ]
        ),
        html.Br(),
    ],
)


@app.callback(
    [Output("cytoscape", "elements"), Output("cytoscape", "leaflet")],
    [Input("cytoscape", "contextmenuData"), Input("dataset-dropdown", "value")],
    [State("cytoscape", "elements"), State("cytoscape", "tapNodeData")],
)
def handleCtxmenuReturn(contextmenuData, dataset, elements_cb, tapNodeData):
    # find input that fired the call back
    ctx = dash.callback_context
    component_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # callback for updating dataset in cytoscape graph
    if component_id == "dataset-dropdown":
        elements_cb = run_network_algo_cytoscape(ALGO_DATA[dataset]["data"])
        leaflet = ALGO_DATA[dataset]["leaflet"]
        return elements_cb, leaflet

    # if context menu data is empty prevent callback from firing
    if contextmenuData is None:
        raise PreventUpdate

    context_menu_id = contextmenuData["id"]

    # core event is not fired
    if context_menu_id != "AN":
        target_node_id = contextmenuData["target"]["data"]["id"]

    # add node
    if context_menu_id == "AN":
        target_lat = contextmenuData["coordinates"][0]
        target_lon = contextmenuData["coordinates"][1]
        letters = string.ascii_letters
        new_node = {
            "data": {
                "id": ("".join(random.choice(letters) for i in range(10))),
                "label": "New node",
                "lon": target_lon,
                "lat": target_lat,
            },
        }
        elements_cb.append(new_node)

    # select and connect nodes with an edge
    elif context_menu_id == "SCN":
        if tapNodeData is None:
            raise PreventUpdate

        selected_node_id = tapNodeData["id"]
        new_edge = {"data": {"source": selected_node_id, "target": target_node_id}}
        elements_cb.append(new_edge)

    # remove node
    elif context_menu_id == "RMN":
        for idx, ele in reversed(list(enumerate(elements_cb))):
            if "source" in ele["data"]:
                if target_node_id == ele["data"]["source"]:
                    elements_cb.pop(idx)
            else:
                if target_node_id == ele["data"]["id"]:
                    elements_cb.pop(idx)

    # remove edge
    elif context_menu_id == "RME":
        target_edge_id = target_node_id
        for idx, ele in reversed(list(enumerate(elements_cb))):
            if "source" in ele["data"]:
                if target_edge_id == ele["data"]["id"]:
                    elements_cb.pop(idx)
                    break
    else:
        raise PreventUpdate
    return elements_cb, dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True)
