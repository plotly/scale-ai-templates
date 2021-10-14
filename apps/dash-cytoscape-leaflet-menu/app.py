import string
import random

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_cytoscape as cyto
from dash.exceptions import PreventUpdate
import plotly.express as px
import numpy as np

# Graph data
x_data_graph = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-12-31"))
y_data_graph = np.random.randint(10, size=len(x_data_graph))

# App description
description = dcc.Markdown(
    """ 
This Dash application contains **2 brand new features** for the [Dash Cytoscape](https://dash.plotly.com/cytoscape) library.

Features:

1. Leaflet map integration for the [Dash Cytoscape](https://dash.plotly.com/cytoscape) library
2. Context menu for the [Dash Cytoscape](https://dash.plotly.com/cytoscape) library (try right clicking on the graph)

These features were created in collaboration with [Plotly](https://plotly.com/), [SCALEAI](https://www.scaleai.ca/), and developed by [Zyphr](https://www.zyphr.ca/).
"""
)

# Cytoscape elements
elements = [
    {
        "data": {
            "id": "a",
            "label": "Trois-Rivières",
            "lat": 46.349998,
            "lon": -72.550003,
        }
    },
    {"data": {"id": "b", "label": "Montreal", "lat": 45.508888, "lon": -73.561668}},
    {"data": {"id": "c", "label": "Quebec City", "lat": 46.829853, "lon": -71.254028}},
    {"data": {"id": "d", "label": "Gaspé", "lat": 48.833332, "lon": -64.483330}},
    {"data": {"id": "ab", "source": "a", "target": "b"}},
    {"data": {"id": "ac", "source": "a", "target": "c"}},
    {"data": {"id": "ad", "source": "a", "target": "d"}},
]

leaflet = {
    "tileUrl": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}{r}.png",
    "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    "maxZoom": 30,
}

# Cytoscape Data
algo_data = {
    "Data Set 1": {
        "data": elements,
        "leaflet": {
            **leaflet,
            "view": [46.829853, -71.254028, 5],
        },
    },
    "Data Set 2": {
        "data": [
            {
                "data": {
                    "id": "e",
                    "label": "Laval",
                    "lat": 45.612499,
                    "lon": -73.707092,
                }
            },
            {
                "data": {
                    "id": "f",
                    "label": "Gatineau",
                    "lat": 45.476543,
                    "lon": -75.701271,
                }
            },
            {"data": {"id": "ef", "source": "e", "target": "f"}},
        ],
        "leaflet": {
            **leaflet,
            "view": [45.612499, -73.707092, 5],
        },
    },
    "Data Set 3": {
        "data": [
            *elements,
            {
                "data": {
                    "id": "f",
                    "label": "Mont-Tremblant",
                    "lat": 46.116669,
                    "lon": -74.599998,
                }
            },
            {"data": {"id": "af", "source": "a", "target": "f"}},
        ],
        "leaflet": {
            **leaflet,
            "view": [46.349998, -72.550003, 5],
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
                                                    value="Data Set 1",
                                                    options=[
                                                        {"label": i, "value": i}
                                                        for i in [
                                                            "Data Set 1",
                                                            "Data Set 2",
                                                            "Data Set 3",
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
                        style={"width": "100%"},
                        children=[
                            cyto.Cytoscape(
                                id="cytoscape",
                                boxSelectionEnabled=True,
                                responsive=True,
                                elements=algo_data["Data Set 1"]["data"],
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
                                    "height": "500px",
                                },
                                leaflet=leaflet
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
def handleCtxmenuReturn(contextmenuData, dataset, elements, tapNodeData):
    ctx = dash.callback_context
    component_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # callback for updating dataset in cytoscape graph
    if component_id == "dataset-dropdown":
        elements = algo_data[dataset]["data"]
        leaflet = algo_data[dataset]["leaflet"]
        return elements, leaflet

    if contextmenuData is None:
        raise PreventUpdate

    # helpful print statements for looking at cytoscape metadata
    # print("contextMenu data >>>\n", contextmenuData)
    # print("elements data >>>\n", elements)

    context_menu_id = contextmenuData["id"]

    # core event is not fired
    if context_menu_id != "AN":
        target_node_id = contextmenuData["target"]["data"]["id"]

    # Add node
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
        elements.append(new_node)

    # Select and connect nodes with an edge
    elif context_menu_id == "SCN":
        if tapNodeData is None:
            raise PreventUpdate

        selected_node_id = tapNodeData["id"]
        new_edge = {"data": {"source": selected_node_id, "target": target_node_id}}
        elements.append(new_edge)

    # Remove node
    elif context_menu_id == "RMN":
        for idx, ele in reversed(list(enumerate(elements))):
            if "source" in ele["data"]:
                if target_node_id == ele["data"]["source"]:
                    elements.pop(idx)
            else:
                if target_node_id == ele["data"]["id"]:
                    elements.pop(idx)

    # Remove edge
    elif context_menu_id == "RME":
        target_edge_id = target_node_id
        for idx, ele in reversed(list(enumerate(elements))):
            if "source" in ele["data"]:
                if target_edge_id == ele["data"]["id"]:
                    elements.pop(idx)
                    break

    # # Collapse node on node 'Node A'
    # elif context_menu_id == "CNJ" and target_node_id == "cp11":
    #     target_node_id = ["cp11_one", "cp11_two"]

    #     for idx, ele in reversed(list(enumerate(elements))):
    #         if "source" in ele["data"]:
    #             if ele["data"]["source"] in target_node_id:
    #                 elements.pop(idx)
    #         else:
    #             if ele["data"]["id"] in target_node_id:
    #                 elements.pop(idx)

    # # Expand node on node 'cp11'
    # elif context_menu_id == "ENJ" and target_node_id == "cp11":
    #     expanded_nodes = [
    #         {
    #             "data": {
    #                 "id": "cp11_one",
    #                 "label": "Cp11 Node One",
    #                 "lat": 45.434044907205525,
    #                 "lon": -73.53985902985069,
    #             },
    #         },
    #         {
    #             "data": {
    #                 "id": "cp11_two",
    #                 "label": "Cp11 Node Two",
    #                 "lat": 45.434008649176754,
    #                 "lon": -73.53938293248736,
    #             },
    #         },
    #     ]

    #     expanded_edges = [
    #         {"data": {"source": "cp11", "target": "cp11_one"}},
    #         {"data": {"source": "cp11", "target": "cp11_two"}},
    #     ]

    #     elements = elements + expanded_nodes + expanded_edges
    else:
        raise PreventUpdate
    return elements, dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True)
