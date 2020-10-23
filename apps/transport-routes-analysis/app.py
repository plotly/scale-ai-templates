from datetime import datetime as dt
from datetime import date, timedelta
import os
from statistics import mean
from random import randint, shuffle

import pandas as pd
import plotly.graph_objs as go
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output

def NamedGroup(children, label, **kwargs):
    return dbc.FormGroup(
        [
            dbc.Label(label),
            children
        ],
        **kwargs
    )

mapbox_access_token = os.getenv("MAPBOX_ACCESS_TOKEN")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.title = "Transport Routes Analysis"

server = app.server

df = pd.read_csv("./data/quebec_cities.csv")

map_layout = go.Layout(
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        center=go.layout.mapbox.Center(lat=mean(df["lat"] + 5), lon=mean(df["lng"])),
        style="dark",
        zoom=3,
    ),
    height=375,
    margin=dict(l=15, r=15, t=15, b=15),
    paper_bgcolor="#303030",
    font_color="white"
)

otif_layout = go.Layout(
    height=400,
    xaxis={
        "range": [-5, 5],
        "showgrid": False,
        "zeroline": False,
        "title": "On Time",
        "dtick": 1,
    },
    yaxis={
        "range": [0, 120],
        "showgrid": False,
        "zeroline": False,
        "title": "In Full",
        "dtick": 10,
    },
    shapes=[
        {
            "type": "line",
            "x0": 0,
            "y0": 0,
            "x1": 0,
            "y1": 120,
            "line": {"dash": "dash", "width": 1, "color": "white"},
        },
        {
            "type": "line",
            "x0": -5,
            "y0": 100,
            "x1": 5,
            "y1": 100,
            "line": {"dash": "dash", "width": 1, "color": "white"},
        },
    ],
    paper_bgcolor="#303030",
    plot_bgcolor="#303030",
    font_color="white"
)

cities = go.Scattermapbox(
    lat=df["lat"].values,
    lon=df["lng"].values,
    mode="markers",
    marker=go.scattermapbox.Marker(size=5),
    text=df["city"],
    showlegend=False,
)

carriers = {
    "A": "#4c78a8",
    "B": "#f58518",
    "C": "#e45756",
    "D": "#72b7b2",
}
routes_per_carrier = 10

dates = [date(2018, randint(10, 12), randint(1, 30)) for i in range(30)]
order_dates = [date(2018, randint(8, 9), randint(1, 30)) for i in range(30)]

routes = []


def get_arrow_char(lats, lons):
    if lats[0] > lats[1]:
        if lons[0] > lons[1]:
            return "↙"
        else:
            return "↘"
    else:
        if lons[0] > lons[1]:
            return "↖"
        else:
            return "↗"


for carrier in carriers:
    for j in range(routes_per_carrier):
        num_cities = randint(2, 10)
        route_cities = df.sample(num_cities)
        num_ordered = randint(20, 80)
        infull = randint(80, 105)
        ontime = randint(-3, 3)
        date = dates[randint(0, len(dates) - 1)]

        routes.append(
            {
                "carrier": carrier,
                "lats": route_cities["lat"].values,
                "lons": route_cities["lng"].values,
                "cities": route_cities["city"].values,
                "date": date,
                "actualdate": date + timedelta(days=ontime),
                "ontime": ontime,
                "infull": infull,
                "distr": list(route_cities["city"])[0].upper(),
                "pointofsale": "POS{}".format(str(randint(100, 200))),
                "city": list(route_cities["city"])[-1],
                "orderdate": order_dates[randint(0, len(dates) - 1)],
                "ordernr": "A{}".format(str(randint(900000, 999999))),
                "ordertype": ["Standard", "Offer"][randint(0, 1)],
                "item": "H{}".format(str(randint(1000000, 9999999))),
                "qtyordered": num_ordered,
                "qtyserved": int(num_ordered * infull / 100),
            }
        )

routes = pd.DataFrame(routes)

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("Transport Routes Analysis"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    md=4,
                    children=[
                        dbc.Card(
                            children=[
                                dbc.CardHeader("Coherence of Travel Routes"),
                                dbc.CardBody(
                                    [
                                        NamedGroup(
                                            dcc.DatePickerRange(
                                                id="travel-routes-date",
                                                min_date_allowed=dt(2018, 9, 1),
                                                max_date_allowed=dt(2018, 12, 31),
                                                initial_visible_month=dt(2018, 12, 1),
                                                start_date=dt(2018, 11, 1),
                                                end_date=dt(2018, 12, 15),
                                            ),
                                            label="Date",
                                        ),
                                        NamedGroup(
                                            dcc.Dropdown(
                                                id="travel-routes-carrier",
                                                options=[
                                                    {"label": carrier, "value": carrier}
                                                    for carrier in carriers.keys()
                                                ],
                                                value=[carrier for carrier in carriers.keys()],
                                                multi=True,
                                            ),
                                            label="Carrier",
                                        ),
                                        NamedGroup(
                                            dcc.RangeSlider(
                                                id="travel-routes-infull",
                                                min=75,
                                                max=125,
                                                value=[90, 105],
                                                marks={i: str(i) for i in range(75, 125, 10)},
                                            ),
                                            label="In Full",
                                        ),
                                    ]
                                )
                            ]
                        ),
                        html.Br(),
                        dbc.Card(
                            children=[
                                dbc.CardHeader(
                                    "OTIF - On Time In Full (Transportation)"
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="otif-graph",
                                        figure=go.Figure(data=[{"type": "scatter"}]),
                                    ),
                                )
                            ]
                        ),
                    ],
                ),
                html.Br(),
                dbc.Col(
                    md=8,
                    children=[
                        dbc.Card(
                            children=[
                                dbc.CardHeader(
                                    "Routes Visualization for Marked OTIF Spots"
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="map",
                                        figure=go.Figure(data=[cities], layout=map_layout),
                                    ),
                                )
                            ]
                        ),
                        html.Br(),
                        dbc.Card(
                            children=[
                                dbc.CardHeader("Routes Optimization"),
                                dbc.CardBody(
                                    html.Div(
                                        style={
                                            "maxHeight": "400px",
                                            "overflow-y": "scroll",
                                        },
                                        children=html.Div(
                                            html.Table(
                                                id="route-optimized",
                                                style={
                                                    "font-size": "10pt",
                                                    "width": "100%",
                                                },
                                            )
                                        ),
                                    ),
                                )
                            ]
                        ),
                    ],
                ),
            ]
        ),
        html.Br(),
        dbc.Card(
            children=[
                dbc.CardHeader("Details on Demand"),
                dbc.CardBody(
                    html.Div(
                        style={"maxHeight": "400px", "overflow-y": "scroll"},
                        children=html.Table(
                            id="details-table", style={"font-size": "10pt", "width": "100%"}
                        ),
                    )
                )
            ]
        ),
    ],
)


@app.callback(
    [
        Output("map", "figure"),
        Output("otif-graph", "figure"),
        Output("details-table", "children"),
        Output("route-optimized", "children"),
    ],
    [
        Input("travel-routes-date", "start_date"),
        Input("travel-routes-date", "end_date"),
        Input("travel-routes-carrier", "value"),
        Input("travel-routes-infull", "value"),
    ],
)
def update_map(start_date, end_date, selected_carriers, infull):
    map_data = [cities]
    otif_data = []

    start_date = dt.strptime(str(start_date)[:10], "%Y-%m-%d").date()
    end_date = dt.strptime(str(end_date)[:10], "%Y-%m-%d").date()

    table_rows = [
        html.Tr(
            [
                html.Td("Distribution center"),
                html.Td("Order date"),
                html.Td("Scheduled delivery"),
                html.Td("Actual delivery"),
                html.Td("Point of Sale"),
                html.Td("City"),
                html.Td("Order Nr"),
                html.Td("Order Type"),
                html.Td("Item"),
                html.Td("Qty served"),
                html.Td("Qty delivered"),
                html.Td("Carrier"),
                html.Td("Region"),
            ]
        )
    ]

    optimized_routes = [
        html.Tr(
            [
                html.Td("Order Nr"),
                html.Td("Current Route"),
                html.Td("Suggested Route"),
                html.Td("Distance Saved (km)"),
            ]
        )
    ]

    for carrier in selected_carriers:
        selected_routes = routes[
            (routes.carrier == carrier)
            & (routes.date > start_date)
            & (routes.date < end_date)
            & (routes.infull > infull[0])
            & (routes.infull < infull[1])
        ]

        carrier_otif_data = {
            "x": [],
            "y": [],
            "type": "scatter",
            "mode": "markers",
            "name": carrier,
            "marker": {"color": carriers[carrier]},
        }

        for selected_route in selected_routes.to_dict("records"):

            carrier_otif_data["x"].append(selected_route["ontime"]),
            carrier_otif_data["y"].append(selected_route["infull"])

            table_rows.append(
                html.Tr(
                    [
                        html.Td(selected_route["distr"]),
                        html.Td(selected_route["orderdate"]),
                        html.Td(selected_route["date"]),
                        html.Td(selected_route["actualdate"]),
                        html.Td(selected_route["pointofsale"]),
                        html.Td(selected_route["city"]),
                        html.Td(selected_route["ordernr"]),
                        html.Td(selected_route["ordertype"]),
                        html.Td(selected_route["item"]),
                        html.Td(selected_route["qtyordered"]),
                        html.Td(selected_route["qtyserved"]),
                        html.Td(carrier),
                        html.Td("Quebec"),
                    ]
                )
            )

            new_route = [city for city in selected_route["cities"]]
            shuffle(new_route)

            fixed_route = False

            for i in range(len(new_route)):
                if new_route[i] == selected_route["cities"][i]:
                    fixed_route = True
                    break

            if fixed_route:
                optimized_routes.append(
                    html.Tr(
                        [
                            html.Td(selected_route["ordernr"]),
                            html.Td(", ".join(selected_route["cities"])),
                            html.Td(", ".join(new_route)),
                            html.Td(randint(100, 150)),
                        ]
                    )
                )

            route_trace = dict(
                lat=selected_route["lats"],
                lon=selected_route["lons"],
                mode="lines",
                name=selected_route["carrier"],
                line={"color": carriers[carrier], "width": 1},
                legendgroup=selected_route["carrier"],
                showlegend=False,
            )

            for i in range(len(selected_route["lats"]) - 1):
                lats = [selected_route["lats"][i], selected_route["lats"][i + 1]]

                lons = [selected_route["lons"][i], selected_route["lons"][i + 1]]

                halfway = [mean(lats), mean(lons)]

                map_data.append(
                    go.Scattermapbox(
                        mode="text",
                        lat=[halfway[0]],
                        lon=[halfway[1]],
                        text=get_arrow_char(lats, lons),
                        textfont={"color": carriers[carrier], "size": 30},
                        showlegend=False,
                    )
                )

            if (
                selected_routes.to_dict("records")[0]["ordernr"]
                == selected_route["ordernr"]
            ):
                route_trace.update(showlegend=True)
                map_data.append(
                    go.Scattermapbox(
                        mode="text",
                        lat=[selected_route["lats"][0]],
                        lon=[selected_route["lons"][0]],
                        text=".",
                        showlegend=False,
                    )
                )

            route_trace = go.Scattermapbox(route_trace)
            map_data.append(route_trace)

        if len(carrier_otif_data["x"]) > 0:
            otif_data.append(carrier_otif_data)

    return (
        {"data": map_data, "layout": map_layout},
        {"data": otif_data, "layout": otif_layout},
        table_rows,
        optimized_routes,
    )


if __name__ == "__main__":
    app.run_server(debug=True)
