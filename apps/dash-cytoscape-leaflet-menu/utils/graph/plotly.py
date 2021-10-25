"""Plotly Graphing Functions."""

import plotly.graph_objects as go

from utils.algorithm import make_nodes_from_dataframe, make_network
from utils.classes import (
    Node,
    Transformer,
    Pole,
    ConnectionPoint,
    InnerNode,
    Line,
)


def plotly_node_trace(
    nodes,
    name,
    color=Node.COLOR,
    size=None,
    legend_group=None,
    show_legend=None,
):
    """Return a node trace for a Plotly Map Figure."""
    if size is None:
        size = 12
    if legend_group is None:
        legend_group = ""
    if show_legend is None:
        show_legend = True

    lats = [node.lat for node in nodes]
    lons = [node.lon for node in nodes]
    texts = [node._id for node in nodes]

    return go.Scattermapbox(
        name=name,
        lat=lats,
        lon=lons,
        text=texts,
        mode="markers",
        marker=dict(size=size, color=color),
        legendgroup=legend_group,
        showlegend=show_legend,
    )


def plotly_line_trace(
    lines,
    name,
    color=Line.COLOR,
    line_width=None,
    legend_group=None,
    show_legend=None,
):
    """Return a line trace for a Plotly Map Figure."""
    if line_width is None:
        line_width = 2
    if legend_group is None:
        legend_group = ""
    if show_legend is None:
        show_legend = True

    lats = []
    lons = []
    for line in lines:
        lats.extend(line.get_lats())
        lons.extend(line.get_lons())

    return go.Scattermapbox(
        name=name,
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=line_width, color=color),
        hoverinfo="skip",
        legendgroup=legend_group,
        showlegend=show_legend,
    )


def run_network_algo_plotly(df):
    """Return a Plotly graph containing the nodes and lines from the algo."""
    transformers, poles, connection_points = make_nodes_from_dataframe(df)
    inner_nodes, network_lines, intersection_lines = make_network(transformers, poles)

    fig = go.Figure()
    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
        height=620,
        mapbox=dict(
            style="open-street-map",
            # for `data.csv` in the data/ folder
            # center=dict(lat=45.433, lon=-73.539),
            # zoom=17,
            # for `data2.csv` in the data/ folder
            # center=dict(lat=45.433, lon=-73.539),
            # zoom=15,
            # for `data3.csv` in the data/ folder
            center=dict(lat=45.436, lon=-73.529),
            zoom=14.2,
            bearing=0,
        ),
    )

    # add inner nodes and lines per subnetwork
    num_subnetworks = len(network_lines)
    for idx in range(num_subnetworks):
        current_network_lines = network_lines[idx]
        current_intersection_lines = intersection_lines[idx]
        current_inner_nodes = inner_nodes[idx]
        fig.add_trace(
            plotly_node_trace(
                nodes=current_inner_nodes,
                name=f"inner nodes",
                legend_group=f"subnetwork {idx}",
                color=InnerNode.COLOR,
                size=6,
            )
        )
        fig.add_trace(
            plotly_line_trace(
                lines=current_network_lines,
                name=f"network lines",
                legend_group=f"subnetwork {idx}",
                color=Line.COLOR,
            )
        )
        for line_idx, line in enumerate(current_intersection_lines):
            fig.add_trace(
                plotly_line_trace(
                    lines=[line],
                    name=f"intersection lines",
                    legend_group=f"subnetwork {idx}",
                    color=Line.COLOR,
                    show_legend=True if line_idx == 0 else False,
                )
            )

    # add main nodes
    fig.add_trace(
        plotly_node_trace(
            nodes=transformers,
            name="transformers",
            color=Transformer.COLOR,
        )
    )
    fig.add_trace(
        plotly_node_trace(
            nodes=poles,
            name="poles",
            color=Pole.COLOR,
        )
    )
    fig.add_trace(
        plotly_node_trace(
            nodes=connection_points,
            name="connection points",
            color=ConnectionPoint.COLOR,
        )
    )

    return fig
