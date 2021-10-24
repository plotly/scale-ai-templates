"""Cytoscape Graphing Functions."""

from utils.algorithm import make_nodes_from_dataframe, make_network
from utils.helpers import flatten_list


LINE_ID_DELIM = "-"


def cyto_node_element(node_id, lat, lon, classes=None, label_id=None):
    """An node for a Cytoscape Graph."""
    element = {
        "data": {
            "id": node_id,
            "label": label_id,
            "lat": lat,
            "lon": lon,
        },
        "classes": classes,
    }
    if label_id is not None:
        element["data"]["label"] = label_id
    return element


def cyto_line_element(line_id, source_id, target_id):
    """An line for a Cytoscape Graph."""
    return {
        "data": {
            "id": line_id,
            "source": source_id,
            "target": target_id,
        }
    }


def run_network_algo_cytoscape(df):
    """
    Return the elements for a Dash Cytoscape Component.

    Args:
        df (pandas.DataFrame): the input data frame.
    """
    transformers, poles, connection_points = make_nodes_from_dataframe(df)
    inner_nodes, network_lines, intersection_lines = make_network(transformers, poles)

    inner_nodes = flatten_list(inner_nodes)
    network_lines = flatten_list(network_lines)
    intersection_lines = flatten_list(intersection_lines)

    elements = []
    for node in transformers + poles + inner_nodes + connection_points:
        node_classes = node._type.replace(" ", "-")
        elements.append(
            cyto_node_element(
                node_id=node._id,
                lat=node.lat,
                lon=node.lon,
                classes=node_classes,
                label_id=node._id,
            )
        )
    for line in network_lines + intersection_lines:
        line_id = f"{LINE_ID_DELIM}".join([line.a._id, line.b._id])
        elements.append(cyto_line_element(line_id, line.a._id, line.b._id))

    return elements
