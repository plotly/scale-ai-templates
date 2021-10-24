"""Functions that create lines to be plotted on the map."""

from utils.classes import (
    Transformer,
    Pole,
    ConnectionPoint,
    InnerNode,
    Line,
)
from utils.constants import MAX_INNER_NODE_DISTANCE_KM


def make_nodes_from_dataframe(df):
    """
    Generate transformers, poles and connection points from a dataframe.
    """
    transformers = []
    poles = []
    connection_points = []
    connection_points_by_transformer = {}
    for row in df.iterrows():
        _id = row[0]
        if _id.startswith("t"):
            node = Transformer(
                _id=_id,
                lat=row[1]["lat"],
                lon=row[1]["lon"],
            )
            transformers.append(node)
        elif _id.startswith("p"):
            node = Pole(
                _id=_id,
                lat=row[1]["lat"],
                lon=row[1]["lon"],
            )
            poles.append(node)
        elif _id.startswith("cp"):
            node = ConnectionPoint(
                _id=_id,
                lat=row[1]["lat"],
                lon=row[1]["lon"],
                root_node_id=row[1]["root node"],
            )
            connection_points.append(node)

            # group connection points by transformer id
            if node.root_node_id not in connection_points_by_transformer.keys():
                connection_points_by_transformer[node.root_node_id] = []
            connection_points_by_transformer[node.root_node_id].append(node)

    # set leaf nodes for transformers
    for node in transformers:
        node.leaf_nodes = connection_points_by_transformer[node._id]

    return transformers, poles, connection_points


def make_network(transformers, poles):
    """
    Generate inner nodes, network lines and intersection lines
    to connect transformers, poles and connection points.
    """

    def get_start_node(nodes):
        # select node with min latitude as starting node
        start_node = None
        if nodes:
            lat_list = [node.lat for node in nodes]
            start_idx = lat_list.index(min(lat_list))
            start_node = nodes[start_idx]
        return start_node

    # traverse poles by proximity, create inner nodes along traversed path,
    # and group trial nodes (poles and inner nodes) into subnetworks
    start_pole = get_start_node(poles)
    traversed_poles = [start_pole]
    remaining_poles = [node for node in poles if node not in traversed_poles]
    trial_nodes = [start_pole]
    grouped_trial_nodes = []
    while start_pole:
        # get next potential pole/line for subnetwork
        next_pole = start_pole.find_closest_node(remaining_poles)
        next_line = Line(start_pole, next_pole)
        # assess whether to add pole and its inner nodes to subnetwork,
        # or mark the subnetwork as complete and start new subnetwork
        if next_pole and next_line.distance() < MAX_INNER_NODE_DISTANCE_KM:
            trial_nodes.extend(next_line.intermediate_nodes())
            trial_nodes.append(next_pole)
            start_pole = next_pole
        else:
            grouped_trial_nodes.append(trial_nodes)
            start_pole = get_start_node(remaining_poles)
            trial_nodes = [start_pole]
        # prepare for next iteration
        if start_pole:
            traversed_poles.append(start_pole)
            remaining_poles.remove(start_pole)

    # create intersection lines from leaf nodes (transformers and connection
    # points) to trial nodes (inner nodes and poles)
    grouped_intersection_lines = [[] for _ in range(len(grouped_trial_nodes))]
    for transformer in transformers:
        closest_pole = transformer.find_closest_node(poles)
        group_idx = closest_pole.get_group_idx(grouped_trial_nodes)
        leaf_nodes = [transformer] + transformer.leaf_nodes
        trial_nodes = grouped_trial_nodes[group_idx]
        for leaf_node in leaf_nodes:
            # create a line between leaf node and closest trial node
            closest_node = leaf_node.find_closest_node(trial_nodes)
            intersection_line = Line(leaf_node, closest_node)
            grouped_intersection_lines[group_idx].append(intersection_line)
            # mark trial node as used to be considered valid for network lines
            closest_node.used = True

    # create network lines between valid inner nodes and poles
    inner_node_count = 0
    grouped_inner_nodes = []
    grouped_inner_lines = []
    for trial_nodes in grouped_trial_nodes:
        # start node would be a pole so inner node checks not necessary
        start_node = trial_nodes[0]
        inner_nodes = []
        inner_lines = []
        for next_node in trial_nodes[1:]:
            # add inner nodes that are used in the network, update inner node id
            if next_node.used and type(next_node) == InnerNode:
                next_node._id = f"in{inner_node_count}"
                inner_nodes.append(next_node)
                inner_node_count += 1
            # create a line towards any node that is used in the network
            if next_node.used:
                next_line = Line(start_node, next_node)
                inner_lines.append(next_line)
                start_node = next_node
        grouped_inner_nodes.append(inner_nodes)
        grouped_inner_lines.append(inner_lines)

    return grouped_inner_nodes, grouped_inner_lines, grouped_intersection_lines
