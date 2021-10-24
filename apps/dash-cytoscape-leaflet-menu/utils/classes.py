"""A collection of useful classes for creating the network."""

from math import sin, cos, sqrt, atan2, radians, degrees
import numpy as np

from utils.constants import EARTH_RADIUS_KM, MIN_INNER_NODE_DISTANCE_KM


class Node:
    """A Node object."""

    COLOR = "yellow"

    def __init__(self, _id=None, _type=None, lat=None, lon=None, used=False):
        self._id = _id
        self._type = _type
        self.lat = lat
        self.lon = lon
        self.used = used

    def find_closest_node(self, test_nodes):
        """Find the closest node from the given list of nodes."""
        min_dist = np.inf
        closest_node = None
        for node in test_nodes:
            line = Line(self, node)
            if node != self and line.distance() < min_dist:
                min_dist = line.distance()
                closest_node = node
        return closest_node

    def get_group_idx(self, test_groups):
        """Find the index of the group containing this node."""
        correct_idx = None
        for group_idx, group in enumerate(test_groups):
            if self in group:
                correct_idx = group_idx
        return correct_idx

    def __repr__(self):
        if self._type:
            _type_split = self._type.split()
            _type_abbr = "".join(w[0] for w in _type_split).upper()
        return f'{_type_abbr}("{self._id}")'


class Transformer(Node):
    """A Node of type `transformer`."""

    COLOR = "red"

    def __init__(self, _id, lat, lon, leaf_nodes=None):
        self._id = _id
        self._type = "transformer"
        self.lat = lat
        self.lon = lon
        self.leaf_nodes = leaf_nodes
        self.used = True


class Pole(Node):
    """A Node of type `pole`."""

    COLOR = "purple"

    def __init__(self, _id, lat, lon):
        self._id = _id
        self._type = "pole"
        self.lat = lat
        self.lon = lon
        self.used = True


class ConnectionPoint(Node):
    """A Node of type `connection point`."""

    COLOR = "green"

    def __init__(self, _id, lat, lon, root_node_id=None):
        self._id = _id
        self._type = "connection point"
        self.lat = lat
        self.lon = lon
        self.root_node_id = root_node_id
        self.used = True


class InnerNode(Node):
    """A Node of type `inner node`."""

    COLOR = "yellow"

    def __init__(self, _id, lat, lon):
        self._id = _id
        self._type = "inner node"
        self.lat = lat
        self.lon = lon
        self.used = False


class Line:
    """Defines a line segment from two Nodes."""

    COLOR = "yellow"

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return f"Line({self.a}, {self.b})"

    def slope(self):
        """Return the slope of a line projected onto a globe."""
        rise = self.b.lat - self.a.lat
        run = self.b.lon - self.a.lon
        return 2 * rise / run

    def get_lats(self):
        """Return the latitude values from the line."""
        return [self.a.lat, self.b.lat]

    def get_lons(self):
        """Return the longitude values from the line."""
        return [self.a.lon, self.b.lon]

    def contains_node(self, node):
        """Determine whether a specific Node is in this line."""
        if self.a == node or self.b == node:
            return True
        return False

    def orthogonal_vector(self):
        """Return an vector that is 90deg this line."""
        point_two_rotated = [
            self.a.lat + 0.5 * (self.a.lon - self.b.lon),
            self.a.lon + (self.b.lat - self.a.lat),
        ]
        rise = point_two_rotated[0] - self.a.lat
        run = point_two_rotated[1] - self.a.lon
        return [rise, run]

    def distance(self):
        """Compute the kilometre distance of the line on a map."""
        lat1 = radians(self.a.lat)
        lon1 = radians(self.a.lon)
        lat2 = radians(self.b.lat)
        lon2 = radians(self.b.lon)

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return EARTH_RADIUS_KM * c

    def intermediate_coordinates(self, fraction):
        """Compute the coordinates at a given fraction along this line."""
        lat1 = radians(self.a.lat)
        lon1 = radians(self.a.lon)
        lat2 = radians(self.b.lat)
        lon2 = radians(self.b.lon)

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        A = sin((1 - fraction) * c) / sin(c)
        B = sin(fraction * c) / sin(c)
        x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
        y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
        z = A * sin(lat1) + B * sin(lat2)

        lat3 = atan2(z, sqrt(x * x + y * y))
        lon3 = atan2(y, x)

        lat = degrees(lat3)
        lon = degrees(lon3)

        return lat, lon

    def intermediate_nodes(self):
        """Recursively create equidistant nodes along this line."""
        distance = self.distance()
        if distance < MIN_INNER_NODE_DISTANCE_KM:
            return []
        else:
            lat, lon = self.intermediate_coordinates(0.5)
            mid_node = InnerNode(
                _id=f"({self.a._id}-{self.b._id})",
                lat=lat,
                lon=lon,
            )
            left_line = Line(self.a, mid_node)
            right_line = Line(mid_node, self.b)
            left_inter_nodes = left_line.intermediate_nodes()
            right_inter_nodes = right_line.intermediate_nodes()
            return left_inter_nodes + [mid_node] + right_inter_nodes
