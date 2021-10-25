"""Helper functions."""


def flatten_list(list_of_sublists):
    flat_list = [item for sublist in list_of_sublists for item in sublist]
    return flat_list
