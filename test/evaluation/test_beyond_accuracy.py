import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from ebrec.evaluation.beyond_accuracy import (
    IntralistDiversity,
    Distribution,
    Serendipity,
    Novelty,
    Coverage,
)

lookup_dict = {
    "101": {"doc_vec": np.array([1, 0, 0]), "v": 1, "sv": [1], "pop_sc": 0.50},
    "102": {"doc_vec": np.array([0, 1, 0]), "v": 2, "sv": [1], "pop_sc": 0.25},
    "103": {"doc_vec": np.array([1, 1, 1]), "v": 3, "sv": [1], "pop_sc": 0.75},
    "104": {"doc_vec": np.array([1, 1, 1]), "v": 4, "sv": [1], "pop_sc": 0.50},
    "105": {"doc_vec": np.array([-1, 0, 0]), "v": 5, "sv": [1], "pop_sc": 0.94},
    "106": {"doc_vec": np.array([-1, 0, 0]), "v": 6, "sv": [1, 2], "pop_sc": 0.95},
    "107": {"doc_vec": np.array([-1, 0, 0]), "v": 7, "sv": [1, 2], "pop_sc": 0.96},
    "108": {"doc_vec": np.array([0, 0, 1]), "v": 8, "sv": [1, 2], "pop_sc": 0.50},
    "400": {"doc_vec": np.array([0, 0, 1]), "v": 9, "sv": [4], "pop_sc": 0.20},
    "401": {"doc_vec": np.array([0, 0, 1]), "v": 9, "sv": [4, 5], "pop_sc": 0.20},
}

# 404 is not excepted, however, setup supports it:
R = np.array(
    [
        ["101", "102", "400"],
        ["101", "103", "400"],
        ["101", "102", "103"],
        ["101", "104", "400"],
        ["101", "106", "404"],
        ["404", "404", "404"],
    ]
)

C = ["1", "2", "101", "102", "103", "104", "105", "106", "107", "108", "400", "401"]

click_histories = [
    np.array([["101", "102"]]),
    np.array([["105", "106", "400"]]),
    np.array([["102", "103", "104"]]),
    np.array([["101", "400"]]),
    np.array([["400"]]),
    np.array([["400"]]),
]
pairwise_distance_function = cosine_distances

# TODO: add the test
