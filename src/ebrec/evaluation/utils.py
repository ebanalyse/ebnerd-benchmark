from collections import Counter
from typing import Iterable
import numpy as np


def convert_to_binary(y_pred: np.ndarray, threshold: float):
    y_pred = np.asarray(y_pred)
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0
    return y_pred


def is_iterable_nested_dtype(iterable: Iterable[any], dtypes) -> bool:
    """
    Check whether iterable is a nested with dtype,
    note, we assume all types in iterable are the the same.
    Check all cases: any(isinstance(i, dtypes) for i in a)

    Args:
        iterable (Iterable[Any]): iterable (list, array, tuple) of any type of data
        dtypes (Tuple): tuple of possible dtypes, e.g. dtypes = (list, np.ndarray)
    Returns:
        bool: boolean whether it is true or false

    Examples:
        >>> is_iterable_nested_dtype([1, 2, 3], list)
            False
        >>> is_iterable_nested_dtype([1, 2, 3], (list, int))
            True
        >>> is_iterable_nested_dtype([[1], [2], [3]], list)
            True
    """
    return isinstance(iterable[0], dtypes)


def compute_combinations(n: int, r: int) -> int:
    """Compute Combinations where order does not matter (without replacement)

    Source: https://www.statskingdom.com/combinations-calculator.html
    Args:
        n (int): number of items
        r (int): number of items being chosen at a time
    Returns:
        int: number of possible combinations

    Formula:
    * nCr = n! / ( (n - r)! * r! )

    Assume the following:
    * we sample without replacement of items
    * order of the outcomes does NOT matter
    """
    return int(
        (np.math.factorial(n)) / (np.math.factorial(n - r) * np.math.factorial(r))
    )


def scale_range(
    m: np.ndarray,
    r_min: float = None,
    r_max: float = None,
    t_min: float = 0,
    t_max: float = 1.0,
) -> None:
    """Scale an array between a range
    Source: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range

    m -> ((m-r_min)/(r_max-r_min)) * (t_max-t_min) + t_min

    Args:
        m ∈ [r_min,r_max] denote your measurements to be scaled
        r_min denote the minimum of the range of your measurement
        r_max denote the maximum of the range of your measurement
        t_min denote the minimum of the range of your desired target scaling
        t_max denote the maximum of the range of your desired target scaling
    """
    if not r_min:
        r_min = np.min(m)
    if not r_max:
        r_max = np.max(m)
    return ((m - r_min) / (r_max - r_min)) * (t_max - t_min) + t_min


# utils for
def compute_item_popularity_scores(R: Iterable[np.ndarray]) -> dict[str, float]:
    """Compute popularity scores for items based on their occurrence in user interactions.

    This function calculates the popularity score of each item as the fraction of users who have interacted with that item.
    The popularity score, p_i, for an item is defined as the number of users who have interacted with the item divided by the
    total number of users.

    Formula:
        p_i = | {u ∈ U}, r_ui != Ø | / |U|

    where p_i is the popularity score of an item, U is the total number of users, and r_ui is the interaction of user u with item i (non-zero
    interaction implies the user has seen the item).

    Note:
        Each entry can only have the same item ones. TODO - ADD THE TEXT DONE HERE.

    Args:
        R (Iterable[np.ndarray]): An iterable of numpy arrays, where each array represents the items interacted with by a single user.
            Each element in the array should be a string identifier for an item.

    Returns:
        dict[str, float]: A dictionary where keys are item identifiers and values are their corresponding popularity scores (as floats).

    Examples:
    >>> R = [
            np.array(["item1", "item2", "item3"]),
            np.array(["item1", "item3"]),
            np.array(["item1", "item4"]),
        ]
    >>> print(popularity_scores(R))
        {'item1': 1.0, 'item2': 0.3333333333333333, 'item3': 0.6666666666666666, 'item4': 0.3333333333333333}
    """
    U = len(R)
    R_flatten = np.concatenate(R)
    item_counts = Counter(R_flatten)
    return {item: (r_ui / U) for item, r_ui in item_counts.items()}


def compute_normalized_distribution(
    R: np.ndarray[str],
    weights: np.ndarray[float] = None,
    distribution: dict[str, float] = None,
) -> dict[str, float]:
    """
    Compute a normalized weigted distribution for a list of items that each can have a single representation assigned.

    Args:
        a (np.ndarray[str]): an array of items representation.
        weights (np.ndarray[float], optional): weights to assign each element in a. Defaults to None.
            * Following yields: len(weights) == len(a)
        distribution (Dict[str, float], optional): dictionary to assign the distribution values, if None it will be generated as {}. Defaults to None.
            * Use case; if you want to add distribution values to existing, one can input it.

    Returns:
        Dict[str, float]: dictionary with normalized distribution values

    Examples:
        >>> a = np.array(["a", "b", "c", "c"])
        >>> compute_normalized_distribution(a)
            {'a': 0.25, 'b': 0.25, 'c': 0.5}
    """
    n_elements = len(R)

    distr = distribution if distribution is not None else {}
    weights = weights if weights is not None else np.ones(n_elements) / n_elements
    for item, weight in zip(R, weights):
        distr[item] = weight + distr.get(item, 0.0)
    return distr


def get_keys_in_dict(id_list: any, dictionary: dict) -> list[any]:
    """
    Returns a list of IDs from id_list that are keys in the dictionary.
    Args:
        id_list (List[Any]): List of IDs to check against the dictionary.
        dictionary (Dict[Any, Any]): Dictionary where keys are checked against the IDs.

    Returns:
        List[Any]: List of IDs that are also keys in the dictionary.

    Examples:
        >>> get_keys_in_dict(['a', 'b', 'c'], {'a': 1, 'c': 3, 'd': 4})
            ['a', 'c']
    """
    return [id_ for id_ in id_list if id_ in dictionary]


def check_key_in_all_nested_dicts(dictionary: dict, key: str) -> None:
    """
    Checks if the given key is present in all nested dictionaries within the main dictionary.
    Raises a ValueError if the key is not found in any of the nested dictionaries.

    Args:
        dictionary (dict): The dictionary containing nested dictionaries to check.
        key (str): The key to look for in all nested dictionaries.

    Raises:
        ValueError: If the key is not present in any of the nested dictionaries.

    Example:
        >>> nested_dict = {
                "101": {"name": "Alice", "age": 30},
                "102": {"name": "Bob", "age": 25},
            }
        >>> check_key_in_all_nested_dicts(nested_dict, "age")
        # No error is raised
        >>> check_key_in_all_nested_dicts(nested_dict, "salary")
        # Raises ValueError: 'salary is not present in all nested dictionaries.'
    """
    for dict_key, sub_dict in dictionary.items():
        if not isinstance(sub_dict, dict) or key not in sub_dict:
            raise ValueError(
                f"'{key}' is not present in '{dict_key}' nested dictionary."
            )
