from typing import Iterable, Callable

from sklearn.metrics.pairwise import cosine_distances
from itertools import combinations, chain
import numpy as np

from ebrec.evaluation.metrics._beyond_accuracy import (
    intralist_diversity,
    coverage_fraction,
    coverage_count,
    serendipity,
    novelty,
)

from ebrec.evaluation.utils import (
    compute_normalized_distribution,
    check_key_in_all_nested_dicts,
    is_iterable_nested_dtype,
    compute_combinations,
    get_keys_in_dict,
)


### IntralistDiversity
class IntralistDiversity:
    """
    A class for calculating the intralist diversity metric for recommendations in a recommendation system, as proposed
    by Smyth and McClave in 2001. This metric assesses the diversity within a list of recommendations by computing the
    average pairwise distance between all items in the recommendation list.

    Examples:
        >>> div = IntralistDiversity()
        >>> R = np.array([
                ['item1', 'item2'],
                ['item2', 'item3'],
                ['item3', 'item4']
            ])
        >>> lookup_dict = {
                'item1': {'vector': [0.1, 0.2]},
                'item2': {'vector': [0.2, 0.3]},
                'item3': {'vector': [0.3, 0.4]},
                'item4': {'vector': [0.4, 0.5]}
            }
        >>> lookup_key = 'vector'
        >>> pairwise_distance_function = cosine_distances
        >>> div(R, lookup_dict, lookup_key, pairwise_distance_function)
            array([0.00772212, 0.00153965, 0.00048792])
        >>> div._candidate_diversity(list(lookup_dict), 2, lookup_dict, lookup_key)
            (0.0004879239129211843, 0.02219758592259058)
    """

    def __init__(self) -> None:
        self.name = "intralist_diversity"

    def __call__(
        self,
        R: np.ndarray[np.ndarray[str]],
        lookup_dict: dict[str, dict[str, any]],
        lookup_key: str,
        pairwise_distance_function: Callable = cosine_distances,
    ) -> np.ndarray[float]:
        """
        Calculates the diversity score for each subset of recommendations in `R` using the provided `lookup_dict`
        to find the document vectors and a `pairwise_distance_function` to calculate the diversity. The diversity is
        calculated as the average pairwise distance between all items within each subset of recommendations.

        Args:
            R (np.ndarray[np.ndarray[str]]): A numpy array of numpy arrays, where each inner array contains the IDs
                (as the lookup value in 'lookup_dict') of items for which the diversity score will be calculated.
            lookup_dict (dict[str, dict[str, any]]): A nested dictionary where each key is an item ID and the value is
                another dictionary containing item attributes, including the document vectors identified by `lookup_key`.
            lookup_key (str): The key within the nested dictionaries of `lookup_dict` that corresponds to the document
                vector of each item.
            pairwise_distance_function (Callable, optional): A function that takes two arrays of vectors and returns a
                distance matrix. Defaults to cosine_distances, which measures the cosine distance between vectors.

        Returns:
            np.ndarray[float]: An array of floating-point numbers representing the diversity score for each subset of
                recommendations in `R`.
        """
        check_key_in_all_nested_dicts(lookup_dict, lookup_key)
        diversity_scores = []
        for sample in R:
            ids = get_keys_in_dict(sample, lookup_dict)
            if len(ids) == 0:
                divesity_score = np.nan
            else:
                document_vectors = np.array(
                    [lookup_dict[id].get(lookup_key) for id in ids]
                )
                divesity_score = intralist_diversity(
                    document_vectors,
                    pairwise_distance_function=pairwise_distance_function,
                )
            diversity_scores.append(divesity_score)
        return np.asarray(diversity_scores)

    def _candidate_diversity(
        self,
        R: np.ndarray[str],
        n_recommendations: int,
        lookup_dict: dict[str, dict[str, any]],
        lookup_key: str,
        pairwise_distance_function: Callable = cosine_distances,
        max_number_combinations: int = 20000,
        seed: int = None,
    ):
        """
        Estimates the minimum and maximum diversity scores for candidate recommendations.

        Args:
            R (np.ndarray[str]): An array of item IDs from which to generate recommendation combinations.
            n_recommendations (int): The number of recommendations per combination to evaluate.
            lookup_dict (dict[str, dict[str, any]]): A dictionary mapping item IDs to their attributes, including the
                vectors identified by `lookup_key` used for calculating diversity.
            lookup_key (str): The key within the attribute dictionaries of `lookup_dict` corresponding to the item
                vectors used in diversity calculations.
            pairwise_distance_function (Callable, optional): A function to calculate the pairwise distance between item
                vectors. Defaults to `cosine_distances`.
            max_number_combinations (int, optional): The maximum number of combinations to explicitly evaluate for
                diversity before switching to random sampling. Defaults to 20000.
            seed (int, optional): A seed for the random number generator to ensure reproducible results when sampling
                combinations. Defaults to None.

        Returns:
            tuple[float, float]: The minimum and maximum diversity scores among the evaluated combinations of
            recommendations.
        """
        #
        check_key_in_all_nested_dicts(lookup_dict, lookup_key)
        R = get_keys_in_dict(R, lookup_dict)
        n_items = len(R)
        if n_recommendations > n_items:
            raise ValueError(
                f"'n_recommendations' cannot exceed the number of items in R (items in candidate list). {n_recommendations} > {n_items}"
            )
        n_combinations = compute_combinations(n_items, n_recommendations)
        # Choose whether to compute or estimate the min-max diversity based on number of combinations to compute:
        if n_combinations > max_number_combinations:
            np.random.seed(seed)
            aids_iterable = chain(
                np.random.choice(R, n_recommendations, replace=False)
                for _ in range(max_number_combinations)
            )
        else:
            aids_iterable = combinations(R, n_recommendations)

        diversity_scores = self.__call__(
            aids_iterable,
            lookup_dict=lookup_dict,
            lookup_key=lookup_key,
            pairwise_distance_function=pairwise_distance_function,
        )
        return diversity_scores.min(), diversity_scores.max()


### Distribution
class Distribution:
    """
    A class designed to compute the normalized distribution of specified attributes for a set of items.

    Examples:
        >>> dist = Distribution()
        >>> R = np.array([['item1', 'item2'], ['item2', 'item3']])
        >>> lookup_dict = {
                "item1": {"g": "Action", "sg": ["Action", "Thriller"]},
                "item2": {"g": "Action", "sg": ["Action", "Comedy"]},
                "item3": {"g": "Comedy", "sg": ["Comedy"]},
            }
        >>> dist(R, lookup_dict, 'g')
            {'Action': 0.75, 'Comedy': 0.25}
        >>> dist(R, lookup_dict, 'sg')
            {'Action': 0.42857142857142855, 'Thriller': 0.14285714285714285, 'Comedy': 0.42857142857142855}
    """

    def __init__(self) -> None:
        self.name = "distribution"

    def __call__(
        self,
        R: np.ndarray[np.ndarray[str]],
        lookup_dict: dict[str, dict[str, any]],
        lookup_key: str,
    ) -> dict[str, float]:
        """
        Args:
            R (np.ndarray[np.ndarray[str]]): A 2D numpy array of item IDs, where each sub-array represents a
                list of item IDs for which to compute the distribution of their attributes.
            lookup_dict (dict[str, dict[str, any]]): A dictionary mapping item IDs to their attributes, where
                each item's attributes are stored in a nested dictionary.
            lookup_key (str): The key to look for within the nested attribute dictionaries of `lookup_dict` to
                retrieve the item's representation for distribution computation.

        Returns:
            dict[str, float]: A dictionary with keys representing the unique values of the item representations
            retrieved with `lookup_key` and values being the normalized frequency of these representations
            across all items in `R`.
        """
        check_key_in_all_nested_dicts(lookup_dict, lookup_key)

        R_flat = np.asarray(R).ravel()
        R_flat = get_keys_in_dict(R_flat, lookup_dict)
        item_representations = [lookup_dict[id].get(lookup_key) for id in R_flat]
        # If an item has multple representations it may be nested
        if is_iterable_nested_dtype(item_representations, (list, np.ndarray)):
            item_representations = np.concatenate(item_representations)
        # Compute distribution
        return compute_normalized_distribution(item_representations)


### Coverage:
class Coverage:
    """
    A class designed to measure the coverage of recommendation systems. Coverage is an important metric in
    recommender systems as it indicates the extent to which a recommendation system utilizes its item catalog.
    There are two types of coverage measured: count coverage and fractional coverage.
    - Count coverage (`Coverage_count`) is the total number of unique items recommended across all users:
        * Coverage_count = |R|
    - Fractional coverage (`Coverage_frac`) is the ratio of the count coverage to the total number of items
        in the candidate set, representing the proportion of the item catalog covered by recommendations.
        * Coverage_frac = |R| / |I|

    Examples:
        >>> cov = Coverage()
        >>> R = np.array([
                ['item1', 'item2'],
                ['item2', 'item3'],
                ['item4',  'item3']
            ])
        >>> C = np.array(['item1', 'item2', 'item3', 'item4', 'item5', 'item6'])
        >>> cov(R, C)
            (4, 0.6666666666666666)
    """

    def __init__(self) -> None:
        self.name = "coverage"

    def __call__(
        self,
        R: np.ndarray[np.ndarray[any]],
        C: np.ndarray[any] = [],
    ):
        coverage_c = coverage_count(R)
        coverage_f = coverage_fraction(R, C) if len(C) > 0 else -np.inf
        return coverage_c, coverage_f


### Sentiment:
class Sentiment:
    """
    A class designed to evaluate sentiment scores for items within nested arrays
    based on a lookup dictionary.

    Args:
        R (np.ndarray): A numpy array of numpy arrays containing strings, where each
            sub-array represents a group of items whose sentiment scores are to be averaged.
        lookup_dict (dict): A dictionary where each key is an item name (as found in `R`)
            and its value is another dictionary containing sentiment scores and potentially
            other information.
        lookup_key (str): The key within the nested dictionaries of `lookup_dict` that
            contains the sentiment score.

    Returns:
        np.ndarray: A numpy array containing the average sentiment score for each sub-array
            in `R`.

    Raises:
        KeyError: If `lookup_key` is not found in any of the nested dictionaries in `lookup_dict`.

    Examples:
        >>> sent = Sentiment()
        >>> R = np.array([['item1', 'item2'], ['item2', 'item3'], ['item2', 'item5']])
        >>> lookup_dict = {
                "item1": {"s": 1.00, "na" : []},
                "item2": {"s": 0.50, "na" : []},
                "item3": {"s": 0.25, "na" : []},
                "item4": {"s": 0.00, "na" : []},
            }
        >>> lookup_key = "s"
        >>> sent(R, lookup_dict, 's')
            array([0.75 , 0.375, 0.5 ])
        >>> sent._candidate_sentiment(list(lookup_dict), 1, lookup_dict, lookup_key)
            (1.0, 0.0)
    """

    def __init__(self) -> None:
        self.name = "sentiment"

    def __call__(
        self,
        R: np.ndarray,
        lookup_dict: dict[str, dict[str, any]],
        lookup_key: str,
    ):
        check_key_in_all_nested_dicts(lookup_dict, lookup_key)
        sentiment_scores = []
        for sample in R:
            ids = get_keys_in_dict(sample, lookup_dict)
            sentiment_scores.append(
                np.mean([lookup_dict[id].get(lookup_key) for id in ids])
            )
        return np.asarray(sentiment_scores)

    def _candidate_sentiment(
        self,
        R: np.ndarray,
        n_recommendations: int,
        lookup_dict: dict[str, dict[str, any]],
        lookup_key: str,
    ):
        """
        Compute the minimum and maximum sentiment scores for candidate recommendations.

        Args:
            R (np.ndarray[str]): An array of item IDs from which to generate recommendation combinations.
            n_recommendations (int): The number of recommendations per combination to evaluate.
            lookup_dict (dict[str, dict[str, any]]): A dictionary mapping item IDs to their attributes, including the
                vectors identified by `lookup_key` used for calculating diversity.
            lookup_key (str): The key within the attribute dictionaries of `lookup_dict` corresponding to the item
                vectors used in diversity calculations.

        Returns:
            tuple[float, float]: The minimum and maximum sentiment scores among the candidate list.
        """
        #
        check_key_in_all_nested_dicts(lookup_dict, lookup_key)
        R = get_keys_in_dict(R, lookup_dict)
        sentiment_scores = sorted([lookup_dict[id].get(lookup_key) for id in R])

        n_lowest_scores = sentiment_scores[:n_recommendations]
        n_highest_scores = sentiment_scores[-n_recommendations:]

        min_novelty = np.mean(n_highest_scores)
        max_novelty = np.mean(n_lowest_scores)

        return min_novelty, max_novelty


### Serendipity:
class Serendipity:
    """
    A class for calculating the serendipity of recommendation sets in relation to users' historical interactions.

    Formula:
        Serendipity(R, H) = ( sum_{i∈R} sum_{j∈R} dist(i, j) )  / ( |R||H| )
    * (It is simply the avarage computation; sum(dist)/(Number of observations)

    Examples:
        >>> ser = Serendipity()
        >>> R = [np.array(['item1', 'item2']), np.array(['item3', 'item4'])]
        >>> H = [np.array(['itemA', 'itemB']), np.array(['itemC', 'itemD'])]
        >>> lookup_dict = {
                'item1': {'vector': [0.1, 0.2]},
                'item2': {'vector': [0.2, 0.3]},
                'item3': {'vector': [0.3, 0.4]},
                'item4': {'vector': [0.4, 0.5]},
                'itemA': {'vector': [0.5, 0.6]},
                'itemB': {'vector': [0.6, 0.7]},
                'itemC': {'vector': [0.7, 0.8]},
                'itemD': {'vector': [0.8, 0.9]}
            }
        >>> lookup_key = 'vector'
        >>> pairwise_distance_function = cosine_distances
        >>> ser(R, H, lookup_dict, lookup_key, pairwise_distance_function)
            array([0.01734935, 0.00215212])
    """

    def __init__(self) -> None:
        self.name = "serendipity"

    def __call__(
        self,
        R: Iterable[np.ndarray[str]],
        H: Iterable[np.ndarray[str]],
        lookup_dict: dict[str, any],
        lookup_key: str,
        pairwise_distance_function: Callable = cosine_distances,
    ):
        """
        Calculates the serendipity scores for a set of recommendations given the users' click histories. Serendipity
        is measured based on the novelty and unexpectedness of recommendations compared to previously interacted items,
        utilizing a pairwise distance function to quantify differences between item vectors.

        Args:
            R (np.ndarray[np.ndarray[str]]): A 2D numpy array where each sub-array contains item IDs for a set of
                recommendations.
            H (Iterable[np.ndarray[str]]): An iterable of numpy arrays, with each array containing item IDs
                that represent a user's click history.
            lookup_dict (dict[str, any]): A dictionary mapping item IDs to their attributes, where each item's attributes
                are stored in a dictionary and `lookup_key` is used to retrieve the item's vector.
            lookup_key (str): The key within the item attribute dictionaries to retrieve the vector used for calculating
                serendipity.
            pairwise_distance_function (Callable, optional): A function to calculate the pairwise distance between item
                vectors. Defaults to cosine_distances.

        Returns:
            np.ndarray: An array of serendipity scores, with one score per set of recommendations. If a recommendation set
            or click history set lacks valid vectors, the corresponding serendipity score is marked as NaN.

        Raises:
            ValueError: If the lengths of `R` and `click_histories` do not match, indicating a mismatch in the number
            of recommendation sets and click history sets.
        """
        # Sanity:
        if len(R) != len(H):
            raise ValueError(
                f"The lengths of 'R' and 'H' do not match ({len(R)} != {len(H)})."
            )
        check_key_in_all_nested_dicts(lookup_dict, lookup_key)
        # =>
        serendipity_scores = []
        for r_u, ch_u in zip(R, H):
            # Only keep the valid IDs:
            r_u = get_keys_in_dict(np.asarray(r_u).ravel(), lookup_dict)
            ch_u = get_keys_in_dict(np.asarray(ch_u).ravel(), lookup_dict)

            r_i_vectors = [lookup_dict[id].get(lookup_key) for id in r_u]
            ch_i_vectors = [lookup_dict[id].get(lookup_key) for id in ch_u]

            if len(r_i_vectors) == 0 or len(ch_i_vectors) == 0:
                serendipity_score = np.nan
            else:
                serendipity_score = serendipity(
                    r_i_vectors, ch_i_vectors, pairwise_distance_function
                )
            serendipity_scores.append(serendipity_score)
        return np.asarray(serendipity_scores)


### Novelty
class Novelty:
    """
    A class for calculating the novelty of recommendation sets based on pre-computed popularity scores.

    Formula:
        Novelty(R) = ( sum_{i∈R} -log2( p(i) ) / ( |R| )

    Examples:
        >>> R = [
                np.array(['item1', 'item2']),
                np.array(['item3', 'item4'])
            ]
        >>> lookup_dict = {
                'item1': {'popularity': 0.05},
                'item2': {'popularity': 0.1},
                'item3': {'popularity': 0.2},
                'item4': {'popularity': 0.3},
                'item5': {'popularity': 0.4}
            }
        >>> nov = Novelty()
        >>> nov(R, lookup_dict, 'popularity')
            array([3.82192809, 2.02944684])
        >>> nov._candidate_novelty(list(lookup_dict), 2, lookup_dict, 'popularity')
            (1.5294468445267841, 3.8219280948873626)
    """

    def __init__(self) -> None:
        self.name = "novelty"

    def __call__(
        self,
        R: np.ndarray[np.ndarray[str]],
        lookup_dict: dict[str, any],
        lookup_key: str,
    ):
        """
        Calculate novelty scores for each set of recommendations based on their popularity scores.

        Args:
            R (np.ndarray): A numpy array of numpy arrays, where each inner array contains recommendation IDs.
            lookup_dict (dict): A dictionary where keys are recommendation IDs and values are dictionaries
                                containing various attributes of each recommendation, including popularity scores.
            lookup_key (str): The key within the nested dictionaries of `lookup_dict` to retrieve the popularity score.

        Returns:
            np.ndarray: An array of novelty scores for each recommendation set in `R`.
        """
        #
        check_key_in_all_nested_dicts(lookup_dict, lookup_key)
        # Popularity scores in recommendations:
        novelty_scores = []
        for r_u in R:
            r_u = get_keys_in_dict(r_u, lookup_dict)
            popularity_scores = [lookup_dict[id].get(lookup_key) for id in r_u]
            novelty_scores.append(novelty(popularity_scores))
        return np.asarray(novelty_scores)

    def _candidate_novelty(
        self,
        R: np.ndarray[str],
        n_recommendations: int,
        lookup_dict: dict[str, dict[str, any]],
        lookup_key: str,
    ):
        """
        Compute the minimum and maximum novelty scores for candidate recommendations.

        Args:
            R (np.ndarray[str]): An array of item IDs from which to generate recommendation combinations.
            n_recommendations (int): The number of recommendations per combination to evaluate.
            lookup_dict (dict[str, dict[str, any]]): A dictionary mapping item IDs to their attributes, including the
                vectors identified by `lookup_key` used for calculating diversity.
            lookup_key (str): The key within the attribute dictionaries of `lookup_dict` corresponding to the item
                vectors used in diversity calculations.

        Returns:
            tuple[float, float]: The minimum and maximum novelty scores among the candidate list.
        """
        #
        check_key_in_all_nested_dicts(lookup_dict, lookup_key)
        R = get_keys_in_dict(R, lookup_dict)
        popularity_scores = sorted([lookup_dict[id].get(lookup_key) for id in R])

        n_lowest_scores = popularity_scores[:n_recommendations]
        n_highest_scores = popularity_scores[-n_recommendations:]

        min_novelty = novelty(n_highest_scores)
        max_novelty = novelty(n_lowest_scores)

        return min_novelty, max_novelty
