from typing import Callable

from sklearn.metrics.pairwise import cosine_distances
from collections import Counter
import numpy as np


def intralist_diversity(
    R: np.ndarray[np.ndarray],
    pairwise_distance_function: Callable = cosine_distances,
) -> float:
    """Calculate the intra-list diversity of a recommendation list.

    This function implements the method described by Smyth and McClave (2001) to
    measure the diversity within a recommendation list. It calculates the average
    pairwise distance between all items in the list.

    Formula:
        Diversity(R) = ( sum_{i∈R} sum_{j∈R_{i}} dist(i, j) )  / ( |R|(|R|-1) )

    where `R` is the recommendation list, and `dist` represents the pairwise distance function used.

    Args:
        R (np.ndarray[np.ndarray]): A 2D numpy array where each row represents a recommendation.
            This array should be either array-like or a sparse matrix, with shape (n_samples_X, n_features).
        pairwise_distance_function (Callable, optional): A function to compute pairwise distance
            between samples. Defaults to `cosine_distances`.

    Returns:
        float: The calculated diversity score. If the recommendation list contains less than or
            equal to one item, NaN is returned to signify an undefined diversity score.

    References:
        Smyth, B., McClave, P. (2001). Similarity vs. Diversity. In: Aha, D.W., Watson, I. (eds)
        Case-Based Reasoning Research and Development. ICCBR 2001. Lecture Notes in Computer Science(),
        vol 2080. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-44593-5_25

    Examples:
        >>> R1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        >>> print(intralist_diversity(R1))
            0.022588438516842262
        >>> print(intralist_diversity(np.array([[0.1, 0.2], [0.1, 0.2]])))
            1.1102230246251565e-16
    """
    R_n = R.shape[0]  # number of recommendations
    if R_n <= 1:
        # Less than or equal to 1 recommendations in recommendation list
        diversity = np.nan
    else:
        pairwise_distances = pairwise_distance_function(R, R)
        diversity = np.sum(pairwise_distances) / (R_n * (R_n - 1))
    return diversity


def serendipity(
    R: np.ndarray[np.ndarray],
    H: np.ndarray[np.ndarray],
    pairwise_distance_function: Callable = cosine_distances,
) -> float:
    """Calculate the serendipity score between a set of recommendations and user's reading history.

    This function implements the concept of serendipity as defined by Feng Lu, Anca Dumitrache, and David Graus (2020).
    Serendipity in this context is measured as the mean distance between the items in the recommendation list and the
    user's reading history.

    Formula:
        Serendipity(R, H) = ( sum_{i∈R} sum_{j∈R} dist(i, j) )  / ( |R||H| )

    where `R` is the recommendation list, `H` is the user's reading history, and `dist` is the pairwise distance function.

    Args:
        R (np.ndarray[np.ndarray]): A 2D numpy array representing the recommendation list, where each row is a recommendation.
            It should be either array-like or a sparse matrix, with shape (n_samples_X, n_features).
        H (np.ndarray[np.ndarray]): A 2D numpy array representing the user's reading history, with the same format as R.
        pairwise_distance_function (Callable, optional): A function to compute pairwise distance between samples.
            Defaults to `cosine_distances`.

    Returns:
        float: The calculated serendipity score.

    References:
        Lu, F., Dumitrache, A., & Graus, D. (2020). Beyond Optimizing for Clicks: Incorporating Editorial Values in News Recommendation.
        Retrieved from https://arxiv.org/abs/2004.09980

    Examples:
        >>> R1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> H1 = np.array([[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
        >>> print(serendipity(R1, H1))
            0.016941328887631724
    """
    # Compute the pairwise distances between each vector:
    dists = pairwise_distance_function(R, H)
    # Compute serendipity:
    return np.mean(dists)


def coverage_count(R: np.ndarray) -> int:
    """Calculate the number of distinct items in a recommendation list.

    Args:
        R (np.ndarray): An array containing the items in the recommendation list.

    Returns:
        int: The count of distinct items in the recommendation list.

    Examples:
        >>> R1 = np.array([1, 2, 3, 4, 5, 5, 6])
        >>> print(coverage_count(R1))
            6
    """
    # Distinct items:
    return np.unique(R).size


def coverage_fraction(R: np.ndarray, C: np.ndarray) -> float:
    """Calculate the fraction of distinct items in the recommendation list compared to a universal set.

    Args:
        R (np.ndarray): An array containing the items in the recommendation list.
        C (np.ndarray): An array representing the universal set of items.
            It should contain all possible items that can be recommended.

    Returns:
        float: The fraction representing the coverage of the recommendation system.
            This is calculated as the size of unique elements in R divided by the size of unique elements in C.

    Examples:
        >>> R1 = np.array([1, 2, 3, 4, 5, 5, 6])
        >>> C1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> print(coverage_fraction(R1, C1))  # Expected output: 0.6
            0.6
    """
    # Distinct items:
    return np.unique(R).size / np.unique(C).size


def novelty(R: np.ndarray[float]) -> float:
    """Calculate the novelty score of recommendations based on their popularity.

    This function computes the novelty score for a set of recommendations by applying the self-information popularity metric.
    It uses the formula described by Zhou et al. (2010) and Vargas and Castells (2011). The novelty is calculated as the
    average negative logarithm (base 2) of the popularity scores of the items in the recommendation list.

    Formula:
        Novelty(R) = ( sum_{i∈R} -log2( p_i ) / ( |R| )

    where p_i represents the popularity score of each item in the recommendation list R, and |R| is the size of R.

    Args:
        R (np.ndarray[float]): An array of popularity scores (p_i) for each item in the recommendation list.

    Returns:
        float: The calculated novelty score. Higher values indicate less popular (more novel) recommendations.

    References:
        Zhou et al. (2010).
        Vargas & Castells (2011).

    Examples:
        >>> print(novelty([0.1, 0.2, 0.3, 0.4, 0.5]))  # Expected: High score (low popularity scores)
            1.9405499757656586
        >>> print(novelty([0.9, 0.9, 0.9, 1.0, 0.5]))  # Expected: Low score (high popularity scores)
            0.29120185606703
    """
    return np.mean(-np.log2(R))


def index_of_dispersion(x: list[int]) -> float:
    """
    Computes the Index of Dispersion (variance-to-mean ratio) for a given dataset of nominal variables.

    The Index of Dispersion is a statistical measure used to quantify the dispersion or variability of a distribution
    relative to its mean. It's particularly useful in identifying whether a dataset follows a Poisson distribution,
    where the Index of Dispersion would be approximately 1.

    Formula:
        D = ( k * (N^2 - Σf^2) ) / ( N^2 * (k-1) )
    Where:
        k = number of categories in the data set (including categories with zero items),
        N = number of items in the set,
        f = number of frequencies or ratings,
        Σf^2 = sum of squared frequencies/ratings.

    Args:
        x (list[int]): A list of integers representing frequencies or counts of occurrences in different categories.
                        Each integer in the list corresponds to the count of occurrences in a given category.

    Returns:
        float: The Index of Dispersion for the dataset. Returns `np.nan` if the input list contains only one item,
                indicating an undefined Index of Dispersion. Returns 0 if there's only one category present in the dataset.

    References:
        Walker, 1999, Statistics in criminal
        Source: https://www.statisticshowto.com/index-of-dispersion/

    Examples:
        Given the following categories: Math(25), Economics(42), Chemistry(13), Physical Education (8), Religious Studies (13).
        >>> N = np.sum(25+42+13+8+13)
        >>> k = 5
        >>> sq_f2 = np.sum(25**2 + 42**2 + 13**2 + 8**2 + 13**2)
        >>> iod = ( k * (N**2 - sq_f2)) / ( N**2 * (k-1) )
            0.9079992157631604

        Validate method:
        >>> cat = [[1]*25, [2]*42, [3]*13, [4]*8, [5]*13]
        >>> flat_list = [item for sublist in cat for item in sublist]
        >>> index_of_dispersion(flat_list)
            0.9079992157631604
    """
    # number of items
    N = len(x)
    # compute frequencies
    count = Counter(x)
    # number of categories
    k = len(count)
    if k == 1:
        if N == 1:
            return np.nan
        else:
            return 0
    # squared frequencies
    f_squared = [count.get(f) ** 2 for f in count]
    # compute Index of Dispersion
    D = k * (N**2 - sum(f_squared)) / (N**2 * (k - 1))
    return D
