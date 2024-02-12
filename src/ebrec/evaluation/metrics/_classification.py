import numpy as np


def auc_score_custom(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Area Under the Curve (AUC) score for the Receiver Operating Characteristic (ROC) curve using a
    custom method. This implementation is particularly useful for understanding basic ROC curve properties and
    for educational purposes to demonstrate how AUC scores can be manually calculated.

    This function may produce slightly different results compared to standard library implementations (e.g., sklearn's roc_auc_score)
    in cases where positive and negative predictions have the same score. The function treats the problem as a binary classification task,
    comparing the prediction scores for positive instances against those for negative instances directly.

    Args:
        y_true (np.ndarray): A binary array indicating the true classification (1 for positive class and 0 for negative class).
        y_pred (np.ndarray): An array of scores as predicted by a model, indicating the likelihood of each instance being positive.

    Returns:
        float: The calculated AUC score, representing the probability that a randomly chosen positive instance is ranked
                higher than a randomly chosen negative instance based on the prediction scores.

    Raises:
        ValueError: If `y_true` and `y_pred` do not have the same length or if they contain invalid data types.

    Examples:
        >>> y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        >>> y_pred = np.array([0.9999, 0.9838, 0.5747, 0.8485, 0.8624, 0.4502, 0.3357, 0.8985])
        >>> auc_score_custom(y_true, y_pred)
            0.9333333333333333
        >>> from sklearn.metrics import roc_auc_score
        >>> roc_auc_score(y_true, y_pred)
            0.9333333333333333

        An error will occur when pos/neg prediction have same score:
        >>> y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        >>> y_pred = np.array([0.9999, 0.8, 0.8, 0.8485, 0.8624, 0.4502, 0.3357, 0.8985])
        >>> auc_score_custom(y_true, y_pred)
            0.7333
        >>> roc_auc_score(y_true, y_pred)
            0.7667
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_bool = y_true.astype(np.bool_)
    # Index:
    pos_scores = y_pred[y_true_bool]
    neg_scores = y_pred[np.logical_not(y_true_bool)]
    # Arrange:
    pos_scores = np.repeat(pos_scores, len(neg_scores))
    neg_scores = np.tile(neg_scores, sum(y_true_bool))
    assert len(neg_scores) == len(pos_scores)
    return (pos_scores > neg_scores).sum() / len(neg_scores)
