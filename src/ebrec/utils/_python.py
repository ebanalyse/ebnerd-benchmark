from typing import Iterable
from pathlib import Path
from tqdm import tqdm
import polars as pl
import numpy as np
import datetime
import zipfile
import torch
import time
import json
import yaml
import time


def read_json_file(path: str, verbose: bool = False) -> dict:
    if verbose:
        print(f"Writing JSON: '{path}'")
    with open(path) as file:
        return json.load(file)


def write_json_file(dictionary: dict, path: str, verbose: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(dictionary, file)
    if verbose:
        print(f"Writing JSON: '{path}'")


def read_yaml_file(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def write_yaml_file(dictionary: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.dump(dictionary, file, default_flow_style=False)


def rank_predictions_by_score(
    arr: Iterable[float],
) -> list[np.ndarray]:
    """
    Converts the prediction scores based on their ranking (1 for highest score,
    2 for second highest, etc.), effectively ranking prediction scores for each row.

    Reference:
        https://github.com/recommenders-team/recommenders/blob/main/examples/00_quick_start/nrms_MIND.ipynb

    >>> prediction_scores = [[0.2, 0.1, 0.3], [0.1, 0.2], [0.4, 0.2, 0.1, 0.3]]
    >>> [rank_predictions_by_score(row) for row in prediction_scores]
        [
            array([2, 3, 1]),
            array([2, 1]),
            array([1, 3, 4, 2])
        ]
    """
    return np.argsort(np.argsort(arr)[::-1]) + 1


def write_submission_file(
    impression_ids: Iterable[int],
    prediction_scores: Iterable[any],
    path: Path = Path("predictions.txt"),
    rm_file: bool = True,
    filename_zip: str = None,
) -> None:
    """
    We align the submission file similar to MIND-format for users who are familar.

    Reference:
        https://github.com/recommenders-team/recommenders/blob/main/examples/00_quick_start/nrms_MIND.ipynb

    Example:
    >>> impression_ids = [237, 291, 320]
    >>> prediction_scores = [[0.2, 0.1, 0.3], [0.1, 0.2], [0.4, 0.2, 0.1, 0.3]]
    >>> write_submission_file(impression_ids, prediction_scores, path="predictions.txt", rm_file=False)
    ## Output file:
        237 [0.2,0.1,0.3]
        291 [0.1,0.2]
        320 [0.4,0.2,0.1,0.3]
    """
    path = Path(path)
    with open(path, "w") as f:
        for impr_index, preds in tqdm(zip(impression_ids, prediction_scores)):
            preds = "[" + ",".join([str(i) for i in preds]) + "]"
            f.write(" ".join([str(impr_index), preds]) + "\n")
    # =>
    zip_submission_file(path=path, rm_file=rm_file, filename_zip=filename_zip)


def read_submission_file(path: Path) -> tuple[int, any]:
    """
    >>> impression_ids = [237, 291, 320]
    >>> prediction_scores = [[0.2, 0.1, 0.3], [0.1, 0.2], [0.4, 0.2, 0.1, 0.3]]
    >>> write_submission_file(impression_ids, prediction_scores, path="predictions.txt", rm_file=False)
    >>> read_submission_file("predictions.txt")
        (
            [237, 291, 320],
            [[0.2, 0.1, 0.3], [0.1, 0.2], [0.4, 0.2, 0.1, 0.3]]
        )
    """
    # Read and parse the file
    impression_ids = []
    prediction_scores = []
    with open(path, "r") as file:
        for line in file:
            impression_id_str, scores_str = parse_line(line)
            impression_ids.append(int(impression_id_str))
            prediction_scores.append(scores_str)
    return impression_ids, prediction_scores


def zip_submission_file(
    path: Path,
    filename_zip: str = None,
    verbose: bool = True,
    rm_file: bool = True,
) -> None:
    """
    Compresses a specified file into a ZIP archive within the same directory.

    Args:
        path (Path): The directory path where the file to be zipped and the resulting zip file will be located.
        filename_input (str, optional): The name of the file to be compressed. Defaults to the path.name.
        filename_zip (str, optional): The name of the output ZIP file. Defaults to "prediction.zip".
        verbose (bool, optional): If set to True, the function will print the process details. Defaults to True.
        rm_file (bool, optional): If set to True, the original file will be removed after compression. Defaults to True.

    Returns:
        None: This function does not return any value.
    """
    path = Path(path)
    if filename_zip:
        path_zip = path.parent.joinpath(filename_zip)
    else:
        path_zip = path.with_suffix(".zip")

    if path_zip.suffix != ".zip":
        raise ValueError(f"suffix for {path_zip.name} has to be '.zip'")
    if verbose:
        print(f"Zipping {path} to {path_zip}")
    f = zipfile.ZipFile(path_zip, "w", zipfile.ZIP_DEFLATED)
    f.write(path, arcname=path.name)
    f.close()
    if rm_file:
        path.unlink()


def parse_line(l) -> tuple[str, list[float]]:
    """
    Parses a single line of text into an identifier and a list of ranks.
    """
    impid, ranks = l.strip("\n").split()
    ranks = json.loads(ranks)
    return impid, ranks


def time_it(enable=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if enable:
                start_time = time.time()
            result = func(*args, **kwargs)
            if enable:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"... {func.__name__} completed in {elapsed_time:.2f} seconds")
            return result

        return wrapper

    return decorator


def df_shape_time_it(enable=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            #
            if enable:
                try:
                    # Incase of LazyFrame, this is not possible:
                    start_shape = args[0].shape
                except:
                    pass
                start_time = time.time()

            # Run function:
            result = func(*args, **kwargs)

            #
            if enable:
                end_time = time.time()
                time_taken = round(end_time - start_time, 6)
                try:
                    # Incase of LazyFrame, this is not possible:
                    end_shape = result.shape
                    row_dropped_frac = round(
                        (start_shape[0] - end_shape[0]) / start_shape[0] * 100, 2
                    )
                    shape_ba = f"=> Before/After: {start_shape}/{end_shape} ({row_dropped_frac}% rows dropped)"
                except:
                    shape_ba = f"=> Before/After: NA/NA (NA% rows dropped)"
                print(
                    f"""Time taken by '{func.__name__}': {time_taken} seconds\n{shape_ba}"""
                )
            return result

        return wrapper

    return decorator


def generate_unique_name(existing_names: list[str], base_name: str = "new_name"):
    """
    Generate a unique name based on a list of existing names.

    Args:
        existing_names (list of str): The list of existing names.
        base_name (str): The base name to start with. Default is 'newName'.

    Returns:
        str: A unique name.
    Example
    >>> existing_names = ['name1', 'name2', 'newName', 'newName_1']
    >>> generate_unique_name(existing_names, 'newName')
        'newName_2'
    """
    if base_name not in existing_names:
        return base_name

    suffix = 1
    new_name = f"{base_name}_{suffix}"

    while new_name in existing_names:
        suffix += 1
        new_name = f"{base_name}_{suffix}"

    return new_name


def compute_npratio(n_pos: int, n_neg: int) -> float:
    """
    Similar approach as:
        "Neural News Recommendation with Long- and Short-term User Representations (An et al., ACL 2019)"

    Example:
    >>> pos = 492_185
    >>> neg = 9_224_537
    >>> round(compute_npratio(pos, neg), 2)
        18.74
    """
    return 1 / (n_pos / n_neg)


def strfdelta(tdelta: datetime.timedelta):
    """
    Example:
    >>> tdelta = datetime.timedelta(days=1, hours=3, minutes=42, seconds=54)
    >>> strfdelta(tdelta)
        '1 days 3:42:54'
    """
    days = tdelta.days
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days} days {hours}:{minutes}:{seconds}"


def str_datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_object_variables(object_: object) -> dict:
    """
    Example:
    >>> class example:
            a = 2
            b = 3
    >>> get_object_variables(example)
        {'a': 2, 'b': 3}
    """
    return {
        name: value
        for name, value in vars(object_).items()
        if not name.startswith("__") and not callable(value)
    }


def batch_items_generator(items: Iterable[any], batch_size: int):
    """
    Generator function that chunks a list of items into batches of a specified size.

    Args:
        items (list): The list of items to be chunked.
        batch_size (int): The number of items to include in each batch.

    Yields:
        list: A batch of items from the input list.

    Examples:
        >>> items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> batch_size = 3
        >>> for batch in chunk_list(items, batch_size):
        ...     print(batch)
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def unnest_dictionary(dictionary, parent_key="") -> dict:
    """
    Unnests a dictionary by adding the key to the nested names.

    Args:
        dictionary (dict): The nested dictionary to be unnested.
        parent_key (str, optional): The parent key to be prepended to the nested keys. Defaults to "".

    Returns:
        dict: The unnested dictionary where each nested key is prefixed with the parent keys, separated by dots.

    Example:
    >>> nested_dict = {
            "key1": "value1",
            "key2": {"nested_key1": "nested_value1", "nested_key2": "nested_value2"},
            "key3": {"nested_key3": {"deeply_nested_key": "deeply_nested_value"}},
        }
    >>> unnest_dictionary(nested_dict)
        {
            "key1": "value1",
            "nested_key1-key2": "nested_value1",
            "nested_key2-key2": "nested_value2",
            "deeply_nested_key-nested_key3-key3": "deeply_nested_value",
        }
    """
    unnested_dict = {}
    for key, value in dictionary.items():
        new_key = f"{key}-{parent_key}" if parent_key else key
        if isinstance(value, dict):
            unnested_dict.update(unnest_dictionary(value, parent_key=new_key))
        else:
            unnested_dict[new_key] = value
    return unnested_dict


def get_torch_device(use_gpu: bool = True):
    if use_gpu and torch.cuda.is_available():
        return "cuda:0"
    elif use_gpu and torch.backends.mps.is_available():
        return "cpu"  # "mps" is not working for me..
    else:
        return "cpu"


def convert_to_nested_list(lst, sublist_size: int):
    """
    Example:
    >>> list_ = [0, 0, 1, 1, 0, 0]
    >>> convert_to_nested_list(list_,3)
        [[0, 0, 1], [1, 0, 0]]
    """
    nested_list = [lst[i : i + sublist_size] for i in range(0, len(lst), sublist_size)]
    return nested_list


def repeat_by_list_values_from_matrix(
    input_array: np.array,
    matrix: np.array,
    repeats: np.array,
) -> np.array:
    """
    Example:
        >>> input = np.array([[1, 0], [0, 0]])
        >>> matrix = np.array([[7,8,9], [10,11,12]])
        >>> repeats = np.array([1, 2])
        >>> repeat_by_list_values_from_matrix(input, matrix, repeats)
            array([[[10, 11, 12],
                    [ 7,  8,  9]],
                    [[ 7,  8,  9],
                    [ 7,  8,  9]],
                    [[ 7,  8,  9],
                    [ 7,  8,  9]]])
    """
    return np.repeat(matrix[input_array], repeats=repeats, axis=0)


def create_lookup_dict(df: pl.DataFrame, key: str, value: str) -> dict:
    """
    Creates a dictionary lookup table from a Pandas-like DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame from which to create the lookup table.
        key (str): The name of the column containing the keys for the lookup table.
        value (str): The name of the column containing the values for the lookup table.

    Returns:
        dict: A dictionary where the keys are the values from the `key` column of the DataFrame
            and the values are the values from the `value` column of the DataFrame.

    Example:
        >>> df = pl.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        >>> create_lookup_dict(df, 'id', 'name')
            {1: 'Alice', 2: 'Bob', 3: 'Charlie'}
    """
    return dict(zip(df[key], df[value]))


def create_lookup_objects(
    lookup_dictionary: dict[int, np.array], unknown_representation: str
) -> tuple[dict[int, pl.Series], np.array]:
    """Creates lookup objects for efficient data retrieval.

    This function generates a dictionary of indexes and a matrix from the given lookup dictionary.
    The generated lookup matrix has an additional row based on the specified unknown representation
    which could be either zeros or the mean of the values in the lookup dictionary.

    Args:
        lookup_dictionary (dict[int, np.array]): A dictionary where keys are unique identifiers (int)
            and values are some representations which can be any data type, commonly used for lookup operations.
        unknown_representation (str): Specifies the method to represent unknown entries.
            It can be either 'zeros' to represent unknowns with a row of zeros, or 'mean' to represent
            unknowns with a row of mean values computed from the lookup dictionary.

    Raises:
        ValueError: If the unknown_representation is not either 'zeros' or 'mean',
            a ValueError will be raised.

    Returns:
        tuple[dict[int, pl.Series], np.array]: A tuple containing two items:
            - A dictionary with the same keys as the lookup_dictionary where values are polars Series
                objects containing a single value, which is the index of the key in the lookup dictionary.
            - A numpy array where the rows correspond to the values in the lookup_dictionary and an
                additional row representing unknown entries as specified by the unknown_representation argument.

    Example:
    >>> data = {
            10: np.array([0.1, 0.2, 0.3]),
            20: np.array([0.4, 0.5, 0.6]),
            30: np.array([0.7, 0.8, 0.9]),
        }
    >>> lookup_dict, lookup_matrix = create_lookup_objects(data, "zeros")

    >>> lookup_dict
        {10: shape: (1,)
            Series: '' [i64]
            [
                    1
            ], 20: shape: (1,)
            Series: '' [i64]
            [
                    2
            ], 30: shape: (1,)
            Series: '' [i64]
            [
                    3
        ]}
    >>> lookup_matrix
        array([[0. , 0. , 0. ],
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]])
    """
    # MAKE LOOKUP DICTIONARY
    lookup_indexes = {
        id: pl.Series("", [i]) for i, id in enumerate(lookup_dictionary, start=1)
    }
    # MAKE LOOKUP MATRIX
    lookup_matrix = np.array(list(lookup_dictionary.values()))

    if unknown_representation == "zeros":
        UNKNOWN_ARRAY = np.zeros(lookup_matrix.shape[1], dtype=lookup_matrix.dtype)
    elif unknown_representation == "mean":
        UNKNOWN_ARRAY = np.mean(lookup_matrix, axis=0, dtype=lookup_matrix.dtype)
    else:
        raise ValueError(
            f"'{unknown_representation}' is not a specified method. Can be either 'zeros' or 'mean'."
        )

    lookup_matrix = np.vstack([UNKNOWN_ARRAY, lookup_matrix])
    return lookup_indexes, lookup_matrix


def batch_items_generator(items: Iterable[any], batch_size: int):
    """
    Generator function that chunks a list of items into batches of a specified size.

    Args:
        items (list): The list of items to be chunked.
        batch_size (int): The number of items to include in each batch.

    Yields:
        list: A batch of items from the input list.

    Examples:
        >>> items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> batch_size = 3
        >>> for batch in chunk_list(items, batch_size):
        ...     print(batch)
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
