import pandas as pd

from pathlib import Path
from src.iu_project_1 import function_finder


def test_function_finder_get_best_functions():
    """
    First Test to test the FunctionFinders Happy Path.
    The goal is to get the best fitting function.

    We create the following DataFrames:

    TRAIN | ----IDEAL----
    x  y1 | x  y1  y2  y3
    1   2 | 1   1   2   4
    2   2 | 2   1   2   4
    3   2 | 3   1   2   4
    4   2 | 4   1   2   4

    It is obvious, that ideal Function y2 is the best fit considering the method of
    "least square".
    Therefore we expect the function "get_best_function()"
    to return the following list dictionaries:

    >>> [{'train_y': 1, 'best_ideal_y': 2}]

    This tests treats the class "FunktionFinder" as a Black Box.
    But I would like to use this test to explain its inner workings:

    first the two dataframes are merged on the "x" column.
        x  y1  IDEAL_y1  IDEAL_y2  IDEAL_y3
        1   2         1         2         4
        2   2         1         2         4
        3   2         1         2         4
        4   2         1         2         4

    Then the residuals are calculated for each ideal function according to the following
    logic and saved into a seperate column:
        For i∈[1;<number of rows>], n∈[1;<number of training functions>],
        n∈[1;<number of ideal functions>] and f(x_i) = f_m

        r_i = y_n_i - f(x_i)

    Finally a Score for Each function is created Score = ∑ r_i².
    The best Score for every training function (n) is added to an array, which
    is finally returned.
    """

    train_df = pd.DataFrame(
        [[1, 2], [2, 2], [3, 2], [4, 2]],
        columns=["x", "y1"],
    )
    ideal_df = pd.DataFrame(
        [[1, 1, 2, 4], [2, 1, 2, 4], [3, 1, 2, 4], [4, 1, 2, 4]],
        columns=["x", "y1", "y2", "y3"],
    )

    # Initialize the Function Finder Class
    # It is possible to pass the dataframes directly or set them later
    ff = function_finder.Function_Finder()
    ff.ideal_set = ideal_df
    ff.train_set = train_df

    nina = ff.get_best_function()

    assert len(nina) == 1
    nina = nina[0]
    assert nina["train_y"] == 1
    assert nina["best_ideal_y"] == 2


def test_function_finder_test_multiple_test_functions():
    """
    Bitches
    """

    train_df = pd.DataFrame(
        [[1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3]],
        columns=["x", "y1", "y2"],
    )
    ideal_df = pd.DataFrame(
        [[1, 1, 2, 3.1], [2, 1, 2, 3.3], [3, 1, 2, 3.0], [4, 1, 2, 3.2]],
        columns=["x", "y1", "y2", "y3"],
    )
    print()
    ff = function_finder.Function_Finder(train_set=train_df, ideal_set=ideal_df)

    nina = ff.get_best_function()

    assert len(nina) == 2

    assert nina[0]["train_y"] == 1
    assert nina[0]["best_ideal_y"] == 2
    assert nina[1]["train_y"] == 2
    assert nina[1]["best_ideal_y"] == 3

    print(nina)


def test_read_csv():
    p = Path.cwd()
    train_path = p / "tests" / "data" / "simple_train.csv"
    ideal_path = p / "tests" / "data" / "simple_ideal.csv"

    ff = function_finder.Function_Finder(train_set=train_path, ideal_set=ideal_path)

    nina = ff.get_best_function()

    assert len(nina) == 3

    assert nina[0]["train_y"] == 1
    assert nina[0]["best_ideal_y"] == 1
    assert nina[1]["train_y"] == 2
    assert nina[1]["best_ideal_y"] == 6
    assert nina[2]["train_y"] == 3
    assert nina[2]["best_ideal_y"] == 8


def test_plot_best_function():
    """
    Bitches
    """

    train_df = pd.DataFrame(
        [[2, 4], [3, 5], [5, 7], [7, 10], [9, 15]],
        columns=["x", "y1"],
    )
    ideal_df = pd.DataFrame(
        [[2, 4.1], [3, 4.8], [5, 7.4], [7, 10.01], [9, 15.345]],
        columns=["x", "y1"],
    )

    p = Path.cwd()
    train_path = p / "data" / "train.csv"
    ideal_path = p / "data" / "ideal.csv"
    test_path = p / "data" / "test.csv"

    print()
    ff = function_finder.Function_Finder(
        train_set=train_path, ideal_set=ideal_path, test_set=test_path
    )

    nina = ff.get_best_function()
    ff.compare_functions(Path.cwd() / "tests" / "plots" / "popo.png")
    ff.plot_alpha(Path.cwd() / "tests" / "plots")

    print(nina)
