"""
To run the tests just call: 
$ pytest -s -v 
in the projects root directory.
"""

from numpy import nan as NaN, isnan
from math import isclose

import pandas as pd
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
    ff = function_finder.FunctionFinder(test_mode=True)
    ff.ideal_set = ideal_df
    ff.train_set = train_df

    nina = ff.get_best_function()

    assert len(nina) == 1
    nina = nina[0]
    assert nina["train_y"] == 1
    assert nina["best_ideal_y"] == 2


def test_function_finder_test_multiple_test_functions():
    """
    Basically the same test as above just with more training functions:
    The following values are expected:
    [{'train_y': 1, 'best_ideal_y': 2},
    {'train_y': 2, 'best_ideal_y': 3}]
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
    ff = function_finder.FunctionFinder(
        train_set=train_df, ideal_set=ideal_df, test_mode=True
    )

    nina = ff.get_best_function()

    assert len(nina) == 2

    assert nina[0]["train_y"] == 1
    assert nina[0]["best_ideal_y"] == 2
    assert nina[1]["train_y"] == 2
    assert nina[1]["best_ideal_y"] == 3

    print(nina)


def test_calculate_test_scores():
    """
    The ideal function is obviously y2.

    Since sqrt(2) is very roughly equal to 1.4, I just check the
    edge cases: The Deviation of Ideal 2 and Test 3.5 is 1.5 ==> Test Data is not mappable
    A deviation of 1.4 is just below the cut-off value and is therefore mappable.

    Not all x values have a corresponding test value.
    In those cases the deviation is NaN and obviously not mappable.

                                    || ---- Expected Result ----
    TRAIN | ----IDEAL---- |  TEST   || mappable to df | deviation
    x  y1 | x  y1  y2  y3 |  x  y   ||                |
    1   2 | 1   1   2   4 |  -  -   ||     False      |   NaN
    2   2 | 2   1   2   4 |  2 3.5  ||     False      |   1.5
    3   2 | 3   1   2   4 |  3 3.4  ||     True       |   1.4
    4   2 | 4   1   2   4 |  4 0.5  ||     False      |   1.5
    5   2 | 5   1   2   4 |  5  2   ||     True       |    0
    6   2 | 6   1   2   4 |  6 2.4  ||     True       |   0.4
    """
    gold_df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "IDEAL_y2": [2, 2, 2, 2, 2, 2],
            "test_deviation_to_Train_1_IDEAL_2": [NaN, 1.5, 1.4, 1.5, 0, 0.4],
            "mappable_to_Train_1_IDEAL_2": [False, False, True, False, True, True],
        },
        columns=[
            "x",
            "IDEAL_y2",
            "test_deviation_to_Train_1_IDEAL_2",
            "mappable_to_Train_1_IDEAL_2",
        ],
    )

    train_df = pd.DataFrame(
        [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2]],
        columns=["x", "y1"],
    )
    ideal_df = pd.DataFrame(
        [
            [1, 1, 2, 4],
            [2, 1, 2, 4],
            [3, 1, 2, 4],
            [4, 1, 2, 4],
            [5, 1, 2, 4],
            [6, 1, 2, 4],
        ],
        columns=["x", "y1", "y2", "y3"],
    )
    test_df = pd.DataFrame(
        [[2, 3.5], [3, 3.4], [4, 0.5], [5, 2], [6, 2.4]],
        columns=["x", "y"],
    )

    # Initialize the Function Finder Class
    # It is possible to pass the dataframes directly or set them later
    ff = function_finder.FunctionFinder(
        train_set=train_df, ideal_set=ideal_df, test_set=test_df, test_mode=True
    )
    # pylint: disable=locally-disabled, protected-access
    ff._calculate_test_scores()
    nina = ff.merged_df

    assert nina is not None

    print(nina.columns)
    nina = nina[
        [
            "x",
            "IDEAL_y2",
            "test_deviation_to_Train_1_IDEAL_2",
            "mappable_to_Train_1_IDEAL_2",
        ]
    ]

    assert nina["mappable_to_Train_1_IDEAL_2"].equals(
        nina["mappable_to_Train_1_IDEAL_2"]
    )

    # Due to rounding errors I can't use pandas built in functions
    # for comparing floats:
    for a, b in zip(
        nina["test_deviation_to_Train_1_IDEAL_2"].values,
        gold_df["test_deviation_to_Train_1_IDEAL_2"].values,
    ):
        if isnan(a):
            assert isnan(b)
        else:
            assert isclose(a, b, rel_tol=1e-5)
