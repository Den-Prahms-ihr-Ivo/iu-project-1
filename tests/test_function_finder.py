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
    ff = function_finder.Function_Finder()
    ff.ideal_set = ideal_df
    ff.train_set = train_df

    nina = ff.get_best_function()

    assert len(nina) == 1
    nina = nina[0]
    assert nina["train_y"] == 1
    assert nina["best_ideal_y"] == 2
