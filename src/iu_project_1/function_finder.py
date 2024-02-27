"""
popo
"""

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Literal, Optional, Union, Any, Tuple
from src.iu_project_1.custom_exceptions import InvalidCallOrder, InvalidDataFrame
from enum import Enum
from math import sqrt
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


class Colors(Enum):
    BACKGROUND = "#fefefe"
    LIGHT = "#ffedd2"
    HIGHLIGHT = "#d24d49"  # "#eb3e00"
    HIGHLIGHT_2 = "#567c57"  # "#294122"
    HIGHLIGHT_3 = "#dd9c7c"  # "#ffba00"
    MAPABLE = "#449aed"
    MAPABLE_LIGHT = "#add8e6"
    NOT_MAPABLE = "#d24d49"
    NOT_MAPABLE_LIGHT = "#ff7f7f"
    DARK = "#162114"
    TEXT = "#282119"


class FunctionFinderBaseClass(ABC):
    """
    Since we have to use inheritance, and I feel like the SQLAlchemy inheritance
    is cheating and I din't have a better idea, I just define all the outgoing
    interface functions in the base class.
    """

    @abstractmethod
    def get_best_function(
        self,
    ) -> List[Dict[Literal["train_y", "best_ideal_y"], int]]:
        """
        TODO
        """

    @abstractmethod
    def plot_functions(self, save_path: Path) -> None:
        """
        TODO
        """


class FunctionFinder(FunctionFinderBaseClass):
    """
    TODO
    """

    __train_df: Optional[pd.DataFrame] = None
    __ideal_df: Optional[pd.DataFrame] = None
    __test_df: Optional[pd.DataFrame] = None
    __merged_df: Optional[pd.DataFrame] = None
    __ideal_dict: Optional[List[Dict[Literal["train_y", "best_ideal_y"], int]]] = None

    def __init__(
        self,
        train_set: Union[None, pd.DataFrame, Path] = None,
        ideal_set: Union[None, pd.DataFrame, Path] = None,
        test_set: Union[None, pd.DataFrame, Path] = None,
    ):
        super(FunctionFinder, self).__init__()

        if isinstance(train_set, Path):
            self.train_set = self._load_csv(train_set)
        else:
            self.train_set = train_set

        if isinstance(ideal_set, Path):
            self.ideal_set = self._load_csv(ideal_set)
        else:
            self.ideal_set = ideal_set

        if isinstance(test_set, Path):
            self.test_set = self._load_csv(test_set)
        else:
            self.test_set = test_set

    @property
    def train_set(self) -> pd.DataFrame:
        if self.__train_df is None:
            raise AttributeError("Trying to read training set before it was written.")
        else:
            return self.__train_df

    @train_set.setter
    def train_set(self, df: pd.DataFrame) -> None:
        if df is None:
            return

        self.__train_df = df
        # TODO: sanity checks und so weiter
        # und mergen etc

        if self.__ideal_df is not None:
            self._merge_dfs(left=self.__train_df, right=self.__ideal_df)

    @property
    def ideal_set(self) -> pd.DataFrame:
        if self.__ideal_df is None:
            raise AttributeError("Trying to read ideal set before it was written.")
        else:
            return self.__ideal_df

    @ideal_set.setter
    def ideal_set(self, df: pd.DataFrame) -> None:
        if df is None:
            return

        if len([c for c in df.columns if re.search(r"^IDEAL_y\d+$", c)]) < 1:
            df = df.add_prefix("IDEAL_")

        self.__ideal_df = df
        # TODO: sanity checks und so weiter
        # und mergen etc

        if self.__train_df is not None:
            self._merge_dfs(left=self.__train_df, right=self.__ideal_df)

    @property
    def test_set(self) -> pd.DataFrame:
        if self.__test_df is None:
            raise AttributeError("Trying to read test set before it was written.")
        else:
            return self.__test_df

    @test_set.setter
    def test_set(self, df: pd.DataFrame) -> None:
        if df is None:
            return

        df = df.add_prefix("TEST_")

        self.__test_df = df
        # TODO: sanity checks und so weiter

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        if not file_path.is_file():
            raise FileNotFoundError

        if not file_path.suffix == ".csv":
            print(file_path.suffix)
            raise FileNotFoundError

        return pd.read_csv(file_path)

    def _merge_dfs(self, left: pd.DataFrame, right: pd.DataFrame) -> None:
        self.__merged_df = pd.merge(
            left=left,
            right=right,
            how="inner",
            left_on="x",
            right_on="IDEAL_x",
            validate="one_to_one",
        ).drop(columns="IDEAL_x")

        self.get_best_function()

    def get_best_function(self) -> List[Dict[Literal["train_y", "best_ideal_y"], int]]:
        """
        Finds the best fitting functions, from the "Ideal Set", for every given training function
        according to the method of least squares.

        :return a List of a Dictionaries. [{'train_y': Int, 'best_ideal_y': Int}, ...]
        """
        if self.__merged_df is None:
            raise InvalidCallOrder("Please set a Training and Ideal Set first.")

        # Looking up the number of columns of the form y1 to know how many "training"
        # functions we have
        y_cols = [c for c in self.__merged_df.columns if re.search(r"^y\d+$", c)]
        # Looking up the number of columns of the form IDEAL_y1
        # this way we know how many ideal functions to check against.
        y_ideal_cols = [
            c for c in self.__merged_df.columns if re.search(r"^IDEAL_y\d+$", c)
        ]

        # If any of the above numbers is less than 1, something went wrong
        # and any further calculations are impossible.
        if len(y_cols) < 1 or len(y_ideal_cols) < 1:
            raise InvalidDataFrame("The Given DataFrames have a faulty structure.")

        # Empty List to save the return values in.
        # Every Training-Function has one ideal function.
        # PS. I know it's bad practice but I like to give my variables likeable names,
        # since programming is kinda lonely :)
        nina: List[Dict[Literal["train_y", "best_ideal_y"], int]] = []

        # Now we iterate over every training function:
        for i in range(1, len(y_cols) + 1):
            # And every ideal function
            for j in range(1, len(y_ideal_cols) + 1):
                # Subtract column ym from columnn yn to calculate the
                # residuals. Each Residual is squared afterwards.
                self.__merged_df[f"r_{i}_squared,y={j}"] = np.square(
                    self.__merged_df[f"y{i}"] - self.__merged_df[f"IDEAL_y{j}"]
                )

            # After every residual for every ideal function has been calculated,
            # we select only the squared columns for the current training function
            tmp_re = re.compile(f"^r_{i}_squared,y=\\d+$")
            sum_r_square = self.__merged_df.filter(regex=tmp_re, axis=1).agg("sum")

            # Now we sort the resulting Series to find the index of the smallest entry
            best_fit_index = str(sum_r_square.sort_values(ascending=True).index[0])
            # Since the index has the form "r_1_squared,y=2", we now extract the numbers
            # and know best fitting ideal y function.
            best_fit = [int(s) for s in re.findall(r"\d+", best_fit_index)]

            nina.append({"train_y": best_fit[0], "best_ideal_y": best_fit[1]})

        # Since every training function has one ideal function,
        # the length of the return list and the length of x columns must be the same.
        if len(y_cols) != len(nina):
            raise InvalidDataFrame("The Given DataFrames have a faulty structure.")

        self.__ideal_dict = nina

        self._calculate_test_scores()

        return nina

    def _calculate_test_scores(self) -> None:
        """
        Polulates the class-scoped dict __merged_df with the test scores
        and decides for every test entry whether it is mappable to any of
        the ideal functions.
        """
        if self.__test_df is None:
            return

        # first the test set is merged into our internal merged_df
        self.__merged_df = pd.merge(
            left=self.__merged_df,
            right=self.__test_df,
            how="left",
            left_on="x",
            right_on="TEST_x",
        )

        # Calculate the max deviation for every entry from the internal __ideal_dict
        for ideal in self.__ideal_dict:
            maximum_regression_train_to_ideal = (
                self.__merged_df[f"y{ideal['train_y']}"]
                - self.__merged_df[f"IDEAL_y{ideal['best_ideal_y']}"]
            ).max()

            cut_off_value = sqrt(2)

            # Save the deviation between the test y value and the ideal y value
            self.__merged_df[
                f"deviation_Train_{ideal['train_y']}_IDEAL_{ideal['best_ideal_y']}"
            ] = self.__merged_df.apply(
                lambda row: abs(row[f"IDEAL_y{ideal['best_ideal_y']}"] - row["TEST_y"]),
                axis=1,
            )

            # Now we save the a boolean value whether the deviation is greater than
            # our cut_off value.
            self.__merged_df[
                f"mappable_to_Train_{ideal['train_y']}_IDEAL_{ideal['best_ideal_y']}"
            ] = self.__merged_df.apply(
                lambda row: cut_off_value
                > abs(
                    maximum_regression_train_to_ideal
                    - row[
                        f"deviation_Train_{ideal['train_y']}_IDEAL_{ideal['best_ideal_y']}"
                    ]
                ),
                axis=1,
            )

    def _fit_polynomial(
        self, x: ArrayLike, y: ArrayLike, degree: int = 1
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Takes x and y values and fits a polynomial of the given degree to it.

        :param x: x values
        :param y: y values
        :param degree: The degree of the fitted polynomial
        :return (x: np.array, y: np.array, fx: np.array)
        np arrays of new interpolated x and y points, aswell as the function-value of
        the newly interpolated points at the given x values to draw nicer errors later.
        """
        # polifit gives us the coefficients of a fitting polynomial of the given degree.
        nina = np.polynomial.polynomial.polyfit(x, y, degree)

        # gives uns evenly spaced samples
        poly_plot_x = np.linspace(x[0], x[-1], num=len(x) * 10)

        fx = []
        poly_plot_y = []

        # the function value for every sample is calculated
        # A little bit overdone, since I ended up only using a degree of 1.
        for i in poly_plot_x:
            pol = 0
            for n in range(degree + 1):
                pol += nina[n] * i**n
            poly_plot_y.append(pol)

        # The same thing for every x value
        for i in x:
            pol = 0
            for n in range(degree + 1):
                pol += nina[n] * i**n
            fx.append(pol)

        return poly_plot_x, poly_plot_y, fx

    def plot_errors(self, ax: Any, x: ArrayLike, fx: ArrayLike, y: ArrayLike) -> None:
        """
        Draws error lines between an actual and ideal value for an array of predeictions

        :param ax: matplotlib subplot to draw the error-lines in.
        :param x: x values
        :param y: actual value
        :param fx: ideal value
        """
        for i, _ in enumerate(x):
            ax.plot(
                [x[i], x[i]],
                [fx[i], y[i]],
                "-",
                color=Colors.HIGHLIGHT_3.value,
                alpha=0.9,
                zorder=0,
            )

    def plot_functions(self, save_path: Path) -> None:
        """
        This function visualizes the assignment.

        :param save_path: The base path where the plots are to be saved.

        If the necessary sets are not set, an Exception is thrown.
        Necessary are Train, Test and Ideal Set.
        """
        if (
            # self.__ideal_dict is None er
            self.train_set is None
            or self.test_set is None
            or self.ideal_set is None
        ):
            raise InvalidCallOrder("Please set a Train, Ideal and Test Set first.")

        # It is important, that the function "get_best_function" was called
        # before we continue further. If this functino was called the collumn
        # "TEST_xy" can be found in the merged DataFrame.
        if (
            len([c for c in self.__merged_df.columns if re.search(r"^TEST_y\d+$", c)])
            < 1
        ):
            self.get_best_function()

        ###############################################################
        # First Plot
        ###############################################################
        # Here beginns the heart of the function
        # We iterate over the previously calculated ideal functions.
        # The dict has the form: List[Dict[Literal['train_y', 'best_ideal_y'], int]]
        # Every Training Function is mapped to its calculated ideal function.
        for f in self.__ideal_dict:
            # numpy array of the all x values
            x = self.__merged_df["x"].values
            # numpy arrays of the respectiv training y-data and the corresponding ideal y
            y = self.__merged_df[f"y{f['train_y']}"].values
            y_ideal = self.__merged_df[f"IDEAL_y{f['best_ideal_y']}"].values

            fig, axd = plt.subplot_mosaic(
                [["upper left", "right"], ["lower left", "right"]],
                figsize=(8, 8),
                gridspec_kw={"width_ratios": [2, 1], "height_ratios": [1, 1]},
                layout="tight",
            )

            fig.set_facecolor(Colors.BACKGROUND.value)
            fig.suptitle(f"Trainings-Set #{f['train_y']}", fontsize=16)

            # In this function the ideal data is interpolated to draw a nicer Line
            # and visualize the errors nicer.
            poly_plot_x, poly_plot_y, _ = self._fit_polynomial(x, y_ideal, degree=1)

            # axd['upper left'].plot(x, fx)
            # Plotting the interpolated ideal function to get a nice line
            axd["upper left"].plot(
                poly_plot_x,
                poly_plot_y,
                linewidth=1.2,
                color=Colors.HIGHLIGHT.value,
                label=f"ideal $f_{{ {f['best_ideal_y']} }}(x)$",
                zorder=2,
            )

            # plotting all training data points.
            axd["upper left"].scatter(
                x,
                y,
                c=Colors.HIGHLIGHT_2.value,
                alpha=0.5,
                s=4,
                zorder=1,
                label=f"training data$_{f['train_y']}$",
            )

            # this is just to draw the edges in the same color
            # Earlier I experimented with some wilder colors and the
            # black borders looked awful. :)
            for a in axd:
                axd[a].tick_params(
                    color=Colors.TEXT.value, labelcolor=Colors.TEXT.value
                )
                for spine in axd[a].spines.values():
                    spine.set_edgecolor(Colors.TEXT.value)

            axd["upper left"].set_title(
                f"It's a match: $y_{f['train_y']}$ ❤ $y_{{ {f['best_ideal_y']} }}$",
                color=Colors.TEXT.value,
            )
            axd["upper left"].set_xlabel("x", color=Colors.TEXT.value)
            axd["upper left"].set_ylabel("y", color=Colors.TEXT.value)
            axd["upper left"].legend()

            ###############################################################
            # Zoomed-in Plot
            ###############################################################
            # I thought it would be neat to see a zoomed-in view of the training data, its ideal
            # function and the respective error
            # Firstly the first x-index, with its corresponding y value in a particular margin
            # around 0, is selected. Just because I thought it would be nicer to zoom in here.
            x_i = np.where(abs(y - 0.5) < 1)[0]

            zoomed_in_view_value_count = 20

            # Select only the values in that particular interval.
            x = x[x_i[0] : x_i[0] + zoomed_in_view_value_count]
            y = y[x_i[0] : x_i[0] + zoomed_in_view_value_count]
            y_ideal = y_ideal[x_i[0] : x_i[0] + zoomed_in_view_value_count]

            # Again the interpolation. This time we use the third value.
            # Since I thought I couldn't be certain, that I always have a corresponding
            # ideal y value that is exactly on the interpolated line, I just calculated
            # the respective y value for every x to draw the error-lines in the next line.
            poly_plot_x, poly_plot_y, fx = self._fit_polynomial(x, y_ideal, degree=1)

            self.plot_errors(axd["lower left"], x, fx, y)
            axd["lower left"].set_title(
                f"zoomed-in view to {zoomed_in_view_value_count} values around $y=0$"
            )
            axd["lower left"].set_xlabel("x", color=Colors.TEXT.value)
            axd["lower left"].set_ylabel("y", color=Colors.TEXT.value)
            axd["lower left"].plot(
                poly_plot_x, poly_plot_y, color=Colors.HIGHLIGHT.value
            )

            ###############################################################
            # Plotting the Test Data
            ###############################################################
            # Selector for the correct column
            mappapble_string = (
                f"mappable_to_Train_{f['train_y']}_IDEAL_{f['best_ideal_y']}"
            )

            # Since we only have a few test values, only the non null rows are selected
            tmp = self.__merged_df.loc[self.__merged_df["TEST_y"].notnull()]
            # numpy array for the color mapping.
            col = [
                Colors.MAPABLE.value if x else Colors.NOT_MAPABLE.value
                for x in tmp[mappapble_string].values
            ]

            # Plotting all test values in the first plot.
            # And coloring them according to their "mappability" to the resperctive function.
            axd["upper left"].scatter(x=tmp["x"], y=tmp["TEST_y"], color=col, alpha=0.5)

            # Now select only the test data that fits in the selected zoomed-in interval
            tmp = tmp.loc[(tmp["x"] < x.max()) & (tmp["x"] > x.min())]
            tmp = tmp.loc[(tmp["TEST_y"] < y.max()) & (tmp["TEST_y"] > y.min())]

            # Since I didn't find a way to map different colors and different labes
            # to the same scatter plot, I had to draw it seperately
            mappable = tmp.loc[tmp[mappapble_string]]
            n_mappable = tmp.loc[~tmp[mappapble_string]]

            axd["lower left"].scatter(
                x=mappable["x"],
                y=mappable["TEST_y"],
                color=Colors.MAPABLE.value,
                label=f"mappable to $f_{{ {f['best_ideal_y']} }}$",
                alpha=0.5,
            )
            axd["lower left"].scatter(
                x=n_mappable["x"],
                y=n_mappable["TEST_y"],
                color=Colors.NOT_MAPABLE.value,
                label=f"not mappable to $f_{{ {f['best_ideal_y']} }}$",
                alpha=0.5,
            )

            # Training Data
            axd["lower left"].scatter(
                x, y, c=Colors.HIGHLIGHT_2.value, zorder=100, label="training data"
            )

            ###############################################################
            # Bar Charts
            ###############################################################
            functions_map = []
            values_map = []
            values_nmap = []
            colors_map = []
            colors_nmap = []

            for it in self.__ideal_dict:
                s = f"mappable_to_Train_{it['train_y']}_IDEAL_{it['best_ideal_y']}"
                t = self.__merged_df.loc[self.__merged_df["TEST_y"].notnull()]

                percentage_of_mappaple = t[s].value_counts(normalize=True)
                values_map.append(percentage_of_mappaple[True] * 100)
                values_nmap.append(percentage_of_mappaple[False] * 100)

                functions_map.append(str(it["best_ideal_y"]))

                is_current_value = f["best_ideal_y"] == it["best_ideal_y"]

                colors_map.append(
                    Colors.MAPABLE.value
                    if is_current_value
                    else Colors.MAPABLE_LIGHT.value
                )
                colors_nmap.append(
                    Colors.NOT_MAPABLE.value
                    if is_current_value
                    else Colors.NOT_MAPABLE_LIGHT.value
                )

                # f"mappable_to_Train_{f['train_y']}_IDEAL_{f['best_ideal_y']}"

            # Stacked bar chart
            axd["right"].bar(
                functions_map,
                values_map,
                color=colors_map,
                # yerr=values1_std,
                # ecolor="green",
                # error_kw=dict(lw=2, capsize=2, capthick=2),
            )

            add_value_labels(axd["right"], functions_map)
            axd["right"].bar(
                functions_map, values_nmap, color=colors_nmap, bottom=values_map
            )
            axd["right"].set_title(
                f"Mappable percentage of test\ndata to the ideal function",
                color=Colors.TEXT.value,
                fontsize=10,
            )
            axd["right"].axis("off")

            ###############################################################
            # Finalization
            ###############################################################

            axd["lower left"].legend()
            fig.tight_layout()
            plt.savefig(save_path / f"T_{f['train_y']}-f_{f['best_ideal_y']}TMP.png")


def add_value_labels(ax, ideal):
    i = 0
    for rect in ax.patches:
        x_value = rect.get_x() + rect.get_width() / 2

        ax.annotate(
            f"$f_{{{ideal[i]}}}$",  # Use `label` as label
            (x_value, 0),  # Place label at end of the bar
            xytext=(0, 5),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha="center",  # Horizontally center label
            va="bottom",
        )  # Vertically align label differently for
        i += 1
        # positive and negative values.
