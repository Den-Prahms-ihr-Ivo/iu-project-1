"""
popo
"""

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Literal, Optional, Union
from src.iu_project_1.custom_exceptions import InvalidCallOrder, InvalidDataFrame
from enum import Enum
from math import sqrt

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
    NOT_MAPABLE = "#d24d49"
    DARK = "#162114"
    TEXT = "#282119"


class DataAccess(ABC):
    """
    abstrakte KLasse ggf mit
    read csv
    safe to sql
    irgendwelche Compare funktionen?!
    Dann noch draw funktionen --> diese könntest ändern in der Vererbung.
        das wäre die einzige Abstrakte funktion.
    """

    @abstractmethod
    def load(self, file_path: Path, sep: str = ",") -> pd.DataFrame:
        """
        asd
        """


class CSV(DataAccess):
    """
    asd
    """

    def load(self, file_path: Path, sep: str = ",") -> pd.DataFrame:
        if not file_path.is_file():
            raise FileNotFoundError

        if not file_path.suffix == ".csv":
            print(file_path.suffix)
            print("POPO!!")
            raise FileNotFoundError

        tim = pd.read_csv(file_path, sep=sep)

        return tim


class Function_Finder:

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
        # TODO: merge mit

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

    def _calculate_test_scores(self):
        if self.__test_df is None:
            return

        self.__merged_df = pd.merge(
            left=self.__merged_df,
            right=self.__test_df,
            how="left",
            left_on="x",
            right_on="TEST_x",
        )

        # Für jeden Eintrag in self.__ideal_dict max deviation berechnen.
        # dann berechnen überall wo gemapped werden kann.
        for ideal in self.__ideal_dict:
            maximum_regression_train_to_ideal = (
                self.__merged_df[f"y{ideal['train_y']}"]
                - self.__merged_df[f"IDEAL_y{ideal['best_ideal_y']}"]
            ).max()

            cut_off_value = sqrt(2)
            # Jetzt habe ich das maximum.
            # Jetzt
            self.__merged_df[
                f"deviation_Train_{ideal['train_y']}_IDEAL_{ideal['best_ideal_y']}"
            ] = self.__merged_df.apply(
                lambda row: abs(row[f"IDEAL_y{ideal['best_ideal_y']}"] - row["TEST_y"]),
                axis=1,
            )

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

    def _fit_polynomial(self, x, y, degree=1):
        print(x)
        print(x[-1])
        tim = np.polynomial.polynomial.polyfit(x, y, degree)

        poly_plot_x = np.linspace(x[0], x[-1], num=len(x) * 10)

        fx = []
        poly_plot_y = []

        for i in poly_plot_x:
            pol = 0
            for n in range(degree + 1):
                pol += tim[n] * i**n
            poly_plot_y.append(pol)  # tim[0] + tim[1] * i + tim[2] * i**2)

        for i in x:
            pol = 0
            for n in range(degree + 1):
                pol += tim[n] * i**n
            fx.append(pol)  # tim[0] + tim[1] * i + tim[2] * i**2)

        return poly_plot_x, poly_plot_y, fx

    def plot_errors(self, ax, x, fx, y):

        for i, _ in enumerate(x):
            ax.plot(
                [x[i], x[i]],
                [fx[i], y[i]],
                "-",
                color=Colors.HIGHLIGHT_3.value,
                alpha=0.9,
                zorder=0,
            )
            # plt.text(x[i] + 0.1, y[i] + 0.2, f"Error {fx[i]- y[i]}")

    def _draw_polynomial(self, x, y):
        pass

    def compare_functions(self, save_path: Path):
        if self.__ideal_dict is None:
            if self.__ideal_df is not None and self.__train_df is not None:
                if self.__merged_df is None:
                    self._merge_dfs(left=self.__train_df, right=self.__ideal_df)
                self.get_best_function()
            else:
                raise InvalidCallOrder("Please set a Training and Ideal Set first.")

        # TODO: hier musst du noch weiter machen mit deinem __ideal_dict

        x = self.__merged_df["x"].values[-40:]
        y = self.__merged_df["y1"].values[-40:]  # train
        y_2 = self.__merged_df["y2"].values[-40:]  # train

        y_ideal = self.__merged_df["IDEAL_y42"].values[-40:]

        colors = np.random.rand(len(x))
        # x, fx = self._fit_polynomial(x_scadder, y_scadder, degree=degree)
        # self.plot_errors(x, fx, y_scadder)

        degree = 1

        # TODO: In Funtion auslagern:

        plt.figure(facecolor="#eeeeee")
        ax = plt.axes()
        ax.set_facecolor("#eeeeee")

        # self._draw_polynomial(x=,y=, degree=1, compare_y=None)
        # _, _, fx = self._fit_polynomial(x, y_2, degree=degree)
        # ax.plot(x, fx)

        poly_plot_x, poly_plot_y, fx = self._fit_polynomial(x, y_ideal, degree=degree)
        # plt.plot(x, fx)
        self.plot_errors(ax, x, fx, y)
        ax.plot(x, fx)
        ax.plot(poly_plot_x, poly_plot_y, color=Colors.LIGHT.value, label="Penis")

        ax.scatter(x, y, c="#0c3b2e", zorder=100, label="Popo")

        ax.legend()

        plt.savefig(save_path)

    def plot_alpha(self, save_path):
        if self.__ideal_dict is None or self.train_set is None:
            raise InvalidCallOrder("Please set a Train, Ideal and Test Set first.")

        for f in self.__ideal_dict:
            # TODO: hier musst du noch weiter machen mit deinem __ideal_dict

            x = self.__merged_df["x"].values
            y = self.__merged_df[f"y{f['train_y']}"].values  # train

            y_ideal = self.__merged_df[f"IDEAL_y{f['best_ideal_y']}"].values

            fig, axs = plt.subplots(2, 1, figsize=(6, 6))

            fig.set_facecolor(Colors.BACKGROUND.value)

            poly_plot_x, poly_plot_y, fx = self._fit_polynomial(x, y_ideal, degree=1)
            # plt.plot(x, fx)

            axs[0].plot(x, fx)
            axs[0].plot(
                poly_plot_x,
                poly_plot_y,
                linewidth=1.2,
                color=Colors.HIGHLIGHT.value,
                label=f"ideal $f_{{ {f['best_ideal_y']} }}(x)$",
                zorder=2,
            )

            axs[0].scatter(
                x,
                y,
                c=Colors.HIGHLIGHT_2.value,
                alpha=0.5,
                s=2.5,
                zorder=1,
                label=f"training data$_{f['train_y']}$",
            )

            for j in range(2):
                axs[j].tick_params(
                    color=Colors.TEXT.value, labelcolor=Colors.TEXT.value
                )
                for spine in axs[j].spines.values():
                    spine.set_edgecolor(Colors.TEXT.value)

            axs[0].set_title(
                f"It's a match: $y_{f['train_y']}$ ❤ $y_{{ {f['best_ideal_y']} }}$",
                color=Colors.TEXT.value,
            )
            axs[0].set_xlabel("x", color=Colors.TEXT.value)
            axs[0].set_ylabel("y", color=Colors.TEXT.value)
            axs[0].legend()

            # ZOOMED IN
            x_i = np.where(abs(y - 0.5) < 1)[0]

            zoomed_in_view_value_count = 20

            x = x[x_i[0] : x_i[0] + zoomed_in_view_value_count]
            y = y[x_i[0] : x_i[0] + zoomed_in_view_value_count]

            y_ideal = y_ideal[x_i[0] : x_i[0] + zoomed_in_view_value_count]
            poly_plot_x, poly_plot_y, fx = self._fit_polynomial(x, y_ideal, degree=1)
            # plt.plot(x, fx)
            self.plot_errors(axs[1], x, fx, y)
            axs[1].set_title(
                f"zoomed-in view to {zoomed_in_view_value_count} values around $y=0$"
            )
            axs[1].set_xlabel("x", color=Colors.TEXT.value)
            axs[1].set_ylabel("y", color=Colors.TEXT.value)
            axs[1].plot(x, fx)
            axs[1].plot(poly_plot_x, poly_plot_y, color=Colors.HIGHLIGHT.value)

            # TODO: Hier nur das richtige auswählen!!
            if self.__test_df is not None:

                # mappapble_string = f"mappable_to_Train_{ideal['train_y']}_IDEAL_{ideal['best_ideal_y']}"
                mappapble_string = (
                    f"mappable_to_Train_{f['train_y']}_IDEAL_{f['best_ideal_y']}"
                )

                tmp = self.__merged_df.loc[self.__merged_df["TEST_y"].notnull()]
                col = [
                    Colors.MAPABLE.value if x else Colors.NOT_MAPABLE.value
                    for x in tmp[mappapble_string].values
                ]

                axs[0].scatter(x=tmp["x"], y=tmp["TEST_y"], color=col, alpha=0.5)

                tmp = tmp.loc[(tmp["x"] < x.max()) & (tmp["x"] > x.min())]
                print("OHOO")
                print(tmp)
                tmp = tmp.loc[(tmp["TEST_y"] < y.max()) & (tmp["TEST_y"] > y.min())]
                print(tmp)
                mappable = tmp.loc[tmp[mappapble_string]]

                n_mappable = tmp.loc[~tmp[mappapble_string]]

                axs[1].scatter(
                    x=mappable["x"],
                    y=mappable["TEST_y"],
                    color=Colors.MAPABLE.value,
                    label=f"mappable to $f_{{ {f['best_ideal_y']} }}$",
                    alpha=0.5,
                )
                axs[1].scatter(
                    x=n_mappable["x"],
                    y=n_mappable["TEST_y"],
                    color=Colors.NOT_MAPABLE.value,
                    label=f"not mappable to $f_{{ {f['best_ideal_y']} }}$",
                    alpha=0.5,
                )

            # ##############

            axs[1].scatter(
                x, y, c=Colors.HIGHLIGHT_2.value, zorder=100, label="training data"
            )
            axs[1].legend()
            fig.tight_layout()
            plt.savefig(save_path / f"T_{f['train_y']}-f_{f['best_ideal_y']}.png")

    def plot_best_functions(self, safe_path: Path) -> None:
        x_scadder = self.__merged_df["x"].values[10:20]
        # y_scadder = self.__merged_df["r_1_squared,y=1"].values
        y_scadder = self.__merged_df["y3"].values[10:20]
        # y_scadder = self.__merged_df["IDEAL_y42"].values[10:20]
        colors = np.random.rand(len(x_scadder))

        # print(self.__merged_df["r_1_squared,y=1"].values)
        degree = 1

        # x, fx = self._fit_polynomial(x_scadder, y_scadder, degree=degree)
        # self.plot_errors(x, fx, y_scadder)

        x, fx = self._fit_polynomial(
            x_scadder, y_scadder, degree=degree, num=len(y_scadder) * 10
        )
        plt.plot(x, fx)

        # plt.plot(dates, values, ".r-")
        # plt.ylim((-1, 1))

        plt.scatter(x_scadder, y_scadder, c=colors, alpha=0.5)
        plt.savefig(safe_path)


if __name__ == "__main__":
    csv = CSV()
    train_df = csv.load(Path("data/train.csv"))
    ideal_df = csv.load(Path("data/ideal.csv"))

    print("PENIS")
