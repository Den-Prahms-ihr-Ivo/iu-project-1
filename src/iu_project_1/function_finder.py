"""
popo
"""

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Literal, Optional
from src.iu_project_1.custom_exceptions import InvalidCallOrder, InvalidDataFrame

import pandas as pd
import numpy as np
import re


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
    __merged_df: Optional[pd.DataFrame] = None

    def __init__(self):
        # TODO: man soll es auch initialisieren können aber optional
        self.__train_df = None
        self.__ideal_df = None

    @property
    def train_set(self) -> pd.DataFrame:
        if self.__train_df is None:
            raise AttributeError("Trying to read training set before it was written.")
        else:
            return self.__train_df

    @train_set.setter
    def train_set(self, df: pd.DataFrame) -> None:
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
        if len([c for c in df.columns if re.search(r"^IDEAL_y\d+$", c)]) < 1:
            df = df.add_prefix("IDEAL_")

        self.__ideal_df = df
        # TODO: sanity checks und so weiter
        # und mergen etc

        if self.__train_df is not None:
            self._merge_dfs(left=self.__train_df, right=self.__ideal_df)

    def _merge_dfs(self, left: pd.DataFrame, right: pd.DataFrame) -> None:
        self.__merged_df = pd.merge(
            left=left,
            right=right,
            how="inner",
            left_on="x",
            right_on="IDEAL_x",
            validate="one_to_one",
        ).drop(columns="IDEAL_x")

        print(self.__merged_df)

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

        return nina


if __name__ == "__main__":
    csv = CSV()
    train_df = csv.load(Path("data/train.csv"))
    ideal_df = csv.load(Path("data/ideal.csv"))

    print("PENIS")
