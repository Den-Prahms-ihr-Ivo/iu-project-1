from typing import List, Dict, Literal
from abc import ABC, abstractmethod
from pathlib import Path


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

    @abstractmethod
    def save_test_data_to_sql(self) -> None:
        """
        TODO
        """
