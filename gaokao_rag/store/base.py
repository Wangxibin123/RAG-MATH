from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np

class BaseStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def build(self, ids: List[str], vecs: np.ndarray):
        """
        Builds the index from scratch with the given IDs and vectors.
        Any existing data in the store for the same collection might be cleared.

        Args:
            ids (List[str]): A list of unique string identifiers for the vectors.
            vecs (np.ndarray): A 2D numpy array of float vectors, shape (n, dim).
        """
        pass

    @abstractmethod
    def add(self, ids: List[str], vecs: np.ndarray):
        """
        Adds new vectors to an existing index.

        Args:
            ids (List[str]): A list of unique string identifiers for the new vectors.
            vecs (np.ndarray): A 2D numpy array of new float vectors, shape (m, dim).
        """
        pass

    @abstractmethod
    def search(self, vec: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        """
        Searches for the top k most similar vectors to the given query vector.

        Args:
            vec (np.ndarray): A 1D numpy array (query vector).
            k (int): The number of nearest neighbors to return.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing two lists:
                - A list of IDs of the k nearest neighbors.
                - A list of corresponding distances/similarity scores.
        """
        pass

    # Optional: Add other common methods like delete, update, count, etc.
    # def delete(self, ids: List[str]):
    #     pass

    # def count(self) -> int:
    #     pass 