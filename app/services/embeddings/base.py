from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    """Abstract base class representing an Embedding provider."""

    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a list of string chunks.

        Args:
            texts: List of string chunks to embed.

        Returns:
            List of vector embeddings (each is a list of floats).
        """
        pass

    @abstractmethod
    async def get_query_embedding(self, text: str) -> List[float]:
        """
        Generate a single vector embedding for a query string.

        Args:
            text: The query string to embed.

        Returns:
            The vector embedding of the query.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the generated vectors."""
        pass
