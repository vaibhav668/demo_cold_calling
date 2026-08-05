from typing import List
from app.core.config import settings
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.bge_m3_provider import BGEM3EmbeddingProvider

class EmbeddingService(EmbeddingProvider):
    """
    Facade class representing the Embedding generation service.
    """
    def __init__(self) -> None:
        self.provider: EmbeddingProvider = BGEM3EmbeddingProvider()

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self.provider.get_embeddings(texts)

    async def get_query_embedding(self, text: str) -> List[float]:
        return await self.provider.get_query_embedding(text)

    @property
    def dimension(self) -> int:
        return self.provider.dimension
