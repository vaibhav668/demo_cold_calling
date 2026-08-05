import asyncio
import os
from typing import List
from app.core.logging import logger
from app.services.embeddings.base import EmbeddingProvider

class BGEM3EmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using sentence-transformers.
    """

    _model_instance = None
    _model_lock = asyncio.Lock()
    _dimension = 384

    def __init__(self) -> None:
        from app.core.config import check_low_memory
        low_mem = check_low_memory()

        configured_model = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2" if low_mem else "BAAI/bge-m3")
        if low_mem and "bge-m3" in configured_model.lower():
            logger.warning(
                f"[EMBEDDINGS] Low-memory environment detected. Overriding heavy model '{configured_model}' "
                f"to lightweight 'all-MiniLM-L6-v2' (384-d, ~30MB RAM) to prevent OOM."
            )
            self.model_name = "all-MiniLM-L6-v2"
        else:
            self.model_name = configured_model

        BGEM3EmbeddingProvider._dimension = 1024 if "bge-m3" in self.model_name else 384

    @classmethod
    async def _get_model(cls, model_name: str):
        if cls._model_instance is not None:
            return cls._model_instance

        async with cls._model_lock:
            if cls._model_instance is not None:
                return cls._model_instance

            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"[EMBEDDINGS] Initializing model '{model_name}' locally (CPU)...")
                def load():
                    return SentenceTransformer(model_name, device="cpu")
                
                cls._model_instance = await asyncio.get_event_loop().run_in_executor(None, load)
                logger.info(f"[EMBEDDINGS] SentenceTransformer model '{model_name}' loaded successfully.")
                try:
                    import psutil
                    rss = psutil.Process().memory_info().rss / (1024 * 1024)
                    logger.info(f"[MEMORY] Embedding Model loaded: RSS {rss:.2f} MB")
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"[EMBEDDINGS] Failed to initialize local model: {e}. Mock fallback enabled.")
                cls._model_instance = "FAILED"
            return cls._model_instance

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        model = await self._get_model(self.model_name)
        if model != "FAILED" and model is not None:
            try:
                def encode_texts():
                    import torch
                    with torch.inference_mode():
                        raw_vecs = model.encode(texts, normalize_embeddings=True)
                    return [vec.tolist() for vec in raw_vecs]

                return await asyncio.get_event_loop().run_in_executor(None, encode_texts)
            except Exception as e:
                logger.error(f"[EMBEDDINGS] Local embedding encoding failed: {e}")

        return self._mock_embeddings(texts)

    async def get_query_embedding(self, text: str) -> List[float]:
        embeddings = await self.get_embeddings([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        return self._dimension

    def _mock_embeddings(self, texts: List[str]) -> List[List[float]]:
        import random
        dim = self.dimension
        logger.warning(f"[EMBEDDINGS] SentenceTransformer failed. Returning {dim}-d mock embeddings...")
        results = []
        for text in texts:
            random.seed(len(text))
            results.append([random.uniform(-1.0, 1.0) for _ in range(dim)])
        return results
