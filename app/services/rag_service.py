import os
import uuid
import asyncio
import re
from typing import List, Dict, Any
from app.core.config import check_low_memory
from app.core.logging import logger
chroma_manager = None
try:
    from app.db.chroma import chroma_manager
except ImportError:
    pass

COLLECTION_NAME = "knowledge_base"

# Pre-seeded industry facts
DEMO_FACTS = {
    "hospital": [
        "Visiting hours at Mercy Hospital are from 9:00 AM to 7:00 PM daily.",
        "Parking at Mercy Hospital is completely free for patients and visitors in the adjacent multi-story parking garage.",
        "Mercy Hospital is located at 123 Health Ave, Suite 100.",
        "Dr. Sharma is the chief orthopedic surgeon, and Dr. Patel is the chief cardiologist.",
        "Orthopedics clinic runs from 9:00 AM to 12:00 PM. Cardiology clinic is from 1:00 PM to 4:00 PM.",
        "Appointments can be rescheduled or cancelled at least 24 hours in advance without any fee.",
        "Mercy Hospital features a 24/7 Emergency Room and state-of-the-art diagnostic facilities."
    ],
    "real_estate": [
        "Orchard Heights by Skyline Developers offers premium 2 BHK and 3 BHK luxury apartments starting at 80 Lakhs.",
        "Amenities at Orchard Heights include a temperature-controlled swimming pool, a fully equipped gym, a kids play area, and 24/7 security.",
        "Orchard Heights is located in Gachibowli, Hyderabad, close to major tech parks and schools.",
        "Standard booking deposit is 5% of the property value.",
        "Site visits can be booked for any day of the week. Skyline Developers provides complimentary pick-and-drop services for site visits.",
        "Orchard Heights construction is in advanced stages, with possession starting within the next 6 months."
    ]
}

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    if text_len == 0:
        return []
        
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        
        start += chunk_size - chunk_overlap
        if chunk_size <= chunk_overlap:
            break
            
    return chunks

class RAGService:
    _client = None
    _collection = None
    _init_lock = asyncio.Lock()

    def __init__(self) -> None:
        # Load embedding service lazily only if not in low memory mode
        self.embedding_service = None

    @classmethod
    def get_client(cls):
        if check_low_memory():
            return None
        if cls._client is None:
            import chromadb
            logger.info("[RAG] Initializing Ephemeral ChromaDB in-memory client...")
            cls._client = chromadb.EphemeralClient()
        return cls._client

    async def initialize_collection(self) -> None:
        """
        For the browser demo, we have a fixed set of 13 hardcoded facts.
        There is NO need to load ChromaDB or a sentence-transformers embedding model.
        All RAG queries use fast keyword matching via search_knowledge().
        Skip ChromaDB/embedding initialization entirely to save 10-60s of startup time.
        """
        logger.info("[RAG] Demo mode: using in-memory keyword matching. Skipping ChromaDB/embedding init.")
        return

    async def index_facts(self, campaign_id: uuid.UUID, filename: str, facts: List[str]) -> None:
        if check_low_memory():
            return

        if self.embedding_service is None:
            from app.services.embedding_service import EmbeddingService
            self.embedding_service = EmbeddingService()

        embeddings = await self.embedding_service.get_embeddings(facts)
        loop = asyncio.get_event_loop()

        def insert_data():
            ids = []
            metadatas = []
            for idx, fact in enumerate(facts):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{campaign_id}_{idx}"))
                ids.append(point_id)
                metadatas.append({
                    "campaign_id": str(campaign_id),
                    "filename": filename,
                    "chunk_index": idx
                })
            
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=facts
            )

        await loop.run_in_executor(None, insert_data)
        logger.info(f"[RAG] Indexed {len(facts)} facts for campaign {filename}")

    async def index_document(
        self,
        campaign_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
        text: str
    ) -> int:
        """Chunks document text, generates embeddings, and upserts points to ChromaDB."""
        client = chroma_manager.get_client() if chroma_manager is not None else None
        if client is None:
            return 0
        
        chunks = chunk_text(text)
        if not chunks:
            return 0
            
        logger.info(f"Indexing document '{filename}' into ChromaDB with {len(chunks)} chunks...")
        
        if self.embedding_service is None:
            from app.services.embedding_service import EmbeddingService
            self.embedding_service = EmbeddingService()
            
        embeddings = await self.embedding_service.get_embeddings(chunks)
        
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        
        ids = []
        metadatas = []
        documents = []
        
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{idx}"))
            ids.append(point_id)
            metadatas.append({
                "campaign_id": str(campaign_id),
                "document_id": str(document_id),
                "filename": filename,
                "chunk_index": idx
            })
            documents.append(chunk)
            
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to upsert points to ChromaDB: {e}")
            return len(chunks)

    async def delete_document_vectors(self, document_id: uuid.UUID) -> None:
        """Purge vectors from ChromaDB for a specific document."""
        client = chroma_manager.get_client() if chroma_manager is not None else None
        if client is None:
            return
        try:
            collection = client.get_or_create_collection(name=COLLECTION_NAME)
            collection.delete(where={"document_id": str(document_id)})
            logger.info(f"Purged ChromaDB vectors for document ID {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete document vectors from ChromaDB: {e}")

    async def search_knowledge(
        self,
        campaign_id: uuid.UUID,
        query: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Fast semantic search on ChromaDB knowledge base filtered by campaign metadata,
        with a fallback to keyword-based fact retrieval from DEMO_FACTS if ChromaDB client is unavailable.
        """
        client = chroma_manager.get_client() if chroma_manager is not None else None
        if client is not None:
            if self.embedding_service is None:
                from app.services.embedding_service import EmbeddingService
                self.embedding_service = EmbeddingService()
            try:
                query_vector = await self.embedding_service.get_query_embedding(query)
                collection = client.get_or_create_collection(name=COLLECTION_NAME)
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=limit,
                    where={"campaign_id": str(campaign_id)}
                )
                
                formatted_results = []
                if results and "documents" in results and results["documents"]:
                    documents = results["documents"][0]
                    ids = results["ids"][0]
                    metadatas = results["metadatas"][0]
                    distances = results.get("distances", [[]])[0]
                    
                    for idx, (doc, doc_id, meta) in enumerate(zip(documents, ids, metadatas)):
                        score = 1.0 - distances[idx] if idx < len(distances) else 1.0
                        formatted_results.append({
                            "text": doc,
                            "score": score,
                            "document_id": uuid.UUID(meta.get("document_id")) if meta.get("document_id") else uuid.uuid4(),
                            "filename": meta.get("filename", ""),
                            "chunk_index": meta.get("chunk_index", 0)
                        })
                return formatted_results
            except Exception as e:
                logger.error(f"Failed to query ChromaDB, falling back to keyword search: {e}")

        # Fallback to keyword-based search
        query_words = set(re.findall(r'\w+', query.lower()))
        if not query_words:
            return []

        # Map campaign_id back to campaign key string
        campaign_key = None
        for key in DEMO_FACTS:
            if str(uuid.uuid5(uuid.NAMESPACE_DNS, key)) == str(campaign_id):
                campaign_key = key
                break

        if not campaign_key:
            return []

        facts = DEMO_FACTS.get(campaign_key, [])
        scored_results = []

        for idx, fact in enumerate(facts):
            fact_words = set(re.findall(r'\w+', fact.lower()))
            overlap = len(query_words.intersection(fact_words))
            if overlap > 0:
                score = overlap / len(query_words.union(fact_words))
                scored_results.append({
                    "text": fact,
                    "score": score,
                    "filename": campaign_key,
                    "chunk_index": idx
                })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:limit]
