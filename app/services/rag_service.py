import os
import uuid
import asyncio
import chromadb
from typing import List, Dict, Any
from app.services.embedding_service import EmbeddingService
from app.core.logging import logger

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
        self.embedding_service = EmbeddingService()

    @classmethod
    def get_client(cls):
        if cls._client is None:
            # Ephemeral / in-memory client
            logger.info("[RAG] Initializing Ephemeral ChromaDB in-memory client...")
            cls._client = chromadb.EphemeralClient()
        return cls._client

    async def initialize_collection(self) -> None:
        """Create ChromaDB collection and seed it with demo facts if empty."""
        async with self._init_lock:
            if self._collection is not None:
                return

            client = self.get_client()
            loop = asyncio.get_event_loop()

            def load_and_seed():
                # 1. Get or create collection
                try:
                    collection = client.get_or_create_collection(name=COLLECTION_NAME)
                except Exception as e:
                    # Dimension mismatch check
                    logger.warning(f"[RAG] Dimension check: deleting collection and recreating: {e}")
                    client.delete_collection(name=COLLECTION_NAME)
                    collection = client.get_or_create_collection(name=COLLECTION_NAME)
                return collection

            self._collection = await loop.run_in_executor(None, load_and_seed)
            
            # Check if seeded
            def check_empty():
                return self._collection.count() == 0

            is_empty = await loop.run_in_executor(None, check_empty)
            if is_empty:
                logger.info("[RAG] Seeding ChromaDB with demo campaign facts...")
                # Seed facts for both campaigns
                for campaign_key, facts in DEMO_FACTS.items():
                    # Generate deterministic UUIDs for campaigns to match schemas
                    campaign_id = uuid.uuid5(uuid.NAMESPACE_DNS, campaign_key)
                    await self.index_facts(campaign_id, campaign_key, facts)

    async def index_facts(self, campaign_id: uuid.UUID, filename: str, facts: List[str]) -> None:
        client = self.get_client()
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

    async def search_knowledge(
        self,
        campaign_id: uuid.UUID,
        query: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on ChromaDB in thread pool (fixing P1 event loop blocking)."""
        if self._collection is None:
            await self.initialize_collection()

        query_vector = await self.embedding_service.get_query_embedding(query)
        loop = asyncio.get_event_loop()

        def query_chroma():
            return self._collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where={"campaign_id": str(campaign_id)}
            )

        try:
            results = await loop.run_in_executor(None, query_chroma)
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
                        "filename": meta.get("filename", ""),
                        "chunk_index": meta.get("chunk_index", 0)
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"[RAG] Failed to execute semantic search: {e}")
            return []
