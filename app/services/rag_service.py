import os
import uuid
import asyncio
import re
from typing import List, Dict, Any
from app.core.config import check_low_memory
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
        """Initialize ChromaDB only if not under strict memory constraints."""
        if check_low_memory():
            logger.info("[RAG] Low memory deployment detected. Skipping ChromaDB / Embedding initialization completely.")
            return

        async with self._init_lock:
            if self._collection is not None:
                return

            client = self.get_client()
            loop = asyncio.get_event_loop()

            def load_and_seed():
                try:
                    collection = client.get_or_create_collection(name=COLLECTION_NAME)
                except Exception as e:
                    logger.warning(f"[RAG] Dimension check: deleting collection and recreating: {e}")
                    client.delete_collection(name=COLLECTION_NAME)
                    collection = client.get_or_create_collection(name=COLLECTION_NAME)
                return collection

            self._collection = await loop.run_in_executor(None, load_and_seed)
            
            def check_empty():
                return self._collection.count() == 0

            is_empty = await loop.run_in_executor(None, check_empty)
            if is_empty:
                logger.info("[RAG] Seeding ChromaDB with demo campaign facts...")
                for campaign_key, facts in DEMO_FACTS.items():
                    campaign_id = uuid.uuid5(uuid.NAMESPACE_DNS, campaign_key)
                    await self.index_facts(campaign_id, campaign_key, facts)

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

    async def search_knowledge(
        self,
        campaign_id: uuid.UUID,
        query: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Perform semantic search, falling back to light keyword matching in low memory modes."""
        
        # 1. Low-Memory Fallback: Simple Word Overlap Matching
        if check_low_memory():
            # Clean and split query words
            query_words = set(re.findall(r'\w+', query.lower()))
            if not query_words:
                return []
            
            # Map campaign_id back to campaign key string
            campaign_key = None
            for key in ["hospital", "real_estate"]:
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
                    # Calculate simple Jaccard-like or overlap ratio score
                    score = overlap / len(query_words.union(fact_words))
                    scored_results.append({
                        "text": fact,
                        "score": score,
                        "filename": campaign_key,
                        "chunk_index": idx
                    })

            # Sort by highest match score
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            return scored_results[:limit]

        # 2. Standard ChromaDB Semantic Search Path
        if self._collection is None:
            await self.initialize_collection()

        if self.embedding_service is None:
            from app.services.embedding_service import EmbeddingService
            self.embedding_service = EmbeddingService()

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
