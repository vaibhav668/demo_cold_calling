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
        "CityCare Hospital is located at 123 Health Ave, Suite 100, City Center, Mumbai, Maharashtra 400001, near Central Park Metro Station.",
        "Contact phone number for CityCare Hospital is +91 22 5550 1234, email is info@citycarehospital.com, and website is www.citycarehospital.com.",
        "OPD working hours and hospital timings at CityCare Hospital are from 8:00 AM to 8:00 PM daily. The Emergency Room operates 24/7.",
        "CityCare Hospital supports English, Hindi, and Telugu languages.",
        "Dr. Sharma is the Chief Orthopedic Surgeon at CityCare Hospital specializing in orthopedic surgery, knee and hip joint replacement, fracture care, spine surgery, and sports medicine with 15+ years of experience, MBBS MS Orthopedics. Specialization: Orthopedics.",
        "Dr. Sharma's clinic is in Room 102. Consultation fee for Dr. Sharma is ₹800. Working hours are Monday to Saturday from 9:00 AM to 1:00 PM.",
        "Dr. Patel is the Chief Cardiologist at CityCare Hospital with 12+ years of experience, MBBS MD DM Cardiology. He specializes in cardiovascular care, heart failure, hypertension, ECG, Echo, angioplasty, and pacemaker clinic.",
        "Dr. Patel's clinic is in Room 204. Consultation fee for Dr. Patel is ₹1000. Working hours are Monday to Saturday from 1:00 PM to 5:00 PM.",
        "General OPD consultation fee at CityCare Hospital is ₹500.",
        "Dr. Mehta is the chief pediatrician in Room 105 specializing in child healthcare and vaccinations. Consultation fee is ₹600.",
        "Cancellation policy: Appointments can be rescheduled or cancelled at least 24 hours in advance without any fee or cancellation charges.",
        "Rescheduling policy allows free appointment rescheduling up to 24 hours prior to the scheduled appointment.",
        "Advance booking is available online or via telephone call. Walk-in appointments are accepted subject to token availability.",
        "Required documents for appointment check-in are government ID proof, previous medical prescriptions or diagnostic reports, and insurance card if applicable.",
        "ECG test fee at CityCare Hospital is ₹400 with instant report delivery.",
        "The in-house diagnostic laboratory operates from 7:00 AM to 9:00 PM daily. Home sample collection is available.",
        "The in-house pharmacy is open 24/7 on the ground floor. Valid prescription is required for medicines. Home delivery is available.",
        "Cashless insurance is supported for major insurance providers including Star Health, HDFC ERGO, ICICI Lombard, Max Bupa, Care Health, and Bajaj Allianz.",
        "Accepted payment methods are Cash, Credit Card, Debit Card, UPI, and Net Banking.",
        "Parking at CityCare Hospital is completely free for patients and visitors in the adjacent multi-story parking garage.",
        "CityCare Hospital features a 24/7 Emergency Room, ICU, state-of-the-art diagnostic facilities, and dedicated 24/7 ambulance service (+91 22 5550 9999).",
        "The hospital cafeteria is located on the ground floor and is open from 7:00 AM to 10:00 PM daily."
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

        # Fallback to keyword-based search with stop-word filtering and stem matching
        STOP_WORDS = {
            "what", "where", "when", "does", "do", "is", "are", "in", "the", "a", "an", "at", "for", "of",
            "to", "you", "have", "i", "can", "it", "this", "that", "there", "if", "or", "and", "happens",
            "on", "by", "with", "from", "my", "your", "our", "any", "some", "available", "availability",
            "here", "there", "tell", "know", "please", "like", "want", "need", "get", "give"
        }
        raw_query_words = [w for w in re.findall(r'\w+', query.lower()) if w not in STOP_WORDS and len(w) > 1]
        if not raw_query_words:
            raw_query_words = re.findall(r'\w+', query.lower())

        if not raw_query_words:
            return []

        # Map campaign_id back to campaign key string
        campaign_key = None
        for key in DEMO_FACTS:
            if str(uuid.uuid5(uuid.NAMESPACE_DNS, key)) == str(campaign_id):
                campaign_key = key
                break

        if not campaign_key:
            # Fallback based on query content or default to hospital
            q_low = query.lower()
            if any(w in q_low for w in ["bhk", "flat", "apartment", "real estate", "property", "gachibowli", "skyline"]):
                campaign_key = "real_estate"
            else:
                campaign_key = "hospital"

        facts = DEMO_FACTS.get(campaign_key, [])
        scored_results = []

        for idx, fact in enumerate(facts):
            fact_lower = fact.lower()
            fact_words = set(re.findall(r'\w+', fact_lower))
            match_score = 0.0
            
            for qw in raw_query_words:
                if qw in fact_words:
                    match_score += 2.0
                elif any((qw[:5] in fw or fw[:5] in qw) for fw in fact_words if len(fw) >= 5 and len(qw) >= 5):
                    match_score += 1.5

            if match_score > 0:
                scored_results.append({
                    "text": fact,
                    "score": match_score / (len(raw_query_words) * 2.0),
                    "filename": campaign_key,
                    "chunk_index": idx
                })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:limit]
