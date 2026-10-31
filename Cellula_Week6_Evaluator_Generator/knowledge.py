import json
import logging
from typing import Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from cache import RedisCache
from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

logger = logging.getLogger("egp.knowledge")


class KnowledgeBase:
    """Retrieval layer shared by the workflow, with Redis-cached searches."""

    def __init__(self, cache: RedisCache) -> None:
        self.cache = cache
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add(self, documents: List[Document], ids: List[str]) -> None:
        self.store.add_documents(documents, ids=ids)

    def search(self, question: str, k: int = TOP_K) -> Tuple[List[Document], bool]:
        key = f"retrieval:{self.cache.version('kb')}:{RedisCache.fingerprint(question, k)}"
        cached = self.cache.get(key)
        if cached is not None:
            documents = [
                Document(page_content=item["text"], metadata=item["metadata"])
                for item in cached
            ]
            logger.info("retrieval cache hit | question=%r results=%s", question, len(documents))
            return documents, True

        hits = self.store.similarity_search_with_relevance_scores(question, k=k)
        documents: List[Document] = []
        payload: List[Dict] = []
        for document, score in hits:
            metadata = {name: str(value) for name, value in document.metadata.items()}
            metadata["score"] = round(float(score), 3)
            documents.append(Document(page_content=document.page_content, metadata=metadata))
            payload.append({"text": document.page_content, "metadata": metadata})
        self.cache.set(key, payload)
        logger.info("retrieval | question=%r results=%s", question, len(documents))
        return documents, False

    def count(self) -> int:
        return self.store._collection.count()

    def is_empty(self) -> bool:
        return self.count() == 0

    def sources(self) -> Dict[str, Dict]:
        raw = self.cache.client.hgetall("kb:documents")
        return {name: json.loads(value) for name, value in raw.items()}

    def clear(self) -> None:
        ids = self.store.get()["ids"]
        if ids:
            self.store.delete(ids=ids)
        self.cache.client.delete("kb:documents")
        self.cache.bump_version("kb")
        logger.info("knowledge base cleared")
