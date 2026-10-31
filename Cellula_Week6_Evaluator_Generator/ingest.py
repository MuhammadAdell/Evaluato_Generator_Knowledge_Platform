import hashlib
import json
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cache import RedisCache
from config import CHUNK_OVERLAP, CHUNK_SIZE
from knowledge import KnowledgeBase
from loaders import empty_check, load_file, load_url, load_wikipedia

logger = logging.getLogger("egp.ingest")


class IngestionPipeline:

    def __init__(self, knowledge: KnowledgeBase, cache: RedisCache) -> None:
        self.knowledge = knowledge
        self.cache = cache
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest_file(self, path: Path) -> str:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        seen_key = f"processed:{digest}"
        if self.cache.client.exists(seen_key):
            name = self.cache.client.get(seen_key)
            logger.info("ingest skipped (already processed) | %s", name)
            return f"{name} is already in the knowledge base and was not processed again."

        documents = load_file(path)
        message = self._store(documents, path.name)
        self.cache.client.set(seen_key, path.name)
        return message

    def ingest_url(self, url: str) -> str:
        documents = load_url(url)
        return self._store(documents, documents[0].metadata["source"])

    def ingest_wikipedia(self, value: str) -> str:
        documents = load_wikipedia(value)
        return self._store(documents, documents[0].metadata["source"])

    def clear(self) -> None:
        self.knowledge.clear()
        self.cache.delete_prefix("processed:")
        self.cache.delete_prefix("retrieval:")
        self.cache.delete_prefix("answer:")
        logger.info("ingestion state and cached results cleared")

    def _store(self, documents: List[Document], source_name: str) -> str:
        empty_check(documents, source_name)
        self.knowledge.store.delete(where={"source": source_name})

        chunks = self.splitter.split_documents(documents)
        ids = [
            RedisCache.fingerprint(source_name, chunk.page_content, index)
            for index, chunk in enumerate(chunks)
        ]
        self.knowledge.add(chunks, ids)

        version = self.cache.bump_version("kb")
        self.cache.client.hset(
            "kb:documents",
            source_name,
            json.dumps(
                {
                    "type": chunks[0].metadata.get("type", "unknown"),
                    "chunks": len(chunks),
                    "characters": sum(len(chunk.page_content) for chunk in chunks),
                    "version": version,
                }
            ),
        )
        logger.info(
            "ingested | source=%s sections=%s chunks=%s version=%s",
            source_name,
            len(documents),
            len(chunks),
            version,
        )
        return f"Added {source_name}: {len(chunks)} chunks from {len(documents)} sections."
