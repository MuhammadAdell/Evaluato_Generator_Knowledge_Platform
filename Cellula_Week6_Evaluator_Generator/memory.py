import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import redis

from cache import RedisCache


def _client(url: str, db: int) -> redis.Redis:
    parsed = urlparse(url)
    return redis.Redis(
        host=parsed.hostname,
        port=parsed.port,
        password=parsed.password,
        db=db,
        decode_responses=True,
    )


class AgentMemory:

    agent = "agent"
    db = 0
    limit = 20

    def __init__(self, cache: RedisCache, session_id: str) -> None:
        self.session_id = session_id
        self.client = _client(cache.url, self.db)

    @property
    def key(self) -> str:
        return f"memory:{self.agent}:{self.session_id}"

    def add(self, entry: Dict[str, Any]) -> None:
        self.client.rpush(self.key, json.dumps(entry))
        self.client.ltrim(self.key, -self.limit, -1)

    def entries(self) -> List[Dict[str, Any]]:
        return [json.loads(item) for item in self.client.lrange(self.key, 0, -1)]

    def clear(self) -> None:
        self.client.delete(self.key)


class GeneratorMemory(AgentMemory):
    """What the generator saw and produced: context, answers, attempts, feedback."""

    agent = "generator"
    db = 1

    def record_attempt(
        self,
        attempt: int,
        question: str,
        answer: str,
        sources: List[str],
        feedback_used: str,
    ) -> None:
        self.add(
            {
                "attempt": attempt,
                "question": question,
                "answer": answer,
                "sources": sources,
                "feedback_used": feedback_used,
            }
        )

    def last_attempt(self, question: str) -> Optional[Dict[str, Any]]:
        for entry in reversed(self.entries()):
            if entry["question"] == question:
                return entry
        return None


class EvaluatorMemory(AgentMemory):
    """What the evaluator judged: scores, decisions and feedback."""

    agent = "evaluator"
    db = 2

    def record_evaluation(
        self,
        attempt: int,
        question: str,
        answer: str,
        review: Dict[str, Any],
    ) -> None:
        self.add(
            {
                "attempt": attempt,
                "question": question,
                "answer": answer,
                "acceptable": review["acceptable"],
                "score": review["score"],
                "feedback": review["feedback"],
                "relevance": review["relevance"],
                "completeness": review["completeness"],
                "grounding": review["grounding"],
            }
        )

    def history_for(self, question: str, limit: int = 3) -> str:
        lines = [
            "attempt {}: score {}, acceptable {}, feedback: {}".format(
                entry["attempt"],
                entry["score"],
                entry["acceptable"],
                entry["feedback"] or "-",
            )
            for entry in self.entries()
            if entry["question"] == question
        ]
        return "\n".join(lines[-limit:])
