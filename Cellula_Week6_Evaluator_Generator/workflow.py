import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from cache import RedisCache
from config import LLM_MODEL, MAX_ITERATIONS
from evaluator import EvaluatorAgent
from generator import GeneratorAgent
from knowledge import KnowledgeBase
from llm import build_llm
from memory import EvaluatorMemory, GeneratorMemory

logger = logging.getLogger("egp.workflow")

EMPTY_KB_ANSWER = (
    "I could not find this information in the provided sources. "
    "No knowledge sources have been added yet."
)


def short_error(error: Exception) -> str:
    text = str(error).splitlines()[0]
    return text if len(text) <= 180 else text[:180] + "..."


@dataclass
class WorkflowResult:
    answer: str
    accepted: bool
    iterations: int
    trace: List[Dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False
    note: str = ""


class EvaluatorGeneratorWorkflow:
    """Runs generator and evaluator in a loop of at most MAX_ITERATIONS rounds."""

    def __init__(self, knowledge: KnowledgeBase, cache: RedisCache, session_id: str) -> None:
        self.knowledge = knowledge
        self.cache = cache
        self.generator_memory = GeneratorMemory(cache, session_id)
        self.evaluator_memory = EvaluatorMemory(cache, session_id)
        self.last_retrieval_hit = False

        llm = build_llm(cache)
        self.generator = GeneratorAgent(llm, self.generator_memory)
        self.evaluator = EvaluatorAgent(llm, self.evaluator_memory)

        self.step = (
            RunnablePassthrough.assign(documents=RunnableLambda(self._retrieve))
            | self.generator.as_runnable()
            | self.evaluator.as_runnable()
            | RunnableLambda(self._record)
        )

    def _retrieve(self, state: Dict[str, Any]) -> List:
        documents, cache_hit = self.knowledge.search(state["question"])
        self.last_retrieval_hit = cache_hit
        return documents

    def _record(self, state: Dict[str, Any]) -> Dict[str, Any]:
        review = state["review"]
        state["trace"].append(
            {
                "attempt": state["attempt"],
                "answer": state["answer"],
                "review": review,
                "feedback_used": state.get("feedback", ""),
                "retrieval_cache_hit": self.last_retrieval_hit,
                "sources": [
                    {
                        "source": document.metadata.get("source"),
                        "page": document.metadata.get("page"),
                        "score": document.metadata.get("score"),
                    }
                    for document in state["documents"]
                ],
            }
        )
        self.cache.bump_metric("generator_runs")
        self.cache.bump_metric("evaluator_runs")
        return state

    def answer(self, question: str) -> WorkflowResult:
        question = question.strip()
        if not question:
            raise ValueError("Please type a question first.")

        if self.knowledge.is_empty():
            logger.info("question rejected: knowledge base is empty")
            return WorkflowResult(
                answer=EMPTY_KB_ANSWER,
                accepted=False,
                iterations=0,
                note="The knowledge base is empty. Add a source before asking.",
            )

        cache_key = (
            f"answer:{self.cache.version('kb')}:"
            f"{RedisCache.fingerprint(question, LLM_MODEL)}"
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("final answer cache hit | question=%r", question)
            return WorkflowResult(**cached, cache_hit=True, note="Loaded from the Redis answer cache.")

        state: Dict[str, Any] = {"question": question, "feedback": "", "attempt": 0, "trace": []}
        for attempt in range(1, MAX_ITERATIONS + 1):
            state["attempt"] = attempt
            try:
                state = self.step.invoke(state)
            except Exception as error:
                logger.error("workflow stopped on attempt %s | %s", attempt, short_error(error))
                logger.debug("full error", exc_info=True)
                return WorkflowResult(
                    answer=state.get("answer") or "The workflow failed before producing an answer.",
                    accepted=False,
                    iterations=attempt,
                    trace=state.get("trace", []),
                    note=f"Stopped after an error: {short_error(error)}",
                )
            if state["review"]["acceptable"]:
                logger.info("answer accepted on attempt %s", attempt)
                break
            state["feedback"] = state["review"]["feedback"]
            logger.info("answer rejected on attempt %s | feedback=%s", attempt, state["feedback"])

        review = state["review"]
        accepted = bool(review["acceptable"])
        result = WorkflowResult(
            answer=state["answer"],
            accepted=accepted,
            iterations=state["attempt"],
            trace=state["trace"],
            note="" if accepted else "The answer could not be fully validated within 4 iterations.",
        )
        if accepted:
            self.cache.set(
                cache_key,
                {
                    "answer": result.answer,
                    "accepted": True,
                    "iterations": result.iterations,
                    "trace": result.trace,
                },
            )
        return result
