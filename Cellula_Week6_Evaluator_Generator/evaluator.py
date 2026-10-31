import logging
import re
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from memory import EvaluatorMemory
from prompts import EVALUATOR_PROMPT, format_context

logger = logging.getLogger("egp.evaluator")


def _number(value: Any) -> float:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return min(max(float(match.group()), 0.0), 10.0) if match else 0.0


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


class EvaluatorAgent:
    """Judges the answer against the extracts and returns a structured decision."""

    def __init__(self, llm: ChatOpenAI, memory: EvaluatorMemory) -> None:
        self.memory = memory
        self.chain: Runnable = EVALUATOR_PROMPT | llm | JsonOutputParser()

    def as_runnable(self) -> Runnable:
        """LCEL step that adds `review` to the workflow state."""
        return RunnablePassthrough.assign(review=self._judge)

    def _judge(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        raw = self.chain.invoke(
            {
                "history": self.memory.history_for(question),
                "context": format_context(state["documents"]),
                "question": question,
                "answer": state["answer"],
            }
        )
        review = self._normalise(raw)
        self.memory.record_evaluation(
            attempt=state["attempt"],
            question=question,
            answer=state["answer"],
            review=review,
        )
        logger.info(
            "evaluator attempt %s | acceptable=%s score=%s",
            state["attempt"],
            review["acceptable"],
            review["score"],
        )
        return review

    @staticmethod
    def _normalise(raw: Dict[str, Any]) -> Dict[str, Any]:
        if "acceptable" not in raw:
            raise ValueError("Evaluator response has no 'acceptable' field.")
        feedback = str(raw.get("feedback") or "").strip()
        if not _flag(raw["acceptable"]) and not feedback:
            feedback = "Rewrite the answer using only facts stated in the extracts."
        return {
            "acceptable": _flag(raw["acceptable"]),
            "score": _number(raw.get("score", 0)),
            "relevance": _number(raw.get("relevance", 0)),
            "completeness": _number(raw.get("completeness", 0)),
            "grounding": _number(raw.get("grounding", 0)),
            "feedback": feedback,
        }
