import logging
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from memory import GeneratorMemory
from prompts import GENERATOR_PROMPT, format_context

logger = logging.getLogger("egp.generator")


class GeneratorAgent:
    """Writes an answer from retrieved extracts, never from its own knowledge."""

    def __init__(self, llm: ChatOpenAI, memory: GeneratorMemory) -> None:
        self.memory = memory
        self.chain: Runnable = GENERATOR_PROMPT | llm | StrOutputParser()

    def as_runnable(self) -> Runnable:
        """LCEL step that adds `answer` to the workflow state."""
        return RunnablePassthrough.assign(answer=self._produce)

    def _produce(self, state: Dict[str, Any]) -> str:
        question = state["question"]
        previous = self.memory.last_attempt(question)
        answer = self.chain.invoke(
            {
                "context": format_context(state["documents"]),
                "question": question,
                "previous_answer": previous["answer"] if previous else "",
                "feedback": state.get("feedback", ""),
            }
        )
        answer = answer.strip()
        self.memory.record_attempt(
            attempt=state["attempt"],
            question=question,
            answer=answer,
            sources=[f"{doc.metadata.get('source')}" for doc in state["documents"]],
            feedback_used=state.get("feedback", ""),
        )
        logger.info("generator attempt %s | %s characters", state["attempt"], len(answer))
        return answer
