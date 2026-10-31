from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions for a documentation assistant.\n"
            "Use only the extracts below. Do not add facts that are missing from them.\n"
            "Cite the source name in brackets after each sentence that uses an extract, "
            "for example [handbook.txt].\n"
            "If the extracts do not contain the answer, reply exactly: "
            "I could not find this information in the provided sources.\n"
            "When a previous answer and reviewer feedback are given, rewrite the answer "
            "and fix every point in the feedback.",
        ),
        (
            "human",
            "Extracts:\n{context}\n\n"
            "Question: {question}\n\n"
            "Previous answer:\n{previous_answer}\n\n"
            "Reviewer feedback:\n{feedback}\n\n"
            "Answer:",
        ),
    ]
)

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You review answers from a documentation assistant.\n"
            "Check the answer against the extracts and the question. Judge relevance, "
            "completeness, accuracy and grounding. Flag any claim the extracts do not support.\n"
            "Return only a JSON object with these keys:\n"
            '{{"acceptable": true or false, "score": 0-10, "relevance": 0-10, '
            '"completeness": 0-10, "grounding": 0-10, "feedback": "what to fix, empty when acceptable"}}\n'
            "Mark the answer acceptable only when every claim is supported by the extracts "
            "and the question is fully answered.\n"
            "If the answer says the extracts do not contain the information and that is true, "
            "mark it acceptable.",
        ),
        (
            "human",
            "Earlier evaluation notes:\n{history}\n\n"
            "Extracts:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer to review:\n{answer}",
        ),
    ]
)


def format_context(documents: List[Document]) -> str:
    blocks = []
    for number, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        label = f"{source}, page {page}" if page else str(source)
        blocks.append(f"[{number}] {label}\n{document.page_content}")
    return "\n\n".join(blocks) if blocks else "No extracts were retrieved."
