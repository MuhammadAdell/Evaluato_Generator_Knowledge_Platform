import uuid
import streamlit as st
import logging_setup
from cache import RedisCache
from config import LLM_MODEL, MAX_ITERATIONS, ROOT, SUPPORTED_EXTENSIONS, UPLOAD_DIR
from ingest import IngestionPipeline
from knowledge import KnowledgeBase
from workflow import EvaluatorGeneratorWorkflow

st.set_page_config(page_title="Evaluator-Generator Platform", layout="wide")


@st.cache_resource
def services():
    cache = RedisCache()
    knowledge = KnowledgeBase(cache)
    pipeline = IngestionPipeline(knowledge, cache)
    return cache, knowledge, pipeline


@st.cache_resource
def workflow_for(session_id: str):
    cache, knowledge, _ = services()
    return EvaluatorGeneratorWorkflow(knowledge, cache, session_id)


cache, knowledge, pipeline = services()

if not cache.ping():
    st.error("Redis is not running. Start it with ./third_party/redis-stable/src/redis-server --port 6379")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "history" not in st.session_state:
    st.session_state.history = []

workflow = workflow_for(st.session_state.session_id)

with st.sidebar:
    st.header("System")
    st.write(f"Model: `{LLM_MODEL}`")
    st.write(f"Redis: connected, knowledge version {cache.version('kb')}")
    st.write(f"Sources: {knowledge.count()} indexed chunks")

    metrics = cache.metrics()
    st.subheader("Monitoring")
    st.table(
        {
            "Metric": ["Cache hits", "Cache misses", "Hit rate (%)", "Generator runs", "Evaluator runs"],
            "Value": [
                str(metrics["cache_hits"]),
                str(metrics["cache_misses"]),
                f"{metrics['hit_rate']}",
                str(metrics["generator_runs"]),
                str(metrics["evaluator_runs"]),
            ],
        }
    )
    st.caption(
        f"Generator memory: {len(workflow.generator_memory.entries())} entries (Redis db 1)  \n"
        f"Evaluator memory: {len(workflow.evaluator_memory.entries())} entries (Redis db 2)"
    )

    if st.button("New conversation"):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.history = []
        st.rerun()

    if st.button("Clear knowledge base"):
        pipeline.clear()
        st.session_state.history = []
        st.rerun()

    with st.expander("Recent log lines"):
        log_file = ROOT / "logs" / "app.log"
        lines = log_file.read_text().splitlines()[-12:] if log_file.exists() else []
        st.code("\n".join(lines) or "No log entries yet.", language="text")

st.title("Evaluator-Generator knowledge platform")
st.caption(
    "Upload sources, ask a question, and let the generator and evaluator refine the answer "
    f"for up to {MAX_ITERATIONS} rounds."
)

sources_tab, ask_tab = st.tabs(["Knowledge sources", "Ask"])

with sources_tab:
    st.subheader("Upload a file")
    uploaded = st.file_uploader(
        "PDF, DOCX, TXT, code, PPTX or WAV",
        type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
    )
    if st.button("Add file", disabled=uploaded is None):
        UPLOAD_DIR.mkdir(exist_ok=True)
        target = UPLOAD_DIR / uploaded.name
        target.write_bytes(uploaded.getvalue())
        try:
            with st.spinner(f"Processing {uploaded.name}"):
                st.success(pipeline.ingest_file(target))
        except Exception as error:
            st.error(f"Could not ingest {uploaded.name}: {error}")

    st.subheader("Add a web page")
    url = st.text_input("URL", placeholder="https://example.com/article")
    if st.button("Add web page"):
        if not url.startswith(("http://", "https://")):
            st.error("Enter a full URL starting with http:// or https://.")
        else:
            try:
                with st.spinner("Reading the page"):
                    st.success(pipeline.ingest_url(url))
            except Exception as error:
                st.error(f"Could not read the page: {error}")

    st.subheader("Add a Wikipedia page")
    wiki = st.text_input("Title or link", placeholder="Retrieval-augmented generation")
    if st.button("Add Wikipedia page"):
        if not wiki.strip():
            st.error("Enter a page title first.")
        else:
            try:
                with st.spinner("Reading the article"):
                    st.success(pipeline.ingest_wikipedia(wiki))
            except Exception as error:
                st.error(f"Could not read the article: {error}")

    st.subheader("Indexed sources")
    rows = knowledge.sources()
    if rows:
        st.table(
            {
                "Source": list(rows.keys()),
                "Type": [row["type"] for row in rows.values()],
                "Chunks": [row["chunks"] for row in rows.values()],
                "Characters": [row["characters"] for row in rows.values()],
            }
        )
    else:
        st.info("No sources yet. Add one above, then ask a question.")

with ask_tab:
    question = st.text_input("Your question", placeholder="What is the refund policy?")
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.error("Type a question first.")
        else:
            with st.spinner("Generator and evaluator are working"):
                result = workflow.answer(question)
            st.session_state.history.insert(0, {"question": question.strip(), "result": result})

    for item in st.session_state.history:
        result = item["result"]
        st.markdown(f"### {item['question']}")
        st.markdown(result.answer)
        if result.accepted:
            st.success(f"Accepted after {result.iterations} round(s).")
        else:
            st.warning(result.note or "The answer was not validated.")
        if result.cache_hit:
            st.caption("This answer came from the Redis cache.")

        for step in result.trace:
            review = step["review"]
            with st.expander(
                f"Round {step['attempt']}: score {review['score']}/10, "
                f"{'accepted' if review['acceptable'] else 'rejected'}"
            ):
                st.write(step["answer"])
                first, second, third = st.columns(3)
                first.metric("Relevance", f"{review['relevance']}/10")
                second.metric("Completeness", f"{review['completeness']}/10")
                third.metric("Grounding", f"{review['grounding']}/10")
                if step["feedback_used"]:
                    st.caption(f"Feedback applied: {step['feedback_used']}")
                if review["feedback"]:
                    st.caption(f"Evaluator feedback: {review['feedback']}")
                if step["sources"]:
                    st.caption(
                        "Extracts: "
                        + ", ".join(
                            f"{source['source']} (score {source['score']})"
                            for source in step["sources"]
                        )
                    )
        st.divider()
