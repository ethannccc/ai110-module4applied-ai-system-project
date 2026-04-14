"""
CLI runner for the DocuBot tinker activity.

Supports four modes:
1. Naive LLM generation over all docs (Phase 0)
2. Retrieval only (Phase 1)
3. RAG: retrieval plus LLM generation (Phase 2)
4. Agentic RAG: plan → retrieve → self-check → refine (Phase 3)
"""

from dotenv import load_dotenv
load_dotenv()

from docubot import DocuBot
from llm_client import GeminiClient
from dataset import SAMPLE_QUERIES


# Each entry is (requires_llm, display_label).
# Adding a new mode only requires adding a row here and a handler in HANDLERS.
MODES = [
    (True,  "Naive LLM over full docs (no retrieval)"),
    (False, "Retrieval only (no LLM)"),
    (True,  "RAG (retrieval + LLM)"),
    (True,  "Agentic RAG (plan → retrieve → self-check → refine)"),
]


def try_create_llm_client():
    try:
        return GeminiClient(), True
    except RuntimeError as exc:
        print(f"Warning: LLM features are disabled. Reason: {exc}")
        print("You can still run retrieval only mode.\n")
        return None, False


def choose_mode(has_llm):
    print("Choose a mode:")
    for i, (requires_llm, label) in enumerate(MODES, 1):
        suffix = " (unavailable, no GEMINI_API_KEY)" if requires_llm and not has_llm else ""
        print(f"  {i}) {label}{suffix}")
    print("  q) Quit")
    return input("Enter choice: ").strip().lower()


def get_query_or_use_samples():
    print("\nPress Enter to run built-in sample queries.")
    custom = input("Or type a single custom query: ").strip()
    return ([custom], "custom query") if custom else (SAMPLE_QUERIES, "sample queries")


def _run_queries(mode_label, answer_fn, answer_header="Answer:"):
    """Shared loop: fetch queries, call answer_fn for each, print results."""
    queries, source_label = get_query_or_use_samples()
    print(f"\nRunning {mode_label} on {source_label}...\n")
    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        try:
            answer = answer_fn(query)
        except RuntimeError as exc:
            answer = f"[Error] {exc}"
        print(answer_header)
        print(answer)
        print()


def _llm_unavailable(mode_name):
    print(f"\n{mode_name} is not available (no GEMINI_API_KEY).\n")


def run_naive_llm_mode(bot, has_llm):
    if not has_llm:
        _llm_unavailable("Naive LLM mode")
        return
    all_text = bot.full_corpus_text()
    _run_queries("naive LLM mode",
                 lambda q: bot.llm_client.naive_answer_over_full_docs(q, all_text))


def run_retrieval_only_mode(bot, _has_llm):
    _run_queries("retrieval only mode", bot.answer_retrieval_only,
                 answer_header="Retrieved snippets:")


def run_rag_mode(bot, has_llm):
    if not has_llm:
        _llm_unavailable("RAG mode")
        return
    _run_queries("RAG mode", bot.answer_rag)


def run_agentic_mode(bot, has_llm):
    if not has_llm:
        _llm_unavailable("Agentic RAG mode")
        return
    _run_queries("agentic RAG mode", bot.answer_agentic)


HANDLERS = [run_naive_llm_mode, run_retrieval_only_mode, run_rag_mode, run_agentic_mode]


def main():
    print("DocuBot Tinker Activity")
    print("=======================\n")

    llm_client, has_llm = try_create_llm_client()
    bot = DocuBot(llm_client=llm_client)

    while True:
        choice = choose_mode(has_llm)
        if choice == "q":
            print("\nGoodbye.")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(HANDLERS):
            HANDLERS[int(choice) - 1](bot, has_llm)
        else:
            print(f"\nUnknown choice. Please pick 1–{len(HANDLERS)} or q.\n")


if __name__ == "__main__":
    main()
