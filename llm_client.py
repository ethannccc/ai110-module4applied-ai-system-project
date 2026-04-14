"""
Gemini client wrapper used by DocuBot.

Handles:
- Configuring the Gemini client from the GEMINI_API_KEY environment variable
- Naive "generation only" answers over the full docs corpus (Phase 0)
- RAG style answers that use only retrieved snippets (Phase 2)
- Agentic helpers: query analysis, sufficiency checking, query reformulation (Phase 3)
"""

import os
import google.generativeai as genai

GEMINI_MODEL_NAME = "gemini-2.5-flash"


class GeminiClient:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it in your shell or .env file to enable LLM features."
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    # -----------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------

    def _generate(self, prompt):
        """Send a prompt to Gemini and return the response text."""
        try:
            response = self.model.generate_content(prompt)
            return (response.text or "").strip()
        except Exception as exc:
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    def _format_context(self, snippets):
        """Format a list of (filename, text) snippets into a prompt-ready block."""
        return "\n\n".join(f"[{fname}]\n{text}" for fname, text in snippets)

    # -----------------------------------------------------------
    # Phase 0: naive generation over full docs
    # -----------------------------------------------------------

    def naive_answer_over_full_docs(self, query, all_text):
        # Intentionally ignores all_text to demonstrate hallucination without grounding.
        prompt = f"You are a documentation assistant.\nAnswer this developer question: {query}"
        return self._generate(prompt)

    # -----------------------------------------------------------
    # Phase 2: RAG style generation over retrieved snippets
    # -----------------------------------------------------------

    def answer_from_snippets(self, query, snippets):
        """
        Generate an answer using only the retrieved snippets.
        Instructs the model to refuse when context is insufficient.
        """
        if not snippets:
            return "I do not know based on the docs I have."

        prompt = f"""You are a cautious documentation assistant helping developers understand a codebase.

Answer the question using only the snippets below. If the snippets are not enough, reply exactly:
"I do not know based on the docs I have."
When you do answer, briefly mention which files you relied on. Do not invent functions, endpoints, or config values.

Snippets:
{self._format_context(snippets)}

Question: {query}"""
        return self._generate(prompt)

    # -----------------------------------------------------------
    # Phase 3 (Agentic): query planning, sufficiency checking, reformulation
    # -----------------------------------------------------------

    def analyze_query(self, query):
        """
        Agentic step 1: extract focused search terms from the user question.
        Returns a comma-separated string of keywords for the retriever.
        """
        prompt = f"""Extract 2-4 specific technical keywords or short phrases from this developer question
that would best retrieve relevant documentation snippets.
Return ONLY a comma-separated list. No explanation.

Question: {query}

Search terms:"""
        return self._generate(prompt)

    def check_sufficiency(self, query, snippets):
        """
        Agentic step 3: decide whether snippets contain enough information to answer.
        Returns (is_sufficient: bool, reason: str).
        """
        prompt = f"""Evaluate whether these documentation snippets fully answer the developer question.

Question: {query}

Snippets:
{self._format_context(snippets)}

Reply with exactly one of:
SUFFICIENT: <one sentence on what the snippets cover>
INSUFFICIENT: <one sentence on what specific information is missing>"""
        text = self._generate(prompt)
        return text.upper().startswith("SUFFICIENT"), text

    def reformulate_query(self, original_query, snippets, gap_description):
        """
        Agentic step 4: generate alternative search terms targeting the identified gap.
        Returns a new comma-separated search string for the next retrieval pass.
        """
        already_checked = ", ".join(fname for fname, _ in snippets)
        prompt = f"""A documentation assistant needs to find missing information.

Original question: {original_query}
Already retrieved from: {already_checked}
Gap: {gap_description}

Generate 2-3 alternative technical search terms to find the missing information.
Return ONLY a comma-separated list. No explanation.

Search terms:"""
        return self._generate(prompt)
