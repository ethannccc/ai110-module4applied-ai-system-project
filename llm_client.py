"""
Gemini client wrapper used by DocuBot.

Handles:
- Configuring the Gemini client from the GEMINI_API_KEY environment variable
- Naive "generation only" answers over the full docs corpus (Phase 0)
- RAG style answers that use only retrieved snippets (Phase 2)

Experiment with:
- Prompt wording
- Refusal conditions
- How strictly the model is instructed to use only the provided context
"""

import os
import google.generativeai as genai

# Central place to update the model name if needed.
# You can swap this for a different Gemini model in the future.
GEMINI_MODEL_NAME = "gemini-2.5-flash"


class GeminiClient:
    """
    Simple wrapper around the Gemini model.

    Usage:
        client = GeminiClient()
        answer = client.naive_answer_over_full_docs(query, all_text)
        # or
        answer = client.answer_from_snippets(query, snippets)
    """

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
    # Phase 0: naive generation over full docs
    # -----------------------------------------------------------

    def _generate(self, prompt):
        """
        Internal helper that wraps generate_content with error handling.
        Returns the response text, or raises RuntimeError on API failure.
        """
        try:
            response = self.model.generate_content(prompt)
            return (response.text or "").strip()
        except Exception as exc:
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    def naive_answer_over_full_docs(self, query, all_text):
        # We ignore all_text and send a generic prompt instead
        prompt = f"""
    You are a documentation assistant. 
    Answer this developer question: {query}
    """
        return self._generate(prompt)

    # -----------------------------------------------------------
    # Phase 2: RAG style generation over retrieved snippets
    # -----------------------------------------------------------

    def answer_from_snippets(self, query, snippets):
        """
        Phase 2:
        Generate an answer using only the retrieved snippets.

        snippets: list of (filename, text) tuples selected by DocuBot.retrieve

        The prompt:
        - Shows each snippet with its filename
        - Instructs the model to rely only on these snippets
        - Requires an explicit "I do not know" refusal when needed
        """

        if not snippets:
            return "I do not know based on the docs I have."

        context_blocks = []
        for filename, text in snippets:
            block = f"File: {filename}\n{text}\n"
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a cautious documentation assistant helping developers understand a codebase.

You will receive:
- A developer question
- A small set of snippets from project files

Your job:
- Answer the question using only the information in the snippets.
- If the snippets do not provide enough evidence, refuse to guess.

Snippets:
{context}

Developer question:
{query}

Rules:
- Use only the information in the snippets. Do not invent new functions,
  endpoints, or configuration values.
- If the snippets are not enough to answer confidently, reply exactly:
  "I do not know based on the docs I have."
- When you do answer, briefly mention which files you relied on.
"""

        return self._generate(prompt)

    # -----------------------------------------------------------
    # Phase 3 (Agentic): query planning, sufficiency checking, reformulation
    # -----------------------------------------------------------

    def analyze_query(self, query):
        """
        Agentic step 1: decompose the user question into focused search terms.
        Returns a refined search string the retriever can use.
        """
        prompt = f"""You are a search query analyzer for a documentation assistant.

Given a developer question, extract 2-4 specific technical keywords or short phrases
that would best retrieve relevant documentation snippets.

Return ONLY a comma-separated list of search terms. No explanation, no punctuation besides commas.

Question: {query}

Search terms:"""
        return self._generate(prompt)

    def check_sufficiency(self, query, snippets):
        """
        Agentic step 3: decide whether the retrieved snippets contain enough
        information to answer the question.

        Returns (is_sufficient: bool, reason: str).
        """
        context = "\n\n".join(f"[{fname}]\n{text}" for fname, text in snippets)
        prompt = f"""You are evaluating whether documentation snippets contain enough
information to fully answer a developer question.

Question: {query}

Snippets:
{context}

Reply with exactly one of these two formats:
SUFFICIENT: <one sentence explaining what the snippets cover>
INSUFFICIENT: <one sentence describing what specific information is missing>"""
        text = self._generate(prompt)
        is_sufficient = text.upper().startswith("SUFFICIENT")
        return is_sufficient, text

    def reformulate_query(self, original_query, snippets, gap_description):
        """
        Agentic step 4: generate new search terms targeting the identified gap.
        Returns a new search string for the next retrieval iteration.
        """
        already_checked = ", ".join(fname for fname, _ in snippets)
        prompt = f"""You are helping a documentation assistant find missing information.

Original question: {original_query}
Already retrieved from files: {already_checked}
Gap: {gap_description}

Generate 2-3 alternative technical search terms to find the missing information.
Return ONLY a comma-separated list of terms. No explanation.

Search terms:"""
        return self._generate(prompt)
