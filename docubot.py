"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
- Supporting agentic RAG when paired with Gemini (Phase 3)
"""

import os
import glob

NOT_FOUND = "I do not know based on these docs."


class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client
        self.documents = self.load_documents()  # list of (filename, text) chunks
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of (filename, chunk_text) tuples.
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith((".md", ".txt")):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                for chunk in text.split("\n\n"):
                    chunk = chunk.strip()
                    if chunk:
                        docs.append((filename, chunk))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        index = {}
        for filename, text in documents:
            for token in text.lower().split():
                token = token.strip(".,!?;:\"'()[]{}")
                if token:
                    index.setdefault(token, [])
                    if filename not in index[token]:
                        index[token].append(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        text_lower = text.lower()
        return sum(1 for word in query.lower().split() if word in text_lower)

    def retrieve(self, query, top_k=3, min_score=2):
        """
        Score every chunk against the query and return the top_k results
        that meet min_score. Sorted by score descending.
        """
        scored = [
            (self.score_document(query, text), filename, text)
            for filename, text in self.documents
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(f, t) for score, f, t in scored if score >= min_score][:top_k]

    # -----------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------

    def _require_llm(self, mode_name):
        """Raise a clear error when a mode is called without an LLM client."""
        if self.llm_client is None:
            raise RuntimeError(
                f"{mode_name} requires an LLM client. Provide a GeminiClient instance."
            )

    def _merge_snippets(self, existing, new):
        """Return existing plus any snippets from new not already present."""
        seen = {(f, t[:50]) for f, t in existing}
        return existing + [(f, t) for f, t in new if (f, t[:50]) not in seen]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """Phase 1: return raw snippets with no LLM involved."""
        snippets = self.retrieve(query, top_k=top_k)
        if not snippets:
            return NOT_FOUND
        return "\n---\n".join(f"[{filename}]\n{text}\n" for filename, text in snippets)

    def answer_rag(self, query, top_k=3):
        """Phase 2: retrieve snippets then ask Gemini to synthesize an answer."""
        self._require_llm("RAG mode")
        snippets = self.retrieve(query, top_k=top_k)
        return self.llm_client.answer_from_snippets(query, snippets) if snippets else NOT_FOUND

    def answer_agentic(self, query, max_iterations=3):
        """
        Phase 3: agentic RAG — the bot plans and checks its own retrieval.

        1. Analyze     — LLM extracts focused search terms from the query.
        2. Retrieve    — fetch snippets using those terms.
        3. Check       — LLM decides if snippets are sufficient to answer.
                         If yes, break out of the loop.
        4. Reformulate — LLM generates better terms targeting the gap, then loop.
        5. Generate    — synthesize a final answer from all accumulated snippets.
        """
        self._require_llm("Agentic mode")

        search_query = self.llm_client.analyze_query(query)
        all_snippets = []

        for iteration in range(max_iterations):
            all_snippets = self._merge_snippets(all_snippets, self.retrieve(search_query))

            if not all_snippets:
                return NOT_FOUND

            is_sufficient, reason = self.llm_client.check_sufficiency(query, all_snippets)
            if is_sufficient:
                break

            if iteration < max_iterations - 1:
                search_query = self.llm_client.reformulate_query(query, all_snippets, reason)

        return self.llm_client.answer_from_snippets(query, all_snippets)

    # -----------------------------------------------------------
    # Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        return "\n\n".join(text for _, text in self.documents)
