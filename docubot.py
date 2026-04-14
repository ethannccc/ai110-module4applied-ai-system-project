"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
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
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        Only includes results that meet min_score to avoid returning weakly
        matched chunks when the docs don't cover the question.
        """
        scored = [
            (self.score_document(query, text), filename, text)
            for filename, text in self.documents
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(filename, text) for score, filename, text in scored if score >= min_score][:top_k]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    def answer_agentic(self, query, max_iterations=3):
        """
        Phase 3 agentic RAG mode.

        The bot actively plans and checks its own work across up to
        max_iterations retrieval passes:

        1. Analyze  — ask the LLM to extract focused search terms from the query.
        2. Retrieve — fetch snippets using those terms.
        3. Check    — ask the LLM if the snippets are sufficient to answer.
                      If yes, break out of the loop.
        4. Reformulate — if not sufficient, ask the LLM for better search terms
                         targeting the identified gap, then loop back to step 2.
        5. Generate — synthesize a final answer from all accumulated snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "Agentic mode requires an LLM client. Provide a GeminiClient instance."
            )

        # Step 1: Turn the user question into retrieval-friendly search terms.
        search_query = self.llm_client.analyze_query(query)

        all_snippets = []
        seen_keys = set()  # deduplicates by (filename, first-50-chars-of-text)

        for iteration in range(max_iterations):
            # Step 2: Retrieve snippets with the current search terms.
            new_snippets = self.retrieve(search_query)

            for fname, text in new_snippets:
                key = (fname, text[:50])
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_snippets.append((fname, text))

            if not all_snippets:
                return "I do not know based on these docs."

            # Step 3: Ask the LLM whether the accumulated snippets are enough.
            is_sufficient, reason = self.llm_client.check_sufficiency(query, all_snippets)

            if is_sufficient:
                break

            # Step 4: Reformulate for the next pass, unless this was the last one.
            if iteration < max_iterations - 1:
                search_query = self.llm_client.reformulate_query(
                    query, all_snippets, reason
                )

        # Step 5: Generate the final answer from everything retrieved.
        return self.llm_client.answer_from_snippets(query, all_snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
