# DocuBot Model Card

This model card is a short reflection on your DocuBot system. Fill it out after you have implemented retrieval and experimented with all three modes:

1. Naive LLM over full docs  
2. Retrieval only  
3. RAG (retrieval plus LLM)

Use clear, honest descriptions. It is fine if your system is imperfect.

---

## 1. System Overview

**What is DocuBot trying to do?**  
Describe the overall goal in 2 to 3 sentences.

DocuBot helps developers get answers from a project's documentation without having to read through every file themselves. You ask it a question and it finds the most relevant chunks of the docs to answer you. 

**What inputs does DocuBot take?**  
For example: user question, docs in folder, environment variables.

A natural language question from the user or a folder of md or txt documentation files.

**What outputs does DocuBot produce?**

Either raw documentation snippets ranked by relevance, or a generated answer grounded in those snippets. If nothing is found, it says "I do not know based on these docs."

---

## 2. Retrieval Design

**How does your retrieval system work?**  
Describe your choices for indexing and scoring.

- How do you turn documents into an index?
- How do you score relevance for a query?
- How do you choose top snippets?

Documents are split into paragraph-sized chunks on double newlines, then each chunk is indexed by its lowercase tokens into an inverted index mapping words to filenames. At query time, each chunk is scored by counting how many query words appear in it. Chunks are sorted by score and only those meeting a minimum threshold of 2 matching words are returned, up to top-k results.

**What tradeoffs did you make?**  
For example: speed vs precision, simplicity vs accuracy.

Simplicity over accuracy. Keyword overlap is fast and easy to reason about, but it misses synonyms and context entirely. "auth token" and "access credential" would score zero against each other even if they mean the same thing. Chunking by paragraph is also a rough heuristic since some paragraphs are too short to be useful, others too long.

---

## 3. Use of the LLM (Gemini)

**When does DocuBot call the LLM and when does it not?**  
Briefly describe how each mode behaves.

- Naive LLM mode: Calls Gemini with just the question. The model answers on its own.
- Retrieval only mode: No LLM call. Returns the raw matched chunks directly to the user.
- RAG mode: Retrieves the top matching chunks first, then passes them to Gemini as the only context it's allowed to use.

**What instructions do you give the LLM to keep it grounded?**  
Summarize the rules from your prompt. For example: only use snippets, say "I do not know" when needed, cite files.

The prompt tells Gemini to answer using only the provided snippets, to say "I do not know based on the docs I have" if the snippets aren't enough, and to mention which files it relied on when it does answer.

---

## 4. Experiments and Comparisons

Run the **same set of queries** in all three modes. Fill in the table with short notes.

You can reuse or adapt the queries from `dataset.py`.

| Query | Naive LLM: helpful or harmful? | Retrieval only: helpful or harmful? | RAG: helpful or harmful? | Notes |
|------|---------------------------------|--------------------------------------|---------------------------|-------|
| Where is the auth token generated? | Harmful — answers confidently but from generic knowledge, not our codebase | Helpful — returns the right AUTH.md chunk | Helpful — clean answer citing AUTH.md | Naive LLM sounds plausible but is making things up |
| How do I connect to the database? | Harmful — gives generic DB advice unrelated to our setup | Helpful — surfaces the DATABASE.md chunk | Helpful — grounded answer using our actual config | RAG wins clearly here |
| Which endpoint lists all users? | Harmful — may hallucinate an endpoint name | Helpful — returns the correct API_REFERENCE.md chunk | Helpful — cites the correct route | |
| How does a client refresh an access token? | Harmful — generic OAuth explanation, not our actual route | Helpful — returns the refresh section from AUTH.md | Helpful — direct answer with file citation | |

**What patterns did you notice?**  

- When does naive LLM look impressive but untrustworthy? When the question is about something generic enough that the model has seen similar content in training — it sounds confident and fluent, but it's describing some other codebase, not yours.
- When is retrieval only clearly better? When you just need to know where something is documented. The raw snippet is honest about what the docs actually say.
- When is RAG clearly better than both? When you want a readable, synthesized answer that's still grounded in your actual docs. It combines the accuracy of retrieval with the readability of generation.

---

## 5. Failure Cases and Guardrails

**Describe at least two concrete failure cases you observed.**  
For each one, say:

- What was the question?  
- What did the system do?  
- What should have happened instead?

> **Failure case 1:** "Is there any mention of payment processing in these docs?" — The system returned unrelated chunks that happened to match one low-signal word. It should have said "I do not know" since the docs have nothing about payments.

> **Failure case 2:** "Which fields are stored in the users table?" — With a min_score of 2, this works, but with a lower threshold the system would return AUTH.md chunks that mention "users" without being about the database schema at all. It should only return DATABASE.md.

**When should DocuBot say "I do not know based on the docs I have"?**  
Give at least two specific situations.

> When no chunks score above the minimum threshold — meaning the docs don't contain the query terms at all. Also when the top-scoring chunks are only weakly related (e.g., one word overlap) and can't reasonably support an answer.

**What guardrails did you implement?**  
Examples: refusal rules, thresholds, limits on snippets, safe defaults.

> A `min_score=2` threshold in `retrieve()` — chunks must match at least 2 query words to be returned. If nothing passes the threshold, both answer modes fall back to "I do not know based on these docs." The RAG prompt also explicitly instructs Gemini to refuse rather than guess when snippets are insufficient.

---

## 6. Limitations and Future Improvements

**Current limitations**  
List at least three limitations of your DocuBot system.

1. Keyword matching has no understanding of meaning of the words in terms of synonyms, paraphrasing, and context are invisible to the scorer.
2. Paragraph chunking is potentially arbitrary. A paragraph might split a concept in half, or be too short to contain enough signal.
3. No memory or conversation history at the moment... but each query is treated completely independently!

**Future improvements**  
List two or three changes that would most improve reliability or usefulness.

1. Smarter chunking — split on section headers or fixed token counts instead of double newlines.
2. Add a confidence signal in the output so users know when an answer is well-supported vs. a weak match.

---

## 7. Responsible Use

**Where could this system cause real world harm if used carelessly?**  
Think about wrong answers, missing information, or over trusting the LLM.

Naive LLM mode can confidently describe APIs, credentials, or database schemas that don't exist in your project. If a developer trusts that output and ships code based on it, they're building with poor code. 

**What instructions would you give real developers who want to use DocuBot safely?**  
Write 2 to 4 short bullet points.

- Always verify answers against the actual source files before acting on them.
- Don't use naive LLM mode for project-specific questions
- Treat "I do not know" as a success, it means the system is being honest rather than guessing.

---
