# AI Tool Usage Declaration

Claude, ChatGPT, and Codex were used during development for architecture and design discussion, code generation, debugging assistance, documentation, and code review.

Their use was limited to the development workflow. The submitted ranking pipeline does not call Claude, ChatGPT, Codex, an LLM API, or any other generative model. It makes no network requests during ranking.

Candidate loading, integrity checks, recall, scoring, penalties, tie-breaking, reasoning, and CSV validation are deterministic and human-reviewed. Dense representations are precomputed offline and loaded as local artifacts. Submission reasoning is generated from fixed templates and observed candidate fields, not by generative AI.

AI assistance does not replace responsibility for the implementation or claims in this repository; the team remains responsible for reviewing, testing, and defending the final system.
