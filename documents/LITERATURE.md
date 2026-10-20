# Agentic search literature for the OMOP EHR agent

Reviewed 2026-10-12. These papers span open-domain search, retrieval-augmented generation (RAG), and structured EHR agents. Results on web or text benchmarks do not establish clinical accuracy on this project's OMOP data. “Agentic” here means the system can choose subsequent retrieval or tool actions based on evidence already seen, rather than selecting tables once and answering from a fixed prompt.

The trajectory is from prompted search loops toward trained search policies, retrieval critics, and jointly optimized retrievers. In EHR work, interactive querying and benchmarking are active, but recent benchmarks still report failures in code lookup, multi-table logic, and repeated-run reliability; this makes retrieval quality and verifiable evidence more immediate targets than fine-tuning a larger answer model alone.

## Starting map

- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG (2025; revised 2026)](https://arxiv.org/abs/2501.09136). A taxonomy of planning, routing, reflection, and multi-agent retrieval architectures. This is a survey, not a trained model; use it to compare design choices, not as performance evidence for a specific implementation.

## Retrieval and search methods

- [ReAct: Synergizing Reasoning and Acting in Language Models (2022/2023)](https://arxiv.org/abs/2210.03629). Interleaves reasoning steps and external actions so the next search can depend on the previous result. Primarily a prompting approach in its main demonstrations, though the paper also studies fine-tuning; an OMOP version could expose bounded, read-only search tools.

- [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT, 2023)](https://aclanthology.org/2023.acl-long.557/). Alternates reasoning and retrieval, letting each intermediate finding guide the next query. It reports gains without requiring additional model training; the analogous EHR pattern is to retrieve a visit, then retrieve measurements or medications linked to that visit.

- [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity (2024)](https://aclanthology.org/2024.naacl-long.389/). Trains a classifier to route questions among no retrieval, one retrieval, and iterative retrieval. For this project, a route for listings/counts could use deterministic database operations while relational or longitudinal questions use multiple retrieval steps.

- [Corrective Retrieval Augmented Generation (CRAG, 2024)](https://arxiv.org/abs/2401.15884). Uses a lightweight evaluator to judge retrieval quality and trigger corrective retrieval; it also filters retrieved material before generation. Its web-search fallback is not directly suitable for patient-specific claims, but a local correction step could detect missing tables, empty matches, or inadequate concept mappings and retry against the EHR.

- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (2024)](https://arxiv.org/abs/2310.11511). Fine-tunes a language model to emit reflection tokens for deciding when to retrieve and judging evidence and output. Relevant for deciding whether a claim is supported, but replicating its training is more involved than adding an external verifier around the current model.

- [RAG-Gym / Supervising the Search Process Produces Reliable and Generalizable Information-Seeking Agents (2025; revised 2026)](https://arxiv.org/abs/2502.13957). Studies supervision of intermediate search decisions rather than rewarding only final answers; its later version emphasizes query quality, reflection, and reusable search critics. An EHR analogue would score whether each tool call retrieved the right table, codes, patient, and time window before grading the final prose.

- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (2025)](https://arxiv.org/abs/2503.09516). Reinforcement-trains a model over trajectories with multiple search calls, using an outcome reward and masking retrieved tokens during optimization. The lesson for us is that tool-use behavior can be trained, but the reported gains are on general QA datasets rather than OMOP patient search.

- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments (2025)](https://arxiv.org/abs/2504.03160). Trains a multi-step research agent end-to-end with reinforcement learning against live web interactions, including navigation and cross-checking. Its open-web setting differs from a fixed EHR, but it argues for evaluating agents on real tool behavior instead of only prepackaged passages.

- [Agentic-R: Learning to Retrieve for Agentic Search (2026)](https://arxiv.org/abs/2601.11888). Trains a retriever with both local query–passage relevance and downstream answer correctness, iteratively improving the retriever as the agent's queries evolve. This suggests a later path for learning which patient rows or concept-linked records are actually useful, instead of taking the newest 100 rows.

## Structured EHR agents and evaluation

- [EHRAgent: Code Empowers Large Language Models for Few-shot Complex Tabular Reasoning on Electronic Health Records (2024)](https://aclanthology.org/2024.emnlp-main.1245/). Gives an LLM EHR metadata, examples, a code interface, execution feedback, and memory of successful cases so it can plan multi-table analyses. It uses few-shot prompting and iterative code repair rather than requiring a newly fine-tuned base model; for our system, constrained query tools would provide a safer first experiment than arbitrary generated code.

- [MedCP: An Agentic AI Framework Connecting Language Models to Electronic Health Records and a Biomedical Knowledge Graph for Real-World Evidence (2026)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1883853/full). Connects an OMOP EHR database and biomedical knowledge graph through audited, read-only tools coordinated by an LLM. The paper reports no model-weight fine-tuning; its relevance is schema-aware tool access and keeping patient-record evidence distinct from external biomedical knowledge.

- [From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents (EHR-ChatQA, 2025/2026)](https://arxiv.org/abs/2509.23415). Benchmarks agents that clarify ambiguous requests, resolve vocabulary mismatches, and execute EHR database queries. It is an evaluation environment rather than a newly trained model, and highlights that a model may succeed once while failing to answer the same question consistently across runs.

- [MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents (2025)](https://arxiv.org/abs/2501.14654). Provides physician-written tasks and an interactive FHIR EHR environment for testing tool-using agents. It does not propose fine-tuning; its task categories and execution-based grading are useful models for a patient-specific OMOP evaluation set.

- [EHR-Complex: Benchmarking Medical Agents for Complex Clinical Reasoning (2026)](https://arxiv.org/abs/2606.23301). Builds interactive MIMIC-IV tasks requiring SQL or Python across multiple tables and documents failures in query logic, code lookup, and clinical semantics. It is a benchmark, not a trained agent; the failure categories are directly useful for grading this project's retrieval traces and answers.

## Project-specific takeaways and possible research question

The current pipeline selects up to five tables once, reads up to 100 recent rows per table, then fits all evidence into a shared 5,000-token budget. A small read-only tool set—`find_concepts`, `count_rows`, `filter_rows`, `aggregate_measurements`, and `get_linked_visit_records`—would let an agent ask narrower follow-up questions. It should return row counts, truncation information, exact record IDs, and the selected filters with every result.

An especially useful research question is whether **budget-aware, schema-aware retrieval with explicit coverage and provenance** improves patient-level answer quality over a fixed one-shot table selection. Compare the current pipeline with: (1) deterministic query plans for simple requests, (2) a bounded multi-step tool agent, and (3) the same agent with a retrieval-quality check. Grade answer correctness, exact record citations, coverage, abstention, tool-call cost, and repeatability across patients and repeated runs. This is a hypothesis to test, not a claim of novelty; the EHR-agent and benchmark papers above already cover nearby territory.
