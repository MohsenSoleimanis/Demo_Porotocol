# Unstructured AI Pipeline — Cross-Vertical Engineering Reference (May 2026)

**Companion artifact to `unstructured_ai_landscape_2026.md`. This one is organized around the actual pipeline that turns messy text documents into structured records + KG, with parallel named examples across four verticals: clinical, legal, industrial/factory, financial.**

**Compiled from:** Phase 1-3 discovery + depth research (~500 primary-source searches, ~120 deep fetches) + fill-in pass on KG construction and chunk-level operations.

**Honest scope caveat:** "Unstructured AI" is not a fixed taxonomy. This artifact maps one usable slice — *the text-document pipeline that emits structured records into a schema-aware downstream system*. It deliberately excludes audio/video/code-base unstructured AI (those are real disciplines but not your factory). What it does cover: the stages you actually build, the cross-vertical patterns, where the field disagrees, what to use, what to skip.

---

## The pipeline (one diagram for the whole doc)

```
RAW INPUT (PDF, DOCX, HTML, scan, image-heavy, mixed)
   │
   ▼
[1] INPUT ADAPTER ──► detect format, route to parser, attach provenance
   │
   ▼
[2] PARSE ──► layout + text + tables + figures + reading order
   │          (Reducto, Hebbia, MinerU, Mistral OCR, SmolDocling, Granite-Docling,
   │           DeepSeek-OCR, Unstructured.io, LlamaParse, Tensorlake)
   │
   ▼
[3] SEGMENT/CHUNK ──► meaningful units (paragraph, section, table, list-item)
   │                  (section-tree, semantic boundaries, TopoChunker, RAPTOR, late chunking,
   │                   GraphRAG-as-chunker, propositions)
   │
   ▼
[4] ENRICH ──► per-chunk classification + metadata + quality + discourse role
   │           + language detection + PII detection
   │
   ▼
[5] ENCODE ──► embeddings (text, late-interaction, multimodal, domain-tuned)
   │
   ▼
[6] INDEX ──► vector + BM25 + KG (hybrid is the production default)
   │
   ▼
[7] RETRIEVE ──► hybrid retrieval + reranker + agentic retrieval
   │             (Hebbia ISD, GraphRAG, HippoRAG2, LightRAG, DRIFT)
   │
   ▼
[8] EXTRACT ──► entity / relation / structured-record extraction against a target schema
   │            (GLiNER, NuExtract, Instructor, BAML, Outlines, XGrammar, LangExtract)
   │            + anaphora / coref / negation / certainty / temporal linguistic ops
   │
   ▼
[9] RECONCILE ──► cross-document entity resolution + canonicalization
   │              + controlled-vocab linking (UMLS/SNOMED, LEI, ECLASS, CIK/CUSIP)
   │
   ▼
[10] VALIDATE ──► schema conformance + per-field confidence + provenance (bbox citations)
   │              + hallucination detection (HHEM, Patronus Lynx, SelfCheckGPT)
   │              + human-in-the-loop routing for below-threshold extractions
   │
   ▼
[11] MEMORY/STATE ──► persistent KG + temporal facts + cross-job experience
                       (Mem0, Letta, Zep/Graphiti, A-MEM, MIRIX)
   │
   ▼
DOWNSTREAM (CTMS/EDC/regulatory submission, contract repo, MES/ERP, financial systems)
```

**Cross-vertical schema targets (the *output* side that everything compiles toward):**

| Vertical | Schema target(s) | Conformance authority |
|---|---|---|
| Clinical | USDM v4 (CDISC), FHIR R5, CDISC ODM | CDISC Rules Engine |
| Legal | LegalRuleML, Akoma Ntoso, OASIS LegalDocML, FCL | LegalRuleML schema validators |
| Industrial / factory | ISA-95, OPC UA, AAS (Asset Administration Shell), STEP/ISO 10303, AutomationML | IDTA AAS validators, OPC Foundation companion specs |
| Financial | XBRL (US-GAAP/IFRS), FIBO, ISO 20022 | XBRL US validator, FIBO/SHACL validators |

---

# STAGE 1 — Input Adapters

**What it does:** detect format, route to format-specific parser, attach baseline provenance (filename, hash, source, ingest timestamp).

**Format taxonomy in practice:**
- **PDF** (60-80% of enterprise unstructured volume across all verticals) — native digital, scanned, hybrid, redacted, watermarked, multi-language
- **DOCX / RTF** (legal contracts, clinical protocols, industrial SOPs) — has explicit structure metadata you should never throw away
- **HTML / XML** — often scraped from regulator sites or vendor portals (FDA labels, SEC filings, EU directives)
- **Email + MIME / EML** (operational unstructured: orders, support, claims)
- **Spreadsheets** (CSV, XLSX) — tabular but messy; treat as semi-structured
- **JSON / XML structured** — USDM JSON, XBRL XML, ISO 20022 messages — *skip parsing; go straight to schema-driven mapping*
- **Images** (PNG, TIFF, JPG) — typically wrap scanned pages; route to OCR
- **Mixed bundles** — ZIP/EDI/multi-file submissions

**Production pattern: format-specific adapters, one IR.** Every adapter projects to the same `RawDocument` model with `content`, `format`, `pages`, `metadata`, `source`. Downstream stages are format-agnostic. This is the "typed semantic IR" pattern Bloomberg, Glean, Hebbia, Harvey all converge on internally.

**The under-talked decision: when input is JSON/XML against a known schema, skip Stage 1 entirely.** USDM JSON, XBRL XML, FHIR JSON, ISO 20022 messages — these are already structured. Validate against schema and proceed to KG mapping. The biggest mistake teams make: running their PDF parser over a structured input "for consistency."

**Failure modes:**
- Format misclassification (corrupted MIME headers, fake extensions)
- Mixed-language sniffing — Latin-script detection breaks on CJK/Arabic protocols
- Encoding gotchas (UTF-8 vs UTF-16-LE in DOCX XML, BOM markers in CSV)
- Watermarks / digital signatures interfering with text extraction
- Redacted/black-bar regions (clinical PHI redaction, legal carve-outs)

**Primary sources:** Unstructured.io ingest connectors (`unstructured.partition.auto`), Reducto ingest API, Tensorlake intake patterns, Loop's logistics ingest (CSV/JSON/PDF/PNG/EDI/API/email).

---

# STAGE 2 — Parse (OCR + layout)

This is the most-published stage with the richest engineering literature. Treat it as the substrate; if you get this wrong everything downstream is contaminated.

**Architectural paradigms (ranked by what production-leading teams actually ship):**

| Paradigm | Representative systems | When it wins |
|---|---|---|
| **Hybrid CV + VLM + agentic review** | **Reducto** (3-stage), Unstructured.io, LlamaParse Agentic Plus, Tensorlake, Extend AI | Production-leading default for regulated docs |
| **Doc-specific VLM, decoupled coarse-to-fine** | **MinerU 2.5** (1.2B NaViT+Qwen2-0.5B), SmolDocling (256M), Granite-Docling, dots.ocr (1.7B) | Best per-watt on text-heavy documents; layout on downsampled image, recognition on native res |
| **Unified end-to-end OCR-2.0 VLM** | GOT-OCR2.0, olmOCR/RolmOCR (Qwen2.5-VL FT), Nemotron-Parse 1.1 | When you need single-model deployment |
| **Vision-as-compression** | **DeepSeek-OCR** (DeepEncoder + 3B MoE A570M) | 97% at <10× compression; 60% at 20×; **200K pages/day on single A100**. The wildcard. |
| **Frontier general VLM** | Claude Opus + Citations, GPT-5, Gemini 2.5 Pro | Simple docs; expensive; hallucinate on tables/checkboxes |
| **OCR-free encoder-decoder** | Donut, Nougat | Lower compute; rigid output; fading |
| **Classical OCR + layout cascade** | AWS Textract, Azure DI, Google Document AI, ABBYY, Hyperscience | 60-80% on hard tables; brittle to long tail |

**Reducto's published architecture (the most-documented hybrid):**
1. Layout-first CV — segments tables/headers/figures/forms/text/images/graphs with spatial coords
2. VLM contextual analysis — establishes "relational hierarchy (which headers correspond to which table columns)"
3. **Agentic OCR** multi-pass correction — targets table misalignment, cross-column associations, field-label mismatches, orientation, mixed-language context loss
4. Block-level confidence + bounding-box citations
5. **RD-TableBench** (1000 hand-labeled tables, Needleman-Wunsch scoring): Reducto 90.2% vs Azure 82.7 / Textract 80.9 / Google DocAI 64.6

**Hebbia's "Goodbye RAG":** ISD architecture (Inference, Search, Decomposition). Embeddings fail on filter-style queries ("Pepsi revenue 2022" — 2022 generates false positives across companies/years; dense vector spaces conflate growth rates across products). Their answer: Full Attention pass over selected document pieces after multi-step generative decomposition; token-level log-likelihoods + character heuristics for hallucination mitigation.

**Databricks ai_parse_document (Nov 2025 GA):**
- **OfficeQA benchmark**: frontier agents score **<50%** on real enterprise docs without preprocessing
- ai_parse_document preprocessing delivers **+16% avg gain** across every agent framework
- 5-7× lower cost than VLM pipelines
- Concrete failure documented: agents hallucinating "$10,000 → $3,000" in insurance claims

**Vertical examples:**
- **Clinical**: Reducto + medical-doc post-processors; John Snow Labs Healthcare NLP; MinerU for trial protocols; SoA tables remain the hardest layer (PHUSE ML08 2025)
- **Legal**: Harvey's text-first/vision-second philosophy (vision activates only when needed for latency); LegalParse for case docs
- **Industrial / factory**: Loop's DUX 2.0 logistics-trained foundation model — **>99% touchless** on transportation docs across CSV/JSON/PDF/PNG/EDI/email; Tensorlake for MSDS/BoM/SOP/maintenance manuals
- **Financial**: Hebbia for SEC filings/contracts; edgartools + XBRL extraction; Mistral OCR for forms

**Failure modes (Reducto CEO + Databricks + Hebbia, all primary sources):**
- Silent column/row dropping in tables — output looks valid but is structurally lossy
- Checkbox interpretation flips ~50/50 — catastrophic in healthcare
- 1-2° skew dramatically degrades VLM extraction
- Watermarks corrupt text extraction
- Merged cells / multi-page tables / 2-D associations — main table failure cluster
- Reading order for multi-column / slide / non-Manhattan layouts
- Currency hallucination ("$10,000 → $3,000") — silent, business-critical, undetectable downstream
- Long-tail entropy: "Models today are incredible with reasoning on good data. What causes accuracy drift is the long tail of cases." (Adit, Reducto)

**Open debates:**
- VLM-only vs hybrid OCR+VLM — production-leading teams ship hybrid
- Specialized 1-3B doc-VLM vs frontier general VLM — MinerU2.5 1.2B and dots.ocr 1.7B beat Gemini 2.5 Pro on OmniDocBench
- Output format: DocTags (IBM) vs Markdown vs HTML — Adit recommends HTML for ≥3 merged cells, Markdown otherwise
- Build vs buy — Reducto/Unstructured argue "engineering drift" + perpetual maintenance + accuracy plateaus

**Primary sources:** reducto.ai/blog/document-parsing-unstructured-files · hebbia.com/blog/goodbye-rag-how-hebbia-solved-information-retrieval-for-llms · databricks.com/blog/why-frontier-agents-cant-read-documents-and-how-were-fixing-it · arxiv.org/abs/2509.22186 (MinerU 2.5) · arxiv.org/abs/2510.18234 (DeepSeek-OCR) · arxiv.org/abs/2503.11576 (SmolDocling) · jxnl.co/writing/2025/09/11/why-most-document-parsing-sucks-adit-reducto/

---

# STAGE 3 — Segment / Chunk

**The 2026 inflection: chunking is no longer "split by N tokens with M overlap."** It's "decide what the smallest queryable unit of meaning is, in this format, for this downstream task."

**Named techniques:**

- **Section-tree from headings** — what your existing FMLS chunker does. Cheap, deterministic, works for protocols with clean numbering. Failure mode: documents without explicit hierarchy.
- **TopoChunker** (arxiv 2603.18409, March 2026) — topology-aware agentic chunking. Treats chunking as a structural understanding problem, not segmentation. Agents analyze document topology (hierarchy, cross-references, logical boundaries) to decide chunk boundaries.
- **Late chunking** (Jina, 2024) — encode the *whole document* with a long-context encoder, then chunk the *embedding sequence* not the text. Each chunk's pooled vector has attended to the entire doc. Preserves cross-chunk semantics at retrieval.
- **RAPTOR** (Sarthi et al., ICLR 2024) — cluster leaf chunks by embedding, summarize each cluster with an LLM, recurse. You get a tree where leaves are facts and the root is the document. Queries hit the abstraction level that matches them.
- **GraphRAG as chunker** (Microsoft, June 2024) — extract entity graph from chunks, run Leiden community detection, generate community summaries. Chunks become graph nodes with community membership.
- **LightRAG** (HKUDS, EMNLP 2025) — dual-level: low-level keywords seed entity/relation lookup, high-level keywords seed cluster/theme lookup. <100 tokens / 1 API call per query vs ~610K tokens for GraphRAG global.
- **HippoRAG2** (arxiv 2502.14802, ICML 2025) — dual-node graph (passage + phrase), Personalized PageRank for retrieval, ~10³ tokens at query time vs 4×10⁴ for MS-GraphRAG global.
- **LazyGraphRAG** (Microsoft, Nov 2024) — defers all LLM work to query time. Indexing uses pure NLP noun-phrase extraction. **"Comparable quality to GraphRAG Global Search with 700× lower query cost."**
- **Propositionalization / Dense X Retrieval** (Chen et al., 2023) — decompose paragraphs into atomic propositions; index propositions for precision, keep paragraphs for context expansion. Up to +12 points on QA over paragraph chunks.
- **Contextual retrieval** (Anthropic, 2024) — LLM rewrites each chunk to prepend situating context. ~35% recall improvement.

**The five-pattern decision tree (when to use what):**

1. **Document has explicit structure (headings, numbered sections, DOCX styles, HTML DOM)?** → section-tree chunker, no LLM cost.
2. **Many short documents on the same topic (your 50k protocols)?** → cross-document topic-aligned chunking + section tree per doc.
3. **Long-form prose without structure (transcripts, free notes)?** → semantic chunking (BERT-based topic boundary, TextTiling), or LLM-as-chunker.
4. **High-precision QA needed on individual facts?** → propositionalization + parent-child indexing.
5. **Cost-bounded production at corpus scale (millions of docs)?** → LazyGraphRAG-style deferred work, or HippoRAG2 PPR.

**For the clinical/factory/legal/financial verticals:**

| Vertical | Chunking approach |
|---|---|
| Clinical | Section-tree by ICH M11 numbering (§1-§11) + SoA table as own chunk + criterion-level propositions in §5 |
| Legal | Section-tree by contract structure + clause-level propositions + cross-references as edges |
| Industrial | Section-tree by SOP step + spec-line propositions + IRDI references as edges |
| Financial | Section-tree by 10-K item structure + XBRL-tagged fact as chunk + table-as-chunk |

**Failure modes:**
- Section-tree fails when docs lack hierarchy (free-form, OCR-corrupt, multi-column)
- Semantic chunking fails on technical docs where topic boundaries don't align with information units
- LLM-as-chunker is expensive at corpus scale and non-deterministic across re-runs
- Late chunking requires the entire document fit in the encoder's context

**Open debates:**
- Static at-ingest vs dynamic at-query chunking (LazyGraphRAG / SmartChunk lean query-time)
- One chunker vs format-routed chunkers (production answer: format-routed)
- Cross-chunk references — encode as edges in KG (HippoRAG, GraphRAG) or as anchor metadata?

**Primary sources:** arxiv.org/abs/2603.18409 (TopoChunker) · jina.ai late chunking · arxiv.org/abs/2401.18059 (RAPTOR) · arxiv.org/html/2410.05779v1 (LightRAG) · arxiv.org/abs/2502.14802 (HippoRAG2) · microsoft.com/research/blog/lazygraphrag

---

# STAGE 4 — Enrich

**This is the stage most pipelines under-engineer.** Between chunking and extraction, every chunk needs typing, metadata, quality scoring, and discourse classification. Skip it and downstream extraction has no priors.

**Sub-operations:**

**4a. Chunk-level classification** — heading vs paragraph vs table vs criterion vs clause vs spec-line vs footnote vs list-item vs caption.

- **DocLayNet** (IBM, KDD 2022; v1.2 in docling-project) — 80,863 hand-annotated pages, 11 classes. The de-facto layout-classification dataset.
- **Tensorlake Page Classification** — define page types in natural language; engine routes pages to matching schema in one API call.
- **Azure AI Document Intelligence prebuilt-layout** + **Google Document AI layout parser** — paragraph/table/heading/footnote/caption tags out of the box.

**4b. Metadata patterns (now standard across vendors):**
- `section_path` / `full_path` ("Chapter 3 > Section 3.2")
- `page`, `bbox`, `parent_heading`, `chunk_type`
- `domain_section` — ICH M11 section for clinical, contract section type for legal, ISA-95 level for industrial, 10-K item for financial
- `MDKeyChunker` (arxiv 2603.23533) — single-call LLM enrichment producing title, summary, keywords, typed entities, hypothetical questions, semantic key per chunk

**4c. Quality scoring** — per-chunk on three axes: OCR conf × layout conf × extraction conf. Production thresholds: 0.8+ for general RAG, 0.9+ for clinical/legal/financial.

- **ChunkRAG** (arxiv 2410.19572) — LLM filters chunks for semantic relevance before generation
- **CONFLARE** (arxiv 2404.04287) — conformal prediction calibrates retrieval thresholds with stated coverage guarantee

**4d. Discourse role tagging** — claim vs evidence vs procedure-step vs exception vs definition. **eRST** (Enhanced RST, CL 2025) handles non-projective + concurrent discourse relations. **UniRST** (CODI 2025) — single parser across 18 treebanks / 11 languages.

**4e. Language detection / translation hooks** — script-aware OCR → language detection → language-specific chunker (CJK doesn't tokenize like Latin) → language-specific NER (mREBEL, GLiNER-X, ClinLinker for Spanish).

**4f. PII detection + redaction** — Microsoft Presidio, AWS Comprehend Medical, Tonic.ai. **GLiNER-PII variants** (knowledgator, NVIDIA, fastino) cover 42 PII types in 7 languages with F1 of 0.477 on the SPY benchmark.

**Vertical examples:**

| Vertical | Critical enrichment |
|---|---|
| Clinical | ICH M11 §-mapping per chunk, criterion-type (inclusion/exclusion), discourse role (claim/procedure/dose), PHI detection |
| Legal | Clause-type tagging (representations / warranties / indemnification / termination), defined-term flags, signature-block detection |
| Industrial | SOP step number, spec line type (requirement vs guideline), part-number presence, ECLASS/IRDI tagging |
| Financial | 10-K item mapping, XBRL concept presence flag, financial table classification (income / balance / cash flow / notes) |

**Failure modes:**
- Enrichment fields propagated as strings instead of typed values (your downstream consumers will hate you)
- No version of the enrichment ontology — when the taxonomy evolves, old chunks become stale
- LLM-based classification without confidence threshold — silent errors compound

**Primary sources:** docling-project/DocLayNet · tensorlake.ai/blog/announcing-page-classifications · arxiv.org/pdf/2603.23533 (MDKeyChunker) · arxiv.org/pdf/2410.19572 (ChunkRAG) · aclanthology.org/2025.codi-1.17/ (UniRST)

---

# STAGE 5 — Encode (Embed)

**The boring stage. Most teams over-think it.**

**Embedding model landscape (mid-2026):**

| Model | Best for | Note |
|---|---|---|
| **Voyage 3 / voyage-3-large** | General + domain (medical/legal/code variants) | Leads MTEB retrieval 2026; voyage-3-large beats text-embed-3-large by 9.74% |
| **OpenAI text-embedding-3-large** | General default | 3072-dim, Matryoshka-truncatable |
| **Cohere embed-v3 / embed-v4** | Multilingual | Strong on non-English |
| **Jina Embeddings v3** | Late chunking | Built for long-context |
| **BGE / GTE / mxbai** | OSS strong | Run on-prem |
| **ColPali / ColQwen** | Late-interaction over document images | Patch-level multi-vector; skip text extraction for retrieval |
| **MedCPT / BioLORD / ClinicalBERT** | Clinical | Domain-tuned |
| **LegalBERT, Legal-BERT-domain** | Legal | Domain-tuned |
| **FinBERT** | Financial | Domain-tuned |

**Production decisions that actually matter:**
1. **Matryoshka representation learning** lets you truncate dim at query time (e.g., 1024 for retrieval, 256 for fast routing). Voyage and OpenAI text-embed-3 support it natively.
2. **Late interaction (ColBERT/ColPali)** dominates when you want patch-level scoring instead of pooled-vector scoring. Critical for document-image retrieval.
3. **Domain fine-tuning** — only matters if your domain is sufficiently niche that off-the-shelf misses. Voyage's domain variants saturate most of the gain without custom training.
4. **Dimensionality**: 768 → 4096 is now the range (Qwen-3 embeddings hit 4096). Bigger isn't always better (Vicki Boykis "How big are our embeddings now and why").

**Vertical examples:**

| Vertical | Embedding choice |
|---|---|
| Clinical | MedCPT or BioLORD for clinical text; Voyage-3-large for general; multi-vector for SoA tables |
| Legal | LegalBERT for clause similarity; Voyage-3 for general retrieval |
| Industrial | Domain-tuned (rare); Voyage-3 + part-number alias dictionary as override |
| Financial | FinBERT for sentiment + risk language; Voyage-3 for filings; XBRL concept embeddings for taxonomy lookup |

**Failure modes:**
- Embedding model swap without re-indexing — silent retrieval quality collapse
- Mixed-language corpora with monolingual embeddings — 30%+ retrieval recall drop
- Tokenization mismatch between embedding model and chunker — boundary tokens get clipped

**Primary sources:** voyage.ai/blog · jina.ai/embeddings · vickiboykis.com/2025/09/01/how-big-are-our-embeddings-now-and-why/ · arxiv.org/abs/2407.01449 (ColPali)

---

# STAGE 6 — Index

**The big shift: vector DB as a category is dying; hybrid index won.**

**The 2026 storage landscape:**

| DB | Architecture | Best for | Drawback |
|---|---|---|---|
| **Neo4j** | On-disk native + Cypher `SEARCH` (vector + text in one engine, Cypher 25) | Production KG + RAG combined | AGPL/commercial license; mixed-workload latency lags Memgraph |
| **Kùzu** | Embedded, columnar | Was the embedded GraphRAG sweet spot — **but Apple archived it October 2025**. Community forks (LadybugDB) exist | No corporate backing |
| **Memgraph** | In-memory C++ | Sub-ms streaming analytics | RAM-bound; BSL license (not OSI) |
| **ArangoDB** | Multi-model (doc + graph) | Mixed doc/graph in one engine | OSS edition single-node only |
| **NebulaGraph** | Distributed native | Billion+ entity KGs | Operational complexity |
| **TigerGraph** | Distributed native | Deep multi-hop OLAP (100× Neo4j on BI workloads reported) | Proprietary |
| **ArcadeDB** | Multi-model on-disk | Apache 2.0 graph alternative | Smaller ecosystem |
| **Oxigraph, Apache Jena Fuseki** | RDF/SPARQL | When standard *mandates* SPARQL (FIBO, AAS-RDF, FHIR-RDF, LegalRuleML) | Smaller LLM-era tooling |
| **Pinecone, Weaviate, Qdrant, Milvus, Vespa** | Vector | Pure-vector RAG | Pinecone revenue halved 2025; category being absorbed |

**Production patterns that won:**

1. **Hybrid retrieval at storage layer.** Vector + BM25 + (optional) KG in one query engine. Neo4j 2025.10+ ships native `VECTOR` property with `SEARCH` clause. Elasticsearch / OpenSearch have hybrid built-in. pgvector + Postgres beats specialized vector DBs at small-to-medium scale.

2. **RDF for ontology-first verticals.** AAS-RDF (industrial), FIBO (financial), FHIR-RDF (clinical), LegalRuleML (legal) all benefit from SPARQL over a triple store. Oxigraph is the fastest embedded option.

3. **Embedded for prototyping, server for production.** SQLite + FAISS + DuckDB for experiments; Neo4j or ArangoDB for prod; Oxigraph when SPARQL is mandatory.

**Vertical examples:**

| Vertical | Index pattern |
|---|---|
| Clinical | Neo4j (USDM-shaped graph + vector + BM25 in one) OR Oxigraph (FHIR-RDF + SPARQL) |
| Legal | Neo4j or ArangoDB (contract clauses as nodes, parties as nodes, obligations as edges) |
| Industrial | Oxigraph (AAS-RDF) + companion Neo4j for analytical queries |
| Financial | Neo4j (FIBO-aligned KG) + XBRL document store + vector index over MD&A/narrative text |

**Open debates:**
- HNSW vs IVF vs DiskANN — HNSW wins on most retrieval benchmarks; DiskANN wins when memory-constrained
- Reranker after retrieval (Cohere Rerank, bge-reranker, Voyage Rerank) — almost always worth it; 18-42% precision lift
- Cross-encoder vs bi-encoder for reranking — cross-encoder wins quality, bi-encoder wins latency

**Primary sources:** neo4j.com/docs/neo4j-graphrag-python · oxigraph.org · spec.edmcouncil.org/fibo · industrialdigitaltwin.org (AAS) · cohere.com/blog/rerank-3

---

# STAGE 7 — Retrieve

**The big inflection: filter queries don't go through embeddings.**

**Hebbia's "Goodbye RAG" production lesson (worth re-stating):**
- *"What was Pepsi's revenue in 2022?"* produces false positives across many companies/years because dense embeddings **conflate filter intent with retrieval intent**
- Dense vector spaces cluster all "growth rate percentages" together regardless of which product they refer to
- Their answer: abandon embedding-based retrieval for filter-style queries; use LLM-as-retrieval-engine for those

**Production retrieval patterns by query type:**

| Query type | Best retrieval pattern |
|---|---|
| Semantic similarity ("find similar protocols") | Hybrid (vector + BM25) + reranker |
| Filter ("Phase 3 oncology Pfizer with creatinine inclusion") | KG-based filtering, not vector similarity |
| Multi-hop ("find related criteria across 50k trials") | GraphRAG / HippoRAG2 with PPR |
| Aggregation ("what's the average...") | Direct SQL/Cypher over structured fields, not RAG |
| Comparison ("compare endpoints across these 3 trials") | Multi-doc agentic retrieval (Hebbia Matrix pattern) |

**Named techniques:**

- **MS GraphRAG query modes**: Global (fan-out over community summaries), Local (entity-grounded neighborhood walk), DRIFT (Dynamic Reasoning and Inference with Flexible Traversal — global pass generates sub-questions, each runs local search). DRIFT is the production default by 2026.
- **HippoRAG2 PPR retrieval**: dual-node graph (passage + phrase), Personalized PageRank from query entities. ~10³ tokens at query time. Beats embedding-only by 7 F1 on multi-hop QA (MuSiQue, 2Wiki, HotpotQA, NarrativeQA).
- **LightRAG dual-level**: low-level keywords (entity/relation lookup) + high-level keywords (cluster/theme lookup). 1 API call per query.
- **ToG / ToG-2**: beam search on KG, LLM-as-agent picks next triple, training-free. SOTA on 6/9 KGQA benchmarks. ToG-2 alternates document-context + graph retrieval.
- **HybridRAG** (BlackRock + NVIDIA, ACM AI in Finance 2024): KG + vector hybrid; **wins all four metrics** (faithfulness, answer relevance, context precision, context recall) over either alone on Nifty 50 earnings calls.
- **Hebbia ISD** (Inference, Search, Decomposition): LLMs as retrieval engine + Full Attention pass over selected pieces. Most expensive but most accurate for complex queries.
- **Reranker stage**: Cohere Rerank 3, bge-reranker-v2-m3, Voyage Rerank 2. Almost always +18-42% precision lift over raw retrieval.

**Vertical examples:**

| Vertical | Retrieval recipe |
|---|---|
| Clinical | Vector + BM25 for similar-protocol search; GraphRAG over USDM-graph for cross-trial analysis; HippoRAG2 for criterion clustering |
| Legal | Hebbia-style multi-doc agentic for contract comparison; GraphRAG over party/obligation graph for due diligence |
| Industrial | SPARQL over AAS-RDF for "find all parts complying with ISO X"; hybrid for free-text technical query |
| Financial | HybridRAG over FIBO/XBRL + earnings transcripts (BlackRock-validated); cross-filing KG retrieval |

**Open debates:**
- Long context vs retrieval — Gemini 2M-token camp vs Chroma context-rot empirical camp. Anthropic firmly in the latter.
- Static retrieval vs agentic retrieval — agentic wins on complex queries, costs 5-15× more
- Pure vector vs hybrid — VentureBeat Q1 2026: hybrid retrieval intent **tripled** (10.3% → 33.3%) in one quarter

**Primary sources:** hebbia.com/blog/goodbye-rag-how-hebbia-solved-information-retrieval-for-llms · arxiv.org/abs/2502.14802 (HippoRAG2) · arxiv.org/abs/2404.16130 (MS GraphRAG) · microsoft.github.io/graphrag/query/drift_search/ · arxiv.org/abs/2408.04948 (HybridRAG BlackRock+NVIDIA) · trychroma.com/research/context-rot

---

# STAGE 8 — Extract

**The deliverable-producing stage. Where text becomes typed records against the target schema.**

**Three approaches, ordered by how production teams ship them:**

**8a. Zero-shot mention detection (cheap pre-filter):**
- **GLiNER lineage** (NAACL 2024, arxiv 2311.08526) — bidirectional encoder <500M params, CPU-runnable, zero-shot label-text matching
  - **GLiNER v2.5** community baseline
  - **GLiNER2** (Jul 2025) — 205M-param multi-task: NER + hierarchical structured extraction + classification through schema interface. CrossNER F1 0.590 vs GPT-4o's 0.599 at 2.62× speed, on CPU.
  - **GLiNER-BioMed** (arxiv 2504.00676) — distilled from OpenBioLLM-70B; +5.96 F1 over GLiNER-v2.5-large on clinical
  - **GLiNER bi-encoder / Million-Label NER** (arxiv 2602.18487) — 130× throughput at 1024 labels; enables Wikidata-scale entity linking
  - **GLiNER-Relex** (arxiv 2605.10108) — joint NER + RE in one forward pass
  - **GLiNER-PII** variants — 42 PII types, 7 languages

**Why GLiNER + LLM beats LLM-only:**
- GLiNER pre-filters: chunks with no candidate spans never hit the LLM (in 200-page protocol, only ~30% of chunks have actionable entities) — 3× LLM-call reduction
- LLM gets a primed prompt: instead of "extract anything", "fill the Intervention schema for each DRUG span and link DOSE spans to their parent drug"
- Provenance is free: GLiNER span offsets link back to chunk offsets link back to PDF page bboxes

**8b. Generative structured extraction (the meat):**
- **NuExtract 2.0 PRO** — beats GPT-4.1 by >9 F1 on text+image extraction at 10× lower cost; beats Claude 4 Opus by 5 points
- **NuExtract 3** — 4B VLM unified text+image+markdown extraction, vLLM serving
- **UniversalNER** (ICLR 2024) — targeted distillation, +30 F1 over Alpaca/Vicuna, +7-9 F1 over ChatGPT on average
- **InstructUIE** — Flan-T5 + 32-dataset IE INSTRUCTIONS benchmark

**8c. Schema-guided extraction libraries (production patterns):**

| Library | Strength | Tradeoff |
|---|---|---|
| **Instructor** (Jason Liu) | ~3M monthly downloads; Pydantic-native, retry-on-validation, 15+ providers; LSEG runs in prod | Breaks on markdown-wrapped JSON, CoT preamble, trailing commas |
| **BAML** (Boundary ML) | DSL → 7-language client code; **60% fewer tokens than JSON Schema**; **Schema-Aligned Parsing** beats function-calling and AST parsing across GPT-4o (93%), Claude 3.5 Sonnet (94.4%), GPT-4o-mini (92.4%) | New DSL to learn |
| **Outlines** | JSON-schema → FSM, O(1) token lookup | Can't handle deeply recursive schemas |
| **XGrammar** | PDA, CFG-expressive, FSM-fast. Default backend for vLLM/SGLang/TensorRT-LLM as of March 2026. <40µs/token | Per-request unique schemas defeat caching |
| **llguidance** (guidance-ai) | Computes masks on-the-fly, ~0 startup cost. **0.12% invalid-JSON rate vs XGrammar 2.21%** in serving frameworks | Younger ecosystem |
| **LangExtract** (Google, Aug 2025) | Gemini-powered; every extracted span has source offset for grounding; HTML visualization | Gemini-only |
| **Marvin** (Prefect) | `@ai_model` decorators on Pydantic | |
| **Mirascope** | Unified interface across all major providers, vision-mode | |
| **TypeChat** (Microsoft) | TypeScript-first | |

**JSONSchemaBench** (Geng et al., 2025, arxiv 2501.10868) — 10K real-world schemas across Guidance, Outlines, llamacpp, XGrammar, OpenAI, Gemini. Constrained decoding speeds generation up to 50% with notable variance, +4% downstream accuracy. Many backends still fail on GitHub Hard and JSON Schema Store edge cases.

**Cross-schema stress tests (the real test):**
- **USDM v4 / ICH M11** (clinical) — deeply nested, references across study arms; SoA tables remain hardest layer (PHUSE ML08 2025)
- **LegalRuleML** (legal) — deontic operators (MUST/SHALL/MAY), defeasibility, cross-references between obligations
- **ISA-95 / IRDI-typed AAS** (industrial) — hierarchical, bound to physical part numbers via IRDI
- **XBRL / US-GAAP** (financial) — **FinTagging benchmark**: DeepSeek-V3 hits 72% F1 on extraction, but even best models reach only **17%** on linking to the >10,000-concept US-GAAP taxonomy

**8d. Relation extraction:**
- **REBEL** (Babelscape, EMNLP 2021) + **mREBEL/REDFM** (ACL 2023, multilingual)
- **GenIE** (NAACL 2022) — first end-to-end autoregressive closed-IE
- **DocContextRE** (2025) — 88.9% F1 on DocRED, 70.4% on Re-DocRED, 93.6% on REBEL
- Inter-sentence F1 is still 10-15 points below intra-sentence — long-distance reasoning is the open frontier
- **Domain RE in production:**
  - Clinical: drug-drug interaction (BioMCL-DDI, CNN-DDI), drug-disease, ADE→drug, BioDEX
  - Legal: party→obligation, party→clause, defined-term→definition, parent→subsidiary
  - Industrial: part-of, requires, complies-with, is-spec-of, is-replacement-of (IRDI-typed)
  - Financial: parent→subsidiary, lender→borrower, issuer→instrument, beneficial-owner→entity

**8e. Linguistic operations critical for production extraction:**

**Anaphora / coreference resolution:**
- **fastcoref / F-coref** — Bar Ilan, online 4096-token Longformer chunks for long docs
- **LQCA** — Long Question Coreference Adaptation: sub-document resolution + representative-mention selection; +3.18% across metrics on o1-mini
- **LegalCore** (Feb 2025, arxiv 2502.12509) — event coreference for legal docs ~25K tokens. The clinical/legal equivalent of OntoNotes for long contexts.
- **CRAC 2026 winner** (arxiv 2605.16984) — Gemma-3-27b two-stage adapter, **CoNLL F1 74.32**
- Cross-domain mention patterns: "the patient" / "this drug" (clinical) | "the Lessee" / "this Agreement" (legal) | "this part" / "the sensor" (industrial) | "the Issuer" / "the Notes" (financial)

**Cross-reference resolution:** "see Section 3.2" / "as defined in Annex B" / "per ISO 9001 §7" / "as described in 21 CFR 312.62". Production pattern: regex-extract references → resolve against `section_path` index built at chunking → re-attach target chunk text/id to source chunk metadata.

**Negation detection (universal but vertical-specific tooling):**

| Vertical | Tools | Cues |
|---|---|---|
| Clinical | NegEx (Chapman 2001), ConText, MedSpaCy ConText, DEEPEN, NegEx+CNN | "rule out X", "no evidence of X", "denies X", "history of X" |
| Legal | Modal parsing (SHALL/SHALL NOT/MAY/MUST NOT); carve-outs ("except where") | "shall not", "is exempt from", "notwithstanding", "subject to" |
| Industrial | Spec negation cues, typically rule-based | "shall comply with", "is exempt from", "except where" |
| Financial | MD&A and risk-factor language | "does not include", "excluding", "other than" |

**Known LLM negation failure rates:**
- Qwen2-VL-72B drops from 92.2% accuracy to 72.7% on NegVQA when negation is introduced (arxiv 2505.22946)
- Thunder-NUBench and CondaQA show models react to surface cues rather than inverting truth conditions
- MedSpaCy ConText: 0.795 recall / 0.356 precision / F1 0.492 on radiology negation
- **Don't trust an LLM to invert truth conditions without specific training** — explicit ConText/NegEx layers are still essential in clinical/legal pipelines

**Certainty / hedging:** BioScope corpus, CoNLL 2010 hedge-cue task. Clinical assertion status: present / absent / hypothetical / possible / conditional / family-history. JSL ships 10+ assertion-status models. Legal hedging maps to modal-strength gradients (must > shall > should > may > can).

**Temporal annotation:** HeidelTime (rule-based TIMEX, precision 0.828 / recall 0.822), SUTime, Clinical TempEval (THYME corpus), SCATE. Production tags: `baseline`, `on-study`, `effective-from`, `amendment-v3`, `as-of-date`, `period-covered`.

**Span-based vs generative tradeoffs:**
- Span-based encoders (DeBERTa, BERT variants) **still 5-15 F1 ahead** on i2b2 2010 / n2c2 medications/ADE under strict matching
- Generative wins for nested / discontinuous / low-resource / heterogeneous schemas
- 2025 consensus: hybrid — span-based mention detection + generative attribute/relation extraction

**Failure modes:**
- Schema-validation retry loops without max-retry — runaway cost
- Constrained decoding cost when schema changes per request — kills XGrammar caching
- LLM hallucinated entities matching the schema but absent in source — needs verifier stage
- Coref accuracy ceiling — long-distance event coref in legal still unsolved
- Negation flipped 15-30% of the time without explicit ConText layer

**Primary sources:** python.useinstructor.com · docs.boundaryml.com/home · boundaryml.com/blog/schema-aligned-parsing · arxiv.org/pdf/2411.15100 (XGrammar) · arxiv.org/abs/2501.10868 (JSONSchemaBench) · arxiv.org/abs/2311.08526 (GLiNER) · arxiv.org/abs/2504.00676 (GLiNER-BioMed) · arxiv.org/abs/2507.18546 (GLiNER2) · numind.ai/blog/outclassing-frontier-llms · arxiv.org/abs/2505.20650 (FinTagging)

---

# STAGE 9 — Reconcile (Cross-document entity resolution + controlled-vocab linking)

**The stage most teams don't realize they need until they're a year in.**

When "Pfizer Inc." appears in 5,000 documents as "Pfizer", "PFE", "Pfizer Inc.", "Pfizer, Inc.", "Pfizer Pharmaceuticals", "Pfizer Pharmaceuticals Inc." — you need one canonical node. Same for drug names, clinical sites, contract parties, equipment IDs, financial instruments.

**Two operations:**

**9a. Cross-document entity resolution (record linkage):**
- **Blocking** on entity type + fuzzy name match
- **Embedding similarity** over node names with a merge threshold
- **LLM adjudication** for ambiguous pairs
- **Maintain multi-valued facts** with confidence + source attribution rather than destructive merges

The empirically reliable stack (KGGen, EDC, LKD-KGC, LLM-Align, EntGPT, 2024-2025).

**Production approaches:**
- **Match-LLM, ZeroER, DITTO** — LLM-based ER families
- **Embedding-based clustering** for dedup
- **Wikidata-grounded linking** when entities exist in public KBs
- **Zep / Graphiti pattern**: cosine + full-text retrieval against existing graph nodes, LLM adjudication for ambiguous pairs

**The Microsoft GraphRAG default does blocking only;** this is the **#1 cause of "near-duplicate node explosion"** reported in production deployments.

**9b. Controlled-vocabulary linking (canonicalization to public IDs):**

| Vertical | Vocabularies | Tools |
|---|---|---|
| **Clinical** | UMLS Metathesaurus (~3M concepts), SNOMED CT (~360K), RxNorm (~120K), LOINC (~95K), MeSH, ICD-10-CM, HPO, MONDO | **MedCAT v2** (CogStack, UK SNOMED + UMLS 2024AA + MIMIC-IV), **scispaCy `UmlsEntityLinker`**, MetaMap, BioMistral fine-tunes, OntoGPT/Gilda |
| **Legal** | LEI/vLEI (GLEIF), OpenCorporates (230M+ legal entities, 145 jurisdictions), ECLI (case law), Akoma Ntoso URIs | OpenCorporates bulk + LEI mapping; LexNLP citation extractor |
| **Industrial** | ECLASS IRDIs (`0173-1#02-BAD856#005`), GS1 GTIN, ISO 11179, IEC CDD, AAS submodel IDs, OPC UA NodeId | AAS RDF serialization + SPARQL; py-aas-rdf |
| **Financial** | CIK (SEC), CUSIP (S&P), ISIN, FIGI (OpenFIGI/Bloomberg), LEI (GLEIF), NAICS/SIC | EODHD ID Mapping API; CIK↔CUSIP via Python helpers; FIBO IRIs |

**Domain-specific linker references:**
- **MedCAT v2** (April 2025) — UK SNOMED + UMLS 2024AA + MIMIC-IV; ships UK NHS clinical models
- **scispaCy** — 4 NER models + UMLS linker; supports UMLS/MeSH/RxNorm/GO/HPO
- **John Snow Labs Healthcare NLP** — 600+ pretrained pipelines, 100+ NER models, 60+ entity resolvers across ICD-10/CPT-4/UMLS; **96% F1 on PHI de-id** (Text2Story 2025); beats AWS Comprehend Medical by 18%, Azure Health by 12%
- **SNOMED-CT Entity Linking Challenge** (DrivenData, 2024) — canonical benchmark; winners mixed dictionary + encoder (PubMedBERT) + decoder (BioMistral)
- **MedPath** (arxiv 2511.10887) — augments linked entities with hierarchical paths across 62 biomedical vocabularies; materially improves multi-hop reasoning
- **OpenCorporates legal-entity KG** (blog post Oct 2025) — 230M+ legal entities reconciled with Open Ownership + GLEIF; canonical 2025 legal-entity-resolution graph
- **ECLASS IRDI spec** (ISO/IEC 11179-6, ISO 29002, ISO/IEC 6523) — IRDI format `0173-1#02-BAD856#005`; ICD prefix `0173` identifies ECLASS as publisher

**Domain-general linker techniques:**
- **BLINK** (Wu et al., EMNLP 2020) — bi-encoder retrieval over 5.9M Wikipedia candidates in 2ms + cross-encoder rerank; +6 F1 zero-shot
- **GENRE / mGENRE** (De Cao et al., ICLR 2021) — autoregressive entity name generation with constrained beam search over prefix trie
- **ReFinED** (Ayoola et al., NAACL Industry 2022) — mention detection + fine-grained typing + disambiguation in single forward pass; 60× faster than predecessors; scales to 15M Wikidata entities
- **LELA** (2025) — fine-tuning-free LLM entity linking with zero-shot domain adaptation
- **GLiNKER** — GLiNER bi-encoder + Wikidata; sub-second linking at millions of labels

**Cross-document coreference at corpus scale:**
- **Maverick pipeline** (2024) — reference for efficient CDCR
- **xCoRe** (EMNLP 2025) — within-context mention extraction + clustering → cross-context cluster merging
- **CRAC 2025 / CODI-CRAC** — multilingual LLM-driven CDCR shared task

**Linking decision rule:**
- **Deterministic candidate gen** (string match, fuzzy) + **embedding ranking** + **LLM disambiguation only on top-k** is the production winner across all four verticals
- *"Deterministic scoring is preferable for entity resolution — it's faster, fully explainable, and cannot hallucinate."* (community wisdom, multiple sources)

**Failure modes:**
- Destructive merges erase provenance — once two records merge, you can't recover which document said what
- Threshold tuning per-domain — universal threshold across verticals fails
- Vocabulary version skew — UMLS updates yearly; entities can shift CUIs
- Hallucinated linking (LLM picks a plausible-but-wrong UMLS CUI) — needs verifier

**Open debates:**
- Open vs closed-world ER — when do you accept "no canonical match" vs force a link?
- Hierarchical canonicalization (MedPath-style) vs flat
- LLM-based linking vs deterministic — production wisdom favors hybrid with LLM only at the top

**Primary sources:** github.com/CogStack/MedCAT · github.com/allenai/scispacy · drivendata.co/blog/snomed-ct-entity-linking-challenge-winners · github.com/facebookresearch/BLINK · github.com/amazon-science/ReFinED · eclass.eu/support/technical-specification/structure-and-elements/irdi · gleif.org/en/about-lei · spec.edmcouncil.org/fibo

---

# STAGE 10 — Validate (Confidence + Provenance + Schema Conformance)

**The regulatory gate. Without this, your pipeline output is unusable for actual sponsor/customer/regulator consumption.**

**Three layers:**

**10a. Per-field bounding-box citations + per-stage confidence propagation:**

Every extracted field carries:
```
{
  value: ...,
  page: 47,
  bbox: [x1, y1, x2, y2],
  chunk_id: "ch_42",
  source_text_span: [start, end],
  extractor_model: "claude-sonnet-4-5",
  extractor_version: "...",
  prompt_hash: "sha256:...",
  schema_version: "USDM-v4.0",
  confidence: 0.87,
  parser_confidence: 0.94,
  layout_confidence: 0.92,
  link_confidence: 0.81,
  timestamp: "2026-05-22T15:30:00Z",
  reviewer: null | "reviewer@org.com"
}
```

**Production references:**
- **Reducto** — block-level confidence + bbox citations; RD-TableBench 90.2%; multi-pass agentic review
- **Tensorlake** — `provide_citations: True` gives per-field bbox + page; reports 91.7% on enterprise docs; processes 100K+ docs/day per customer
- **Anthropic Citations API** (June 2025) — sentence-level grounded references; **0% hallucinated citations** in 115-query eval; users don't pay output tokens for quoted text
- **Extend** — per-field bbox citations productized for KYC and underwriting
- **CiteLLM** — verification layer with snippets, confidence, page numbers, bbox

**10b. Hallucination detection (independent claim verification):**

| Tool | What it does |
|---|---|
| **HHEM** (Vectara) | Few-shot LLM-as-judge calibrated to human labels; 2025 benchmark expanded to 32K tokens across law/medicine/finance/tech. Reasoning models score worse: Grok-4-fast-reasoning hits 20.2% hallucination |
| **Patronus Lynx 8B / 70B** | Open hallucination detector for RAG: doc+Q+A → faithfulness. Lynx-70B beats GPT-4o by 8.3% on PubMedQA. NVIDIA NeMo Guardrails integration |
| **SelfCheckGPT** | Reference-less; samples N answers, checks consistency |
| **FActScore** | Decomposes generation into atomic facts, verifies each against retrieved evidence |
| **RAGAS faithfulness** | Claims-supported / total-claims; mature OSS eval metric |
| **HaluBench** | Real-world hallucination benchmark from Patronus |

**Stanford 2025 legal-AI study finding:** even RAG-grounded legal tools hallucinate **17-34%** of queries; invalid-citation rate up 80.9% over 2020-2024. **Citations alone are necessary but not sufficient** — independent claim verification is now standard.

**10c. Schema conformance + auto-retry:**

- **Pydantic validation** at every stage boundary; retries on validation error with the error message fed back into prompt (Instructor pattern)
- **BAML Schema-Aligned Parsing** — accepts dirty output and runs schema-aware error correction at parse time
- **JSON Schema validation** (XGrammar, llguidance) — enforced at decoding time

**Per-vertical conformance authorities:**

| Vertical | Conformance authority | Tool |
|---|---|---|
| Clinical | CDISC Rules Engine | github.com/cdisc-org/cdisc-rules-engine |
| Legal | LegalRuleML schema validators | OASIS-published |
| Industrial | IDTA AAS validators, OPC Foundation companion specs | py-aas-validator |
| Financial | XBRL US validator, FIBO/SHACL validators | xbrl.us validator |

**10d. Confidence-based human-in-the-loop routing:**

Below-threshold extractions route to a reviewer queue. This is the *real* product, not the model. The discipline:
- Per-field confidence threshold tunable per use case (clinical eligibility = 0.95+; industrial spec = 0.85+; financial number = 0.99+)
- Reviewer interface shows the bbox citation on the source PDF + the extracted field
- Corrections feed back as training signal (active learning loop)

**The actual production metric:** *reviewer hours saved per protocol* (or per contract, per spec, per filing). Not F1. Reducto, Hyperscience, Klarity all measure this internally.

**Lineage / provenance engineering for regulatory audit:**
- BCBS 239, GDPR, CCPA, SOX, EU AI Act, FDA 21 CFR Part 11 all require full lineage
- Atlan and OpenCorporates blog posts in 2025 codify this as audit-ready lineage
- Production teams ship lineage from byte to KG node

**Open debates:**
- Confidence calibration — most vendors ship decorative confidence scores; only some calibrate (post-hoc isotonic regression or Platt scaling)
- Synthetic confidence (from LLM logprobs) vs calibrated (from held-out validation)
- HITL routing thresholds per-field vs per-document

**Primary sources:** anthropic.com/news/introducing-citations-api · tensorlake.ai/blog/announcing-citations · docs.reducto.ai/v/legacy/extraction/citations · github.com/vectara/hallucination-leaderboard · patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model · arxiv.org/abs/2305.14251 (FActScore) · cdisc.org/standards/data-exchange/usdm

---

# STAGE 11 — Memory / State

**For the factory, this is where cross-document knowledge accumulates.** When you process 50k protocols / 500k contracts / 5M parts, you need persistent memory across jobs.

**Two layers:**

**11a. Domain KG as persistent memory** (the dominant pattern for unstructured-AI factories):
- Every extracted entity becomes a node in the domain KG (USDM Study, LegalRuleML Obligation, AAS Asset, FIBO Instrument)
- Every relation becomes an edge
- Every document is provenance metadata on the nodes/edges it produced
- Queries hit the KG, not the raw documents (until expansion is needed)

This is the HippoRAG / GraphRAG vision applied to the production factory: the KG IS your memory.

**11b. Temporal / bi-temporal facts** (when knowledge changes over time):

**Zep / Graphiti** (Rasmussen et al., arxiv 2501.13956, Jan 2025) — production reference for bi-temporal agent memory:
- Four timestamps per edge:
  - `t'_created`, `t'_expired` on the **transaction timeline T'** (when ingested / invalidated)
  - `t_valid`, `t_invalid` on the **event timeline T** (when the fact was actually true)
- Three subgraph layers: **Episodic** (raw), **Semantic** (extracted entities/edges), **Community** (clusters with summaries)
- Edge invalidation = LLM compares new facts to semantically related existing edges, sets `t_invalid` on contradicted edges
- Entity resolution = cosine + full-text + LLM adjudication
- DMR benchmark: **94.8% (Zep) vs 93.4% (MemGPT)**; LongMemEval: **+18.5% accuracy and 90% latency reduction**; P95 300ms

This is why your clinical extraction project will need temporal: protocols have amendments (Amendment v3 supersedes Amendment v2); contracts have amendments; specs have revisions; financial statements are point-in-time.

**Other memory systems:**
- **Mem0** (ECAI 2025) — 81.95% on LoCoMo at ~1,294 tokens/query (5% of full context); 91% p95-latency reduction
- **Letta / MemGPT** — OS-style memory hierarchy (main context + recall + archival); sleep-time compute consolidates between turns
- **A-MEM** (NeurIPS 2025) — Zettelkasten-inspired memory evolution
- **MIRIX** — six memory types (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault)

**Benchmarks (now standard):**
- **LoCoMo** — 1,540 questions, single-hop / multi-hop / temporal / open-domain
- **LongMemEval** (ICLR 2025) — 500 questions, 5 abilities including temporal reasoning
- **BEAM** (ICLR 2026) — 100 conversations, 2,000 questions, scales to **10M tokens**

**Vertical examples:**

| Vertical | Memory pattern |
|---|---|
| Clinical | USDM KG as memory; amendments as temporal events on Study/StudyVersion edges |
| Legal | Contract KG; amendments + assignments + waivers as temporal events |
| Industrial | AAS as living digital twin; part-spec changes as version events; maintenance history as episodic memory |
| Financial | FIBO KG + XBRL fact store; filings as periodic snapshots; corporate-action edges (M&A, restructure) as events |

**Failure modes:**
- Memory staleness in high-relevance entries (Mem0 2026 names this as open problem)
- Cross-session identity break across anonymous sessions
- ~25% performance loss at 10M tokens (BEAM)
- Memory poisoning via indirect prompt injection persisting across sessions (MemoryGraft, Dec 2025)
- Treating change as replacement instead of evolution (overwrite vs versioning)

**Open debates:**
- Graph vs vector vs filesystem for memory (three camps with public benchmarks against each other)
- Sleep-phase consolidation needed? (Letta/Anthropic Auto-dream yes; ACE/Memori no)
- In-weights (fine-tuning on accumulated memory) vs in-context (retrieval at query time)

**Primary sources:** arxiv.org/abs/2501.13956 (Zep/Graphiti) · arxiv.org/abs/2504.19413 (Mem0) · arxiv.org/abs/2603.19935 (Memori) · letta.com/blog/sleep-time-compute · mem0.ai/blog/state-of-ai-agent-memory-2026

---

# CROSS-CUTTING PATTERNS (recur across 5+ stages)

These are what production-leading teams across all four verticals have converged on, regardless of domain.

**1. Hybrid wins everywhere.**
- Retrieval: vector + BM25 + KG + reranker (VentureBeat Q1 2026: hybrid intent tripled)
- Extraction: GLiNER candidate generation + LLM structured-fill
- Storage: graph + vector + text in one engine (Neo4j Cypher `SEARCH`)
- Linking: deterministic candidate + embedding ranking + LLM disambiguation only at top-k

**2. Schema-constrained backbone + open long-tail + post-hoc canonicalization.**
The empirically reliable pattern: define the high-value schema (USDM, LegalRuleML, AAS, FIBO) for the relations the application *needs*; keep open extraction in a "long-tail" namespace; post-hoc canonicalize. Pure open extraction creates duplicate-node explosions within weeks. Pure schema-constrained extraction misses long-tail relations.

**3. Provenance per record is non-negotiable.**
Every extracted field traces to PDF byte range, model version, prompt hash, validator, timestamp, reviewer signoff. Regulators require it. Add it day one or rewrite the pipeline later. This is what makes the difference between a usable production system and a hallucination-prone toy.

**4. Confidence propagation across stages.**
Confidence is not a final scalar — it's a *vector* that travels with the data. OCR conf × layout conf × extraction conf × link conf. Routing decisions (auto-accept vs reviewer queue) happen *per-field*, not per-document.

**5. The agent legibility shift.**
Code, docs, schemas, tools authored for the model first. AGENTS.md, BAML DSLs, Symphony's `core_beliefs.md` are all artifacts engineered for agent consumption. Applies to factory engineering: write extraction prompts assuming the model is the primary reader.

**6. Per-vertical eval discipline.**
- Clinical: i2b2/n2c2, BioASQ, CDR, GDA, BC5CDR, NCBI Disease, BioRED, CHIA, PubMedQA
- Legal: CUAD (510 contracts, 13K labels, 41 clause types), ContractNLI, ContractEval (Aug 2025), LegalCore
- Industrial: RD-TableBench, ParseBench, ISA-95 KG benchmarks (early); SDS/MSDS no public benchmark yet
- Financial: FinTagging, HIFI-KPI, EdgarTools, FinBERT-MRC datasets

**7. Filesystem all you need (for *agent* state, not for *corpus* retrieval).**
Letta's contrarian thesis: `grep + read_file + write_file` beats specialized graph stores on LoCoMo (74.0% vs Mem0 graph 68.5%). Why? Tools the model has seen in pretraining outperform specialized stores. **This applies to agent working memory, NOT to corpus-scale retrieval over 50k documents.** Don't confuse the two.

**8. RAG isn't dead. RAG-as-a-standalone-category dissolved into hybrid.**
- AI Engineer 2025 "killed the RAG track"
- RAGFlow's 2025 review titled "From RAG to Context"
- *But:* VentureBeat Q1 2026 enterprise hybrid retrieval intent tripled
- For corpus-scale extraction at 50k+ documents, you need retrieval. Filesystem grep doesn't scale.

**9. Negation / coref / cross-reference are still rule-augmented.**
LLMs flip truth conditions wrong 15-30% on NegVQA-style evals. Production pipelines stack a rule-based ConText / NegEx / modal-parser layer on top of LLM extraction. Especially in clinical and legal.

**10. Reviewer throughput is the real economic gate.**
MIT NANDA: 95% pilots fail; bought tools succeed 67% vs internal builds ~22%. The product metric isn't model F1 — it's reviewer-hours-saved-per-document. The pipeline must produce per-field confidence + bounding-box provenance to enable HITL routing, or the economics don't work.

---

# WHAT TO ACTUALLY BUILD (the factory blueprint)

If you're implementing a domain-agnostic unstructured-AI factory that ships:

**Layer 1: Input gateway**
- Format-routed adapters → typed `RawDocument`
- Format-specific parsers: PDF (Reducto-style hybrid), DOCX (read styles directly), HTML (DOM walker), XML/JSON (skip-to-mapping when schema known)
- Versioned ingest provenance (filename, hash, source, timestamp, customer/tenant)

**Layer 2: Substrate processing**
- Parse → typed blocks with bbox, conf, reading order
- Chunk → section-tree (cheap default) + format-specific overrides + GraphRAG-as-chunker for cross-document tasks
- Enrich → DocLayNet-style chunk classification + section_path metadata + domain-section mapping + PII detection + discourse role + language

**Layer 3: Knowledge representation**
- Embed → Voyage-3-large default + domain-tuned for specialized verticals (MedCPT/LegalBERT/FinBERT)
- Index → Neo4j (graph + vector + text in one engine) OR Oxigraph (when SPARQL-mandated for FIBO/AAS/FHIR-RDF)
- Hybrid retrieval at the query layer

**Layer 4: Extraction**
- GLiNER / GLiNER2 / GLiNER-BioMed → cheap mention detection (CPU)
- BAML or Instructor → schema-guided structured extraction with target schema (USDM / LegalRuleML / ISA-95-AAS / XBRL-FIBO)
- Domain-specific extraction prompts conditioned on enriched chunk metadata
- Linguistic ops: ConText/NegEx for negation, fastcoref for anaphora, regex+section-path for cross-reference

**Layer 5: Reconciliation**
- Cross-doc ER: blocking → embedding similarity → LLM adjudication at top-k
- Controlled-vocab linking: domain-specific (MedCAT clinical, OpenCorporates legal, ECLASS industrial, OpenFIGI financial)
- Maintain multi-valued facts with provenance; never destructive merge

**Layer 6: Validation**
- Schema validation against target (CDISC Rules Engine, LegalRuleML validators, AAS validators, XBRL US validator)
- Confidence propagation (OCR × layout × extract × link × judge)
- Hallucination detection (HHEM / Patronus Lynx for high-risk fields)
- HITL routing for below-threshold extractions
- Per-field bounding-box citations attached to every record

**Layer 7: Memory + downstream**
- Persistent domain KG (USDM-shaped / LegalRuleML / AAS / FIBO)
- Temporal facts via Zep/Graphiti bi-temporal model (handles amendments, revisions, restatements)
- Conformance export to downstream systems (CTMS/EDC, contract repo, MES/ERP, financial systems)

**The customer-configurable surface:**
- Target schema (per-tenant)
- Confidence thresholds per field
- Domain controlled-vocab linkers (medical → UMLS; legal → LEI/Westlaw; industrial → ECLASS; financial → CIK/CUSIP)
- Reviewer routing rules
- Conformance authority (which validator to run)

**The non-configurable backbone:**
- Pipeline stages 1-11 (stable)
- Pydantic typed contracts at every boundary
- Provenance schema (immutable from day one)
- Confidence propagation (immutable)

---

# WHEN IT'S NOT YOUR USE CASE

Honest scoping. If your factory serves:

**Audio / voice / call transcripts** — different pipeline (Whisper + diarization + endpoint detection + speech-LM). Some stages overlap (extract, reconcile, validate); parse/chunk/enrich are voice-specific. Not covered here.

**Video / multimodal scene** — also different (frame sampling + scene parsing + video VLM). Out of scope.

**Code / codebase understanding** — different (AST parsing, language servers, Sourcegraph Cody / Aider patterns). Out of scope.

**Live web browsing / web-as-data** — agentic retrieval over live web (Parallel, Comet, browser agents). Different pipeline.

**Sensor / time-series telemetry** — anomaly detection + forecasting + symbolic regression. Different field entirely.

If your factory needs to span multiple of these (e.g., clinical protocols + transcribed investigator interviews + scanned consent forms), build a *router at Stage 1* that dispatches to modality-specific sub-pipelines. The Stage 8-11 output schema is the unification point.

---

# WHAT I'M HONESTLY STILL THIN ON

For the artifact to be complete, the following would need additional research:

- **Synthetic data generation specifically for unstructured-AI training under privacy constraints** (PHI for clinical, attorney-client for legal, trade secrets for industrial, MNPI for financial)
- **Federated learning across customers/sponsors** without raw data sharing (pharma uses this)
- **Active learning loops for reviewer corrections** at corpus scale — how to update extractors from per-field reviewer signals
- **Tabular / table-specific extraction** beyond what Reducto/MinerU publish (e.g., financial statement reconstruction, BoM hierarchy preservation, SoA cross-row dependencies)
- **Multilingual extraction at production scale** (EU trials, Japan/China filings, multi-jurisdiction contracts)
- **Edge / on-device deployment** for sensitive-data extraction without cloud
- **Streaming / continuous ingestion** vs batch (Kafka-style pipelines for live document streams)

These are real gaps. If you commit to a vertical, narrow the project to a fillable scope, and want depth on any of these, ask and I'll dig.

---

# PRIMARY-SOURCE INDEX (selected, by stage)

## Stage 2 — Parse
- reducto.ai/blog/document-parsing-unstructured-files
- reducto.ai/blog/rd-tablebench
- hebbia.com/blog/goodbye-rag-how-hebbia-solved-information-retrieval-for-llms
- databricks.com/blog/why-frontier-agents-cant-read-documents-and-how-were-fixing-it
- mistral.ai/news/mistral-ocr
- arxiv.org/abs/2509.22186 (MinerU 2.5)
- arxiv.org/abs/2510.18234 (DeepSeek-OCR)
- arxiv.org/abs/2503.11576 (SmolDocling)
- jxnl.co/writing/2025/09/11/why-most-document-parsing-sucks-adit-reducto/

## Stage 3 — Chunk
- arxiv.org/abs/2603.18409 (TopoChunker)
- arxiv.org/abs/2401.18059 (RAPTOR)
- arxiv.org/abs/2502.14802 (HippoRAG2)
- arxiv.org/html/2410.05779v1 (LightRAG)
- arxiv.org/abs/2404.16130 (MS GraphRAG)
- microsoft.com/research/blog/lazygraphrag

## Stage 4 — Enrich
- github.com/DS4SD/DocLayNet
- tensorlake.ai/blog/announcing-page-classifications
- arxiv.org/pdf/2603.23533 (MDKeyChunker)
- arxiv.org/pdf/2410.19572 (ChunkRAG)

## Stage 5 — Encode
- voyage.ai/blog
- jina.ai/embeddings
- arxiv.org/abs/2407.01449 (ColPali)
- vickiboykis.com/2025/09/01/how-big-are-our-embeddings-now-and-why/

## Stage 6 — Index
- neo4j.com/docs/neo4j-graphrag-python
- oxigraph.org
- spec.edmcouncil.org/fibo
- industrialdigitaltwin.org (AAS)
- thedataquarry.com/blog/embedded-db-2/ (Kùzu archival)

## Stage 7 — Retrieve
- hebbia.com/blog/goodbye-rag-how-hebbia-solved-information-retrieval-for-llms
- microsoft.github.io/graphrag/query/drift_search/
- arxiv.org/abs/2408.04948 (HybridRAG BlackRock+NVIDIA)
- trychroma.com/research/context-rot

## Stage 8 — Extract
- python.useinstructor.com
- docs.boundaryml.com/home
- boundaryml.com/blog/schema-aligned-parsing
- arxiv.org/pdf/2411.15100 (XGrammar)
- arxiv.org/abs/2501.10868 (JSONSchemaBench)
- arxiv.org/abs/2311.08526 (GLiNER)
- arxiv.org/abs/2504.00676 (GLiNER-BioMed)
- arxiv.org/abs/2507.18546 (GLiNER2)
- numind.ai/blog (NuExtract)
- arxiv.org/abs/2505.20650 (FinTagging)
- developers.googleblog.com/introducing-langextract

## Stage 8 — Linguistic ops
- arxiv.org/abs/2209.04280 (fastcoref)
- arxiv.org/html/2502.12509v1 (LegalCore)
- arxiv.org/abs/2605.16984 (CRAC 2026)
- pmc.ncbi.nlm.nih.gov/articles/PMC5863758/ (DEEPEN negation)
- arxiv.org/pdf/2505.22946 (NegVQA)

## Stage 9 — Reconcile
- github.com/CogStack/MedCAT
- github.com/allenai/scispacy
- drivendata.co/blog/snomed-ct-entity-linking-challenge-winners
- github.com/facebookresearch/BLINK
- github.com/amazon-science/ReFinED
- arxiv.org/abs/2511.10887 (MedPath)
- eclass.eu/support/technical-specification/structure-and-elements/irdi
- gleif.org/en/about-lei
- blog.opencorporates.com/2025/10/01/legal-entity-knowledge-graphs/

## Stage 10 — Validate
- anthropic.com/news/introducing-citations-api
- tensorlake.ai/blog/announcing-citations
- github.com/vectara/hallucination-leaderboard
- patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model
- arxiv.org/abs/2305.14251 (FActScore)
- github.com/cdisc-org/cdisc-rules-engine

## Stage 11 — Memory
- arxiv.org/abs/2501.13956 (Zep/Graphiti)
- arxiv.org/abs/2504.19413 (Mem0)
- arxiv.org/abs/2603.19935 (Memori)
- letta.com/blog/sleep-time-compute
- mem0.ai/blog/state-of-ai-agent-memory-2026

## Schema targets (per vertical)
- cdisc.org/standards/data-exchange/usdm (Clinical USDM)
- federalregister.gov/documents/2026/05/22/2026-10295 (ICH M11 final, May 2026)
- akomantoso.org (Legal Akoma Ntoso)
- groups.oasis-open.org/communities/community-home?CommunityKey=06fcdf80-d4e9-4cda-bc70-2d6c1cd3d3b3 (LegalRuleML)
- industrialdigitaltwin.org (AAS / Industrial)
- isa.org/standards-and-publications/isa-standards/isa-standards-committees/isa95 (ISA-95)
- spec.edmcouncil.org/fibo (Financial FIBO)
- iso20022.org (ISO 20022)
- fasb.org (XBRL US-GAAP)

---

**End of pipeline-focused artifact. ~9,500 words. The companion broader-landscape doc is at `unstructured_ai_landscape_2026.md`. Use this one as the engineering reference for building; use the other one as the context for understanding why the field is where it is.**
