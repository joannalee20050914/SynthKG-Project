# SynthKG: Scaling Knowledge Graph Construction from Social Media Transcripts

A Text-to-Graph pipeline designed to construct robust Knowledge Graphs (KGs) from fragmented, colloquial social media short videos (specifically Instagram fitness reels) and apply them to a downstream Graph RAG question-answering application. This project adapts the methodology of the ICLR 2026 paper *SynthKG: Scaling Knowledge Graph Construction through Synthetic Data Generation and Distillation*.

---

##  Project Overview & Pipeline Architecture

Social media transcripts are inherently noisy, highly fragmented, colloquial, and heavily dependent on uncaptured visual context (e.g., "do it like this"). Direct triple extraction on raw short-video text fails catastrophically due to modality gaps and text ambiguity. 

To resolve this, we built a 3-stage In-Context Learning (ICL) and Graph RAG pipeline:

```
[Social Media Video] 
       │
       ▼  (Stage 1: Data Preparation)
 1. Domain-Adaptive Transcription (yt-dlp + mlx_whisper + FITNESS_PROMPT)
 2. Text Decontextualization (Llama-3.3-70b-versatile with Strict Guardrails)
 3. Quality Control Filter (ROUGE-1 F1 Score >= 0.45 Calibration)
       │
       ▼  (Stage 2: Structured Triplet Extraction)
 4. Zero-shot / Few-shot Extraction (Enforced Pydantic Schema Validation)
 5. Entity Normalization & Safe Compression (Closed-Ontology Mapping)
       │
       ▼  (Stage 3: Integration, Visualization & Graph RAG)
 6. Automated Pipeline Batch Processing (NetworkX + PyVis HTML Output)
 7. Downstream Graph RAG QA (Embedding Retrieval -> Subgraph Traversal -> Gemini Reranking & Generation)
```

---

## Core Modules & Implementation

### 1. Data Preparation & Domain Adaptation (`/data`)
* **`ig_whisper_integrator.py`**: Downloads audio from video URLs via `yt-dlp` and performs local automatic speech recognition using `mlx_whisper` (with `whisper-large-v3-mlx`). To combat recognition errors on domain-specific terminology, we inject a targeted `FITNESS_PROMPT` containing mandatory keywords (e.g., *TDEE, Hypertrophy, Squat, Deadlift*).
* **`decontextualize.py`**: Leverages `llama-3.3-70b-versatile` (at temperature 0.1) to convert raw first/second-person text into objective third-person narratives. It strictly filters out unresolvable visual indicators to prevent hallucinations.
* **`evaluate_decontext.py`**: Implements a ROUGE-1 F1 lexical-overlap quality gate. Recognizing that decontextualization naturally introduces valid semantic expansion, we calibrated the baseline academic threshold down from 0.70 to a domain-adapted **0.45**, preserving high-yield valid data points.

### 2. Structured Triplet Extraction (`/extraction`)
* **`schema.py`**: Defines strict Pydantic structures (`Triplet`, `TripletList`) to guarantee that every single tuple extracted strictly contains non-empty `source_post_id`, `head`, `relation`, `tail`, and a supporting factual `proposition`.
* **`zero_shot.py` & `few_shot.py`**: Implements the rule-based core and the examplar-driven extraction logic. Features highly optimized negative constraints to prevent "Relation Sprawl", restrict entity lengths to $\le 5$ words, and enforce literal syntactic compression (avoiding anatomical hallucinations like mapping "side neck muscle" to "sternocleidomastoid" unless explicitly written).
* **`extractor.py`**: The standardized unified module interface handling strategy switching, error logging, and JSON parsing cleanups.

### 3. Pipeline Integration & Visualization (`/pipeline`)
* **`main.py`**: The central automated pipeline runner that maps the evaluated cleaned posts across both zero-shot and few-shot modules, generating unified graph databases (`graph_zeroshot.json`, `graph_fewshot.json`).
* **`visualize_graph.py`**: Loads structured graph outputs into `NetworkX` and exports fully interactive, dynamic topological visualizations via `PyVis` to interactive HTML files.

### 4. Downstream Application: Graph RAG (`/pipeline`)
* **`graph_rag.py` & `graph_rag_demo.py`**: Implements an advanced multi-hop retrieval and generation architecture:
    1.  **Vector Search**: Retrieves the Top-M ($M=10$) most relevant propositions based on embedding cosine similarity.
    2.  **Entity Linking**: Automatically maps extracted entities from the user query to graph keys.
    3.  **Subgraph Construction**: Builds a concise localized subgraph from the intersecting components.
    4.  **N-Hop Traversal**: Computes a 2-hop graph neighborhood expansion to surface hidden contextual linkages.
    5.  **LLM Reranking & Generation**: Utilizes the Google Gemini API to rerank the dense context down to the Top-K ($K=5$) text chunks and outputs grounded, 1-2 sentence precise answers.

---

##  Experimental Insights & Key Findings

* **Cost–Quality Trade-off**: The Few-shot strategy induced a **~75% increase in average input tokens** and elevated synchronous sequential latency (peaking at 17.88s vs. 1.403s in zero-shot). However, it achieved a **~29% increase in average output tokens**, yielding significantly denser graph components.
* **Zero Hallucination Rate**: Manual validation of ~300 triplets across both strategies verified a **0% hallucination rate**, confirming that the text-only Decontextualization phase successfully isolated the pipeline from modality-related ambiguity.
* **Ontology Standardization**: Enforcing Pydantic schemas along with closed-vocabulary ontology constraints reduced the relation error rate down to **1.00%** under the Few-shot setting, compared to 4.14% in the zero-shot baseline.
* **Infrastructure Bottlenecks**: Stability analysis revealed that pipeline disruptions were entirely associated with API Rate Limiting (Error 429), indicating that free-tier API frameworks heavily benefit from asynchronous retry/cooldown layers.

---
