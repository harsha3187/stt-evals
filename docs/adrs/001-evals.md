# ADR — EvaluationOnly for ASR, Diarization, Retrieval & RAG (English + Arabic, Medical Conversations)

```md
status: "Accepted"
date: "2025-10-21"
authors: "ISE Oryx Crew"
tags: ["evaluations", "ASR", "ai", "speech-to-text", "speech-recognition", "RAG"]
supersedes: ""
superseded_by: ""
```

---

## Objectives & Design Decisions

### Design Objectives

  &#x2714;  One reproducible evaluation standard for ASR, speaker diarization, retrieval, and RAG answer quality with no model training/pre-training/few-shot learning .

  &#x2714;  **Bilingual coverage** (English & Arabic, multidialect) for medical conversational scenarios (patient–doctor dialogues, longform talk, overlap).

  &#x2714;  **Cost/licensing compliance:** free/open datasets and OSS tools (Common Voice, LibriSpeech, QASR, MGB2/3, VoxConverse; MTSDialog, MedDialogEN, AHQAD, Arabic Medical Dialogue, MedArabiQ)

  &#x2714;  **Frameworks:** Use proven, open-source evaluation tools that work consistently across all our tasks and datasets. This ensures our results can be easily reproduced and compared by other teams.

  &#x2714;  **Evaluation-only scope:** no finetuning/domain adaptation/prompt learning; zeroshot on fixed test datasets.

---

## 1) Executive Summary

### What we evaluate (zeroshot):

- **ASR:** Mixed EN/AR speech (read + broadcast/YouTube; multidialect AR)
- **Diarization:** Overlap-rich multispeaker audio (news/debates) + AR broadcast with RTTM (Rich Transcription Time Marked) format.
- **Retrieval:** Rank & coverage metrics on labelled sets; per-chunk context quality for RAG
- **RAG answers:** Groundedness, relevance, completeness, coherence, fluency using RAGAS + LLMasJudge

---

## 2) Evaluation Metrics — Rationale & Comparison
### 2.1 Speech-to-Text (Core Accuracy)

_All computed after uniform normalization (lowercase, Unicode NFKC, punctuation policy; optional Arabic diacritic stripping) and minimum-edit alignment._

| Metric | Measures | Rationale | Notes / Source |
|--------|----------|-----------|----------------|
| WER | Word substitutions/insertions/deletions | Community standard | [blog](https://medium.com/@ramadhanimassawe14/understanding-and-calculating-word-error-rate-wer-in-automatic-speech-recognition-using-python-661f18b518a5) |
| MER | Match-based word error | Diagnostic complement to WER | [blog](https://lightning.ai/docs/torchmetrics/stable/text/match_error_rate.html) |
| WIP / WIL | Word info preserved / lost | Info-centric diagnostics | [blog- WIP](https://lightning.ai/docs/torchmetrics/stable/text/word_info_preserved.html) [blog- WIL](https://stackoverflow.com/questions/76862465/why-is-word-information-lost-wil-calculated-the-way-it-is) |
| CER | Character-level edits | Sensitive to Arabic orthography | [blog](https://medium.com/@tam.tamanna18/deciphering-accuracy-evaluation-metrics-in-nlp-and-ocr-a-comparison-of-character-error-rate-cer-e97e809be0c8) |
| LD (word/char) | Raw edit distance | Transparent absolute difficulty | [blog](https://en.wikipedia.org/wiki/Levenshtein_distance) |

---

### 2.2 Diarization
| Metric | Measures | Rationale | Notes / Source |
|--------|----------|-----------|----------------|
| DER | Missed speech + false alarm + speaker confusion | NIST/DIHARD standard | [blog](https://arxiv.org/html/2506.05796v1) |
| JER | 1 − avg. Jaccard over mapped speakers | Overlap-aware complement to DER | [blog](https://picovoice.ai/blog/speaker-diarization) |

---

### 2.3 Retrieval (for RAG)

| Metric | Measures | Why we need it | Notes / Source |
|--------|----------|----------------|----------------|
| nDCG@K | Rank quality with graded gains | Rewards early relevant hits | [blog](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems) |
| MRR@K | Speed to first relevant hit | Measures user effort | [blog](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems) |
| Precision@K | Share of top-K that are relevant | Measures exactness | [blog](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems) |
| Recall@K | Share of all relevant retrieved in top-K | Measures coverage | [blog](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems) |
| MAP@K | Mean Avg. Precision across queries | Rank-aware overall precision | [blog](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems) |
| Hit Rate@K | Binary: any relevant in top-K | Success signal for RAG retrievers | [blog](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems) |
| Per-chunk relevance | Relevance of each retrieved chunk | Tracks context precision/utilisation | [blog](https://www.evidentlyai.com/blog/open-source-rag-evaluation-tool) |

---

### 2.4 Answer Quality (RAG / QA / Summarisation)

| Metric | Measures | Why we need it | Notes / Source |
|--------|----------|----------------|----------------|
| LLMJudge relevance | How directly & completely the answer addresses the question | Captures task fit beyond lexical overlap | [Microsoft](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/evaluation/g-eval-metric-for-summarization), [arxiv](https://arxiv.org/abs/2412.05579) |
| Groundedness | Each claim is evidence-backed by retrieved passages | Controls hallucination at claim level | [blog](https://www.deepset.ai/blog/rag-llm-evaluation-groundedness) |
| Faithfulness to retrieved context | No contradiction/overgeneralisation vs. sources | Ensures alignment to citations | [RAGAS](https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/available_metrics/faithfulness.md) |
| Coherence & Fluency | Readability, organisation, grammar | User-perceived quality & trust | [Microsoft](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/evaluation/g-eval-metric-for-summarization) |

---

## 3) Datasets (Free/Open) — English & Arabic

### 3.1 Speech (ASR & Diarization)

| Dataset | Lang(s) | Size / Facts | Domain / Labels | License / Access | Why here |
|--------|---------|--------------|------------------|------------------|----------|
| Common Voice Mozilla | EN, AR (+135) | 33,815 h, 137 languages | Read + spontaneous; text refs | CC0 (Creative Commons Zero) | Accent-diverse EN/AR WER baselines ([mozilla/huggingface](https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0)) |
| LibriSpeech | EN | ~1,000 h, 16 kHz | Read; strong refs | CCBY 4.0 | Canonical English ASR baseline ([huggingface](https://huggingface.co/datasets/openslr/librispeech_asr)) |
| QASR | AR (MSA + dialects + codeswitch) | ~2,041 h, 1.6M segments | Broadcast; multilayer (speaker, punctuation) | Free (research) | Longform, multidialect Arabic ASR/diarization ([ArabicSpeech/QASR](https://arabicspeech.org/resources/)) |
| MGB2 | AR | ~1,200 h | Broadcast; transcripts | Research challenge | Widely used Arabic ASR eval splits ([ArabicSpeech/QASR](https://arabicspeech.org/resources/)) |
| MGB3 | AREGY | ~16 h; 4× transcripts/file | YouTube, multigenre | Research challenge | Egyptian dialect + multireference scoring ([ArabicSpeech/QASR](https://arabicspeech.org/resources/)) |
| VoxConverse | EN | 50+ h; RTTM labels | YouTube debates/news | CCBY 4.0 | Open diarization benchmark with RTTM ([huggingface](https://huggingface.co/datasets/diarizers-community/voxconverse)) |

> _Note:_ Open, large Arabic medical audio is scarce; we evaluate Arabic speech on broadcast/YouTube (QASR/MGB2/3) and Arabic medical reasoning on text sets below.

> Additional Resource : ASR leaderboard arxiv ([Arxiv](https://arxiv.org/html/2412.13788v1))

---

### 3.2 Medical Text/Dialog (for RAG/QA/Summarisation)

| Dataset | Lang | Size / Facts | Task Fit | License / Access |
|--------|------|---------------|----------|------------------|
| MTSDialog | EN | 1.7k dialogs + clinician summaries (+3.6k aug) | Dialog → SOAP/summary | Open ([GitHub](https://github.com/abachaa/MTS-Dialog)) |
| MedDialogEN | EN | ~260k doctor–patient dialogs | QA/dialog relevance | Open ([huggingface](https://huggingface.co/datasets/UCSD26/medical_dialog)) |
| AHQAD | AR | ~808k Arabic healthcare Q&A; 90 categories | AR medical QA/RAG | CCBY 4.0 (Mendeley)([kaggle](https://www.kaggle.com/datasets/abdoashraf90/ahqad-arabic-healthcare-q-and-a-dataset)) |
| Arabic Medical Dialogue | AR | ~3.6k dialogs | AR dialog evaluation | MIT ([huggingface](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations)) |
| MedArabiQ | AR | 7 Arabic medical tasks | AR medical reasoning accuracy | Open ([paper/repo](https://github.com/nyuad-cai/MedArabiQ)) |

---

## 4) Frameworks (Open Source)

| Layer | Framework | License | What we compute | Notes |
|-------|-----------|---------|------------------|-------|
| ASR metrics | jiwer | Apache 2.0 | WER, MER, WIP, WIL, CER | Minimum-edit via RapidFuzz backend ([github](https://github.com/beir-cellar/beir/wiki/Metrics-available)) |
| Edit distance | RapidFuzz | MIT | LD & NLD (word/char) | Length-robust ([deepeval](https://deepeval.com/docs/metrics-llm-evals)) |
| Diarization | pyannote.metrics | MIT | DER/JER from RTTM | Research standard toolkit ([lightning.ai](https://lightning.ai/docs/torchmetrics/stable/text/word_info_preserved.html)) |
| Diarization | NIST dscore | BSD-2-Clause | DER/JER (+ breakdown) | DIHARD reference scorer ([lightning.ai](https://lightning.ai/docs/torchmetrics/stable/text/word_info_lost.html)) |
| Retrieval | BEIR + pytrec_eval | Apache 2.0 | nDCG/MRR/Precision/Recall/MAP/Hit@K | BEIR wiki & pytrec_eval docs ([github](https://github.com/beir-cellar/beir/wiki/Metrics-available)) |
| RAG | RAGAS | MIT | Faithfulness, Answer Relevance, Context Precision/Recall/Utilisation | Per-chunk scoring & groundedness ([docs.ragas.io](https://docs.ragas.io/en/v0.1.21/concepts/metrics/)) |
| LLM-as-Judge | GEval style | Various | Rubric-based relevance/completeness/coherence/fluency | Microsoft Learn & community guides ([learn.microsoft.com](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/evaluation/g-eval-metric-for-summarization)) |

---

## 5) Evaluation Matrix (ZeroShot, Bilingual)

### 5.1 Speech

- **ASR (English):** LibriSpeech; Common Voice EN → WER, MER, WIP, WIL, CER, LD(word/char), NLD(word/char)  
- **ASR (Arabic):** QASR; MGB2; MGB3 (multireference) → same metric set; MGB3 enables multireference WER  
- **Diarization:** VoxConverse (EN RTTM) and QASR speaker segments (when available) → DER, JER (+ missed/FA/confusion)

### 5.2 Retrieval & RAG

- **Retrieval:** nDCG@10, MRR@10, Precision@10, Recall@10, MAP@10, Hit@10 on labelled sets (pytrec_eval/BEIR)  
- **RAG Answers (EN):** MTSDialog (dialog→SOAP), MedDialogEN (QA) → LLMJudge relevance, Completeness, Coherence & Fluency, plus Groundedness/Faithfulness (RAGAS)  
- **RAG Answers (AR):** AHQAD, Arabic Medical Dialogue → same answer metrics; MedArabiQ → task accuracy

---

## 6) Normalisation & Alignment Rules (ASR)

- **Text normalisation:** lowercase; Unicode Normalization Form Compatibility Composition (NFKC) ; standardised punctuation; Arabic diacritics stripped (primary runs)  
- **Tokenisation:** whitespace word tokens (word-level); raw codepoints (char-level)  
- **Alignment:** Levenshtein minimum-edit dynamic programming  
  - Report LD and NLD (normalised by max(len(ref), len(hyp))) at both word and character levels  
  - [HuggingFace](https://huggingface.co/learn/audio-course/chapter5/evaluation)

---

## 7) Risks & Mitigations

- **Arabic medical audio scarcity:**  
  - Arabic ASR measured on multidialect broadcast/YouTube (QASR, MGB2/3)  
  - Arabic medical reasoning measured on text (AHQAD, MedArabiQ, Arabic Medical Dialogue)  
  - Document this limitation in each evaluation cycle  

---

## 8) Appendix : Rubric Templates for Evaluation Metrics


### Metric Value Ranges and Rubric Score Mapping

This table maps each evaluation metric to its valid value range and defines how raw metric values correspond to rubric scores (1–4).

| Metric        | Min Value | Max Value | Score 4 (Excellent)         | Score 3 (Good)               | Score 2 (Fair)               | Score 1 (Poor)               |
|---------------|-----------|-----------|-----------------------------|------------------------------|------------------------------|------------------------------|
| WER           | 0         | No upper limit         | < 0.05                      | 0.05–0.15                    | 0.15–0.30                    | > 0.30                       |
| MER           | 0         | No upper limit         | < 0.05                      | 0.05–0.15                    | 0.15–0.30                    | > 0.30                       |
| WIP           | 0         | 1         | > 0.95                      | 0.80–0.95                    | 0.60–0.80                    | < 0.60                       |
| WIL           | 0         | No upper limit         | < 0.05                      | 0.05–0.20                    | 0.20–0.40                    | > 0.40                       |
| CER           | 0         | No upper limit         | < 0.05                      | 0.05–0.15                    | 0.15–0.30                    | > 0.30                       |
| LD (word/char)| 0         | ∞         | < 5 edits                   | 5–15 edits                   | 15–30 edits                  | > 30 edits                   |
| NLD (word/char)| 0        | 1         | < 0.05                      | 0.05–0.15                    | 0.15–0.30                    | > 0.30                       |
| DER           | 0         | 1         | < 0.10                      | 0.10–0.20                    | 0.20–0.30                    | > 0.30                       |
| JER           | 0         | 1         | < 0.10                      | 0.10–0.20                    | 0.20–0.30                    | > 0.30                       |
| nDCG@K        | 0         | 1         | > 0.90                      | 0.75–0.90                    | 0.50–0.75                    | < 0.50                       |
| MRR@K         | 0         | 1         | > 0.90                      | 0.75–0.90                    | 0.50–0.75                    | < 0.50                       |
| Precision@K   | 0         | 1         | > 0.80                      | 0.60–0.80                    | 0.30–0.60                    | < 0.30                       |
| Recall@K      | 0         | 1         | > 0.80                      | 0.60–0.80                    | 0.30–0.60                    | < 0.30                       |
| MAP@K         | 0         | 1         | > 0.80                      | 0.60–0.80                    | 0.30–0.60                    | < 0.30                       |
| Hit Rate@K    | 0         | 1         | = 1                         | ≥ 0.75                       | ≥ 0.50                       | < 0.50                       |

> Note: These thresholds are based on common evaluation practices in ASR, IR, and RAG literature. You can adjust them based on task difficulty or dataset characteristics.

The below section provides rubric-style scoring templates for evaluating ASR, Diarization, Retrieval, and RAG answer quality metrics. Each metric includes a 4-point scale with clear criteria.

---

## ASR Metrics

### Word Error Rate (WER)
Measures substitutions, insertions, and deletions in word-level transcription.

| Score | Label      | Criteria                                      |
|-------|------------|-----------------------------------------------|
| 4     | Excellent  | WER < 5%; near-perfect transcription.         |
| 3     | Good       | WER between 5–15%; minor errors.              |
| 2     | Fair       | WER between 15–30%; noticeable errors.        |
| 1     | Poor       | WER > 30%; significant transcription issues.  |

### Match Error Rate (MER)
Diagnostic complement to WER.

| Score | Label              | Criteria                                |
|-------|--------------------|-----------------------------------------|
| 4     | High match accuracy| Most words correctly matched.           |
| 3     | Moderate accuracy  | Some mismatches but mostly aligned.     |
| 2     | Low accuracy       | Frequent mismatches.                    |
| 1     | Poor accuracy      | Very few correct matches.               |

### Word Info Preserved / Lost (WIP / WIL)
Semantic retention or degradation.

| Score | Label             | Criteria                                |
|-------|-------------------|-----------------------------------------|
| 4     | High preservation | Most semantic content retained.         |
| 3     | Moderate retention| Some semantic loss.                     |
| 2     | Low retention     | Significant semantic degradation.       |
| 1     | High loss         | Most information lost.                  |

### Character Error Rate (CER)
Sensitive to orthography and diacritics.

| Score | Label      | Criteria                              |
|-------|------------|---------------------------------------|
| 4     | Excellent  | Minimal character-level errors.       |
| 3     | Good       | Few character-level errors.           |
| 2     | Fair       | Frequent character-level mistakes.    |
| 1     | Poor       | Severe character-level distortion.    |

### Levenshtein Distance / Normalised LD (LD / NLD)
Raw and scaled edit distance.

| Score | Label             | Criteria                                      |
|-------|-------------------|-----------------------------------------------|
| 4     | Low edit distance | Minimal changes needed to match reference.    |
| 3     | Moderate distance | Some edits required.                          |
| 2     | High distance     | Substantial edits needed.                     |
| 1     | Very high distance| Major divergence from reference.              |

---

## Diarization Metrics

### Diarization Error Rate (DER)
Missed speech, false alarms, speaker confusion.

| Score | Label      | Criteria                                           |
|-------|------------|----------------------------------------------------|
| 4     | Excellent  | DER < 10%; highly accurate speaker segmentation.   |
| 3     | Good       | DER between 10–20%; minor speaker errors.         |
| 2     | Fair       | DER between 20–30%; noticeable confusion.         |
| 1     | Poor       | DER > 30%; unreliable speaker attribution.        |

### Jaccard Error Rate (JER)
Overlap-aware complement to DER.

| Score | Label              | Criteria                                 |
|-------|--------------------|------------------------------------------|
| 4     | High accuracy      | Speaker overlap well captured.           |
| 3     | Moderate accuracy  | Some overlap errors.                     |
| 2     | Low accuracy       | Frequent overlap misattribution.         |
| 1     | Poor accuracy      | Overlap handling is unreliable.          |

---

## Retrieval Metrics

### nDCG@K
Rank quality with graded relevance.

| Score | Label      | Criteria                                      |
|-------|------------|-----------------------------------------------|
| 4     | Excellent  | Highly relevant items ranked early.           |
| 3     | Good       | Relevant items mostly ranked well.            |
| 2     | Fair       | Relevant items scattered in ranking.          |
| 1     | Poor       | Relevant items ranked low or missing.         |

### MRR@K
Speed to first relevant hit.

| Score | Label         | Criteria                                      |
|-------|---------------|-----------------------------------------------|
| 4     | Fast retrieval| First relevant item appears very early.       |
| 3     | Good retrieval| First relevant item appears in top 5.         |
| 2     | Slow retrieval| First relevant item appears late.             |
| 1     | Missed        | No relevant item found.                       |

### Precision@K
Proportion of top-K items that are relevant.

| Score | Label         | Criteria                              |
|-------|---------------|---------------------------------------|
| 4     | High precision| >80% of top-K items are relevant.     |
| 3     | Good precision| 60–80% relevant.                      |
| 2     | Fair precision| 30–60% relevant.                      |
| 1     | Low precision | <30% relevant.                        |

### Recall@K
Proportion of all relevant items retrieved in top-K.

| Score | Label       | Criteria                              |
|-------|-------------|---------------------------------------|
| 4     | High recall | >80% of relevant items retrieved.     |
| 3     | Good recall | 60–80% retrieved.                     |
| 2     | Fair recall | 30–60% retrieved.                     |
| 1     | Low recall  | <30% retrieved.                       |

### MAP@K
Mean Average Precision across queries.

| Score | Label      | Criteria                                      |
|-------|------------|-----------------------------------------------|
| 4     | Excellent  | High precision across all queries.            |
| 3     | Good       | Moderate precision across most queries.       |
| 2     | Fair       | Inconsistent precision.                       |
| 1     | Poor       | Low precision across queries.                 |

### Hit Rate@K
Binary success — any relevant item in top-K.

| Score | Label | Criteria                          |
|-------|-------|-----------------------------------|
| 4     | Hit   | Relevant item found in top-K.     |
| 3     | Near hit | Relevant item just outside top-K. |
| 2     | Weak hit | Relevant item found but low rank. |
| 1     | Miss  | No relevant items found.          |

---

## RAG Answer Quality Metrics

### Relevance
How directly and completely the answer addresses the question.

| Score | Label            | Criteria                                          |
|-------|------------------|---------------------------------------------------|
| 4     | Highly relevant  | Fully addresses the question with clear alignment.|
| 3     | Mostly relevant  | Covers main points but misses some details.       |
| 2     | Partially relevant| Touches topic but lacks clarity or depth.        |
| 1     | Irrelevant       | Does not address the question at all.             |

### Groundedness
Are claims backed by retrieved evidence?

| Score | Label         | Criteria                                      |
|-------|---------------|-----------------------------------------------|
| 4     | Fully grounded| All claims traceable to retrieved passages.   |
| 3     | Mostly grounded| Most claims supported, some gaps.            |
| 2     | Weak grounding| Few claims supported.                         |
| 1     | Ungrounded    | No evidence or hallucinated claims.           |

### Faithfulness
Does the answer contradict or misrepresent the retrieved context?

| Score | Label         | Criteria                                      |
|-------|---------------|-----------------------------------------------|
| 4     | Fully faithful| Accurately reflects retrieved content.        |
| 3     | Mostly faithful| Minor inconsistencies.                       |
| 2     | Partially faithful| Some overgeneralisation or misrepresentation. |
| 1     | Contradictory | Contradicts retrieved evidence.               |

### Completeness
Does the answer cover all key aspects or subquestions?

| Score | Label         | Criteria                                      |
|-------|---------------|-----------------------------------------------|
| 4     | Fully complete| All relevant aspects are addressed.           |
| 3     | Mostly complete| Most aspects addressed.                      |
| 2     | Partially complete| Some aspects missing.                     |
| 1     | Incomplete    | Major parts of the question are missing.      |

### Coherence
Is the answer logically organised and easy to follow?

| Score | Label         | Criteria                                      |
|-------|---------------|-----------------------------------------------|
| 4     | Fully coherent| Well-organised and logically structured.      |
| 3     | Mostly coherent| Clear structure with minor issues.           |
| 2     | Somewhat coherent| Basic structure but hard to follow.        |
| 1     | Incoherent    | Disorganised or confusing structure.          |

### Fluency
Is the answer grammatically correct and readable?

| Score | Label             | Criteria                                      |
|-------|-------------------|-----------------------------------------------|
| 4     | Excellent fluency | Natural, error-free language.                 |
| 3     | Good fluency      | Minor errors, mostly smooth.                  |
| 2     | Fair fluency      | Some errors but readable.                     |
| 1     | Poor fluency      | Frequent grammar issues or awkward phrasing.  |

---

