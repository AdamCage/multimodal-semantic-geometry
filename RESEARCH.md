# Research proposal: Multimodal Semantic Geometry

**Status:** Research proposal / planning. No experiments or results yet.  
**Last updated:** 2026-11-20.

This document describes a working research design, not a preregistration. Dataset selections, checkpoints, budgets, and statistical decisions remain provisional until the pilot protocol is frozen.

## 1. Research question

Do independently trained text, image, environmental-audio, and silent-video encoders preserve similar relationships between the same semantic concepts?

The primary objective is to measure **which relationships agree, at which semantic granularity, and under which information constraints**. The motivating example is whether the relationship between concepts such as cars, motorcycles, trains, and sand dunes is similar across representation spaces. Similarity of one distant pair is insufficient evidence; the study will compare structured sets of relationships.

A secondary objective is to investigate whether the measured geometry can predict the performance and paired-data requirements of a shared multimodal embedding system. This applied stage follows the geometry analysis; it does not replace it.

The project does not assume that all modalities contain the same information or that every concept has a recognizable realization in every modality.

## 2. Position relative to existing work

The Platonic Representation Hypothesis motivates convergence between independently learned representations [1]. Representational Similarity Analysis (RSA) provides an established way to compare spaces through relationships among matched conditions [2], while CKA offers a complementary representation-similarity measure [3].

Recent work makes stronger claims of universal convergence difficult to justify. Null-calibrated analyses distinguish robust neighborhood agreement from weaker distance agreement [4]. Dataset scale, semantic granularity, and many-to-many correspondences also affect observed alignment [5]. Indra representations already use relational profiles across vision, language, and audio [6].

GeoWarP connects shared geometry to unpaired and few-pair alignment [7]. Research on multimodal information structure shows that stronger alignment is not universally beneficial [8]. ImageBind and Omni-Embed-Nemotron already provide broad multimodal embedding capabilities [9, 10].

Consequently, adding modalities, computing distance correlations, or attaching projection heads is not sufficient novelty. Possible contributions must be established through a new reproducible finding, a more informative evaluation protocol, or a predictive method that improves a concrete decision beyond existing baselines.

## 3. Testable hypotheses

| Hypothesis | Question | Planned evidence |
|---|---|---|
| **Metric agreement** | Are numerical inter-concept dissimilarities similar after a prespecified scale treatment? | Held-out metric distortion relative to calibrated nulls and within-model reliability |
| **Ordinal agreement** | Do models agree about which concept pairs are closer? | RSA rank correlations and semantic triplet agreement |
| **Local agreement** | Do concepts have similar neighbors? | Neighborhood overlap across prespecified neighborhood sizes and dataset scales |
| **Information-conditioned agreement** | Does agreement depend on semantic factors observable in both channels? | Independent observability annotations and controlled removal or alteration of information |
| **Predictive utility** | Does initial geometry predict alignment quality or the required pair budget? | Prospective predictions evaluated on held-out concepts, datasets, and encoder families |

These hypotheses are distinct. Neighborhood agreement will not be reported as evidence of equal distances. A positive correlation will not establish a universal representation of reality. Supported null results and systematic failures are valid outcomes.

## 4. Data design

### Pilot scope

The provisional pilot targets **100 concepts with approximately 30 independent source recordings per concept**, yielding about 3,000 sources before exclusions. The initial scope emphasizes recognizable objects and events: barking dogs, musical instruments, moving trains, running engines, pouring water, bells, and applause.

Each usable source may yield audio, silent video, selected image frames, and independently written descriptions. A separate concept-matched condition will draw examples from different sources. The final sample size depends on source availability and annotation quality; this is not a claim that a complete dataset has already been assembled.

Two complementary conditions are required:

1. **Instance-paired:** channels describe the same recorded event. This supports correspondence at the event level.
2. **Concept-matched:** channels use independently sampled instances of the same concept. This tests whether agreement survives removal of shared recording context.

Multiple frames or excerpts from one recording do not count as independent sources. Related recordings and duplicates must remain within one split. Cross-source concept matching must preserve enough information to determine relevance without pretending that every example has one uniquely correct counterpart.

### Observability and semantic granularity

Annotators will record which factors are available in each channel: object, action, material, color, motion, temporal order, and sound source. They should judge a channel without being shown hidden information from the other channels.

Sand dunes paired with generic wind are an ambiguity control, not an ordinary fully observable concept. A spoken category name is a speech/language condition, not an independent environmental-sound observation. Similarly, a still image may identify a guitar without establishing that it is being played.

The dataset should include broad categories, finer distinctions within categories, and selected relational or temporal distinctions. Controlled conditions may remove audio, shuffle frame order, or restrict descriptions to visible or audible content. These interventions require explicit checks that the intended semantic factor changes while relevant alternatives remain controlled.

Candidate resources include VGGSound and AudioSet/AudioCaps for audiovisual material and Clotho for an audio–text replication. Availability, redistribution rights, training overlap, and actual observability will be checked before fixing a manifest. No single candidate is assumed to provide complete four-way supervision.

## 5. Encoders and extraction

The initial comparison will use two families per modality, subject to checkpoint and feasibility checks:

| Modality | Candidate families |
|---|---|
| Text | BGE-M3; Qwen3-Embedding |
| Image | Original DINOv2; pretrained MAE |
| Environmental audio | Pretrained BEATs; AudioMAE |
| Silent video | Pretrained VideoMAE; V-JEPA 2 |

The intended main condition is **no explicit cross-modal alignment training in the selected checkpoints**. This does not imply completely disjoint pretraining data or absence of shared labels, teachers, or sources. Pretrained and supervised-fine-tuned checkpoints must be distinguished. Co-aligned models, including ImageBind and an available modern multimodal retriever, form a separate comparison group [9, 10].

Extraction choices—layer, pooling, input duration, frame sampling, and normalization—will be fixed before test evaluation. Alternative choices are robustness checks, not an unrestricted search for the strongest agreement. Frozen encoders run separately and their features are cached; the pilot does not require joint backbone pretraining.

## 6. Measuring native geometry

For modality `m`, encoder `e`, concept `c`, and example `i`, let `z[m,e,c,i]` be the normalized native embedding. Let `mu[m,e,c]` be its concept mean. The initial representational dissimilarity matrix is:

```text
D[m,e,a,b] = 1 - cosine(mu[m,e,a], mu[m,e,b])
```

Every matrix has the same concept axes even when encoders have different feature dimensions. No shared coordinate basis is needed. Direct cosine comparisons between unrelated native encoders are not meaningful merely because their dimensions match.

The primary analysis will use RSA rank correlation; complementary analyses cover neighborhood overlap, metric distortion, and calibrated CKA [2–4]. Scale fitting, when needed, uses designated development concepts and is evaluated on held-out concepts. Learned projections, CCA, and metric learning are excluded from the native-geometry claim.

Centroids are a starting point. Split-half estimates assess stability; within-concept dispersion, multiple prototypes, and instance-level analyses test whether averaging hides inconsistent fine structure. Instance-level comparisons require valid matching or graded relevance annotations. Independent clustering is a secondary analysis rather than the definition of semantic agreement.

UMAP and t-SNE may illustrate findings. Conclusions will use original-space relationships, not visual resemblance of two-dimensional plots.

## 7. Statistical controls and generalization

The study will include the following controls:

- Split-half reliability within each encoder and comparisons between different encoders of the same modality.
- Concept-identity permutations, applied jointly to rows and columns of one dissimilarity matrix.
- Permutations and evaluation within broad semantic groups, separating fine agreement from coarse taxonomy.
- Dimension-matched random features and, where useful, untrained encoders.
- Explicitly co-aligned multimodal models as contextual baselines.
- Paired-instance versus independently sampled concept-matched data.
- Single-frame, frame-averaged, and temporal-video representations.

Pairwise distances are dependent observations. Uncertainty estimation will respect concepts, recordings, and shared sources; individual distance cells will not be treated as independent samples. If layer or metric aggregation is used, null calibration must reproduce the final selection procedure [4]. Primary outcomes and multiple-comparison treatment will be recorded before confirmatory evaluation.

Source-held-out evaluation tests new recordings; concept-held-out evaluation tests new meanings; family-held-out evaluation tests transfer beyond previously studied encoders. Dataset-held-out replication will be required for broader claims. Model pairs sharing an encoder will not be counted as independent evidence of predictive generalization.

There is no universal correlation threshold for success. Effect sizes must be interpreted against null distributions, reliability, semantic granularity, and the sampling scope.

## 8. Secondary stage: a shared multimodal embedding system

If native-geometry findings are reproducible, the next stage will connect frozen encoders through linear or small nonlinear projection heads. The provisional pair budgets are **16, 64, 256, and 1,024**, with repeated sampling and reporting of both pair count and concept coverage.

Candidate objectives include multi-positive contrastive learning and preservation of reliable relational structure. Missing or uninformative channels must be masked appropriately. Modality-specific information should be evaluated explicitly; a shared/private decomposition is a candidate design, not a claimed new contribution.

Required comparisons include a simple linear alignment, a small MLP, GeoWarP where applicable [7], and an available pretrained multimodal model. A brief, fixed-budget trial alignment is also a baseline for the proposed diagnostic: it may be cheaper and more accurate than an elaborate geometry analysis.

Evaluation will cover all feasible directed retrieval pairs, using Recall@K and a graded-relevance metric such as nDCG where appropriate. Multiple valid matches must be represented. Within-modality tasks will check whether alignment loses useful distinctions. Success after learned alignment will not be presented as evidence that native spaces were already identical.

Prospective quality or budget predictions must be fixed before observing the corresponding alignment outcomes. Diagnostic labels, matched concepts, inference, annotation, and fitting costs all count toward the budget. RSA and ordinary cross-model CKA require matched observations; their use cannot be described as fully unpaired prediction. Pair-free diagnostics remain a separate, more demanding question [7].

## 9. Milestones and decision criteria

1. **Freeze the pilot protocol:** confirm sources, checkpoints, observability labels, splits, metrics, and baselines.
2. **Establish reliable geometry measurements:** determine which agreement survives source changes, null calibration, and within-category analysis.
3. **Attempt independent replication:** test held-out concepts, encoder families, and another dataset before generalizing.
4. **Evaluate applied utility:** determine whether the findings predict alignment behavior or improve a concrete quality–cost tradeoff.

Expansion is justified by an effect that either explains a previously unresolved discrepancy or supports useful predictions beyond simple alternatives. A finding of coarse agreement alone should be reported as a bounded replication. A negative result is informative when it identifies the failed claim and rules out measurement instability or unavailable input information.

At this stage, the repository contains documentation only. Code, data manifests, trained heads, and results will be added only in later work.

## References

1. Huh et al. **The Platonic Representation Hypothesis.** ICML, 2024. [Paper](https://proceedings.mlr.press/v235/huh24a.html).
2. Kriegeskorte, Mur, and Bandettini. **Representational Similarity Analysis — Connecting the Branches of Systems Neuroscience.** Frontiers in Systems Neuroscience, 2008. [Paper](https://doi.org/10.3389/neuro.06.004.2008).
3. Kornblith et al. **Similarity of Neural Network Representations Revisited.** ICML, 2019. [Paper](https://proceedings.mlr.press/v97/kornblith19a.html).
4. Gröger, Wen, and Brbic. **Revisiting the Platonic Representation Hypothesis: An Aristotelian View.** ICML, 2026. [Paper](https://proceedings.mlr.press/v306/groger26a.html).
5. Koepke et al. **Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale.** arXiv preprint, 2026. [Paper](https://arxiv.org/abs/2604.18572).
6. Lu et al. **The Indra Representation Hypothesis for Multimodal Alignment.** NeurIPS, 2025. [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e63e91b27d16867cdd09c4722f3b0942-Abstract-Conference.html).
7. Schnaus et al. **Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data.** arXiv preprint, 2026. [Paper](https://arxiv.org/abs/2610.09411).
8. Tjandrasuwita et al. **Understanding the Emergence of Multimodal Representation Alignment.** ICML, 2025. [Paper](https://proceedings.mlr.press/v267/tjandrasuwita25a.html).
9. Girdhar et al. **ImageBind: One Embedding Space To Bind Them All.** CVPR, 2023. [Paper](https://arxiv.org/abs/2305.05665).
10. Xu et al. **Omni-Embed-Nemotron: A Unified Multimodal Retrieval Model for Text, Image, Audio, and Video.** arXiv preprint, 2025. [Paper](https://arxiv.org/abs/2510.03458).
