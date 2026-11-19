# Multimodal Semantic Geometry

Investigating which semantic relationships are shared by independently trained text, image, audio, and video encoders, and whether those relationships can help predict the cost of aligning their representations.

**Status: research proposal.** This repository currently contains documentation only. No experiments have been run for this project, and no empirical results or trained models are claimed.

## Research question

Do independently trained encoders preserve similar relationships between the same concepts across modalities? If they do, does that agreement make it easier to build a useful shared embedding space?

The study will distinguish four claims:

- **Local agreement:** concepts have similar nearest neighbors.
- **Ordinal agreement:** distances between concepts have similar rankings.
- **Metric agreement:** numerical distances agree after a predefined normalization.
- **Partial agreement:** relationships are shared only for semantic factors observable in the compared modalities.

These claims will be evaluated separately. Similar neighborhoods do not establish identical geometry, and successful learned alignment does not establish that representations were aligned beforehand.

## Planned approach

1. Build a curated pilot with text, images, environmental audio, and silent video, including cases where information is missing from one modality.
2. Extract frozen representations from multiple encoder families and compare concept-level distance matrices and neighborhood structure.
3. Test robustness to sampling, coarse category structure, source overlap, representation extraction, and chance similarity.
4. Evaluate small alignment heads on held-out concepts and sources, then investigate whether pre-alignment measurements predict performance and paired-data requirements on new encoder families.

The initial pilot targets approximately 100 concepts and 30 independent source recordings per concept. These are planning targets, not collected data. Dataset availability, checkpoints, and the evaluation protocol will be finalized before the confirmatory experiments.

## Scope and contribution

The project builds on the [Platonic Representation Hypothesis](https://proceedings.mlr.press/v235/huh24a.html), subsequent studies of representational similarity, and existing multimodal alignment methods. Cross-modal geometric agreement and multimodal embedding models are established research topics.

The intended contribution is a reproducible account of the limits of shared semantic structure and, if supported by experiments, a useful way to anticipate alignment success or data requirements. Novelty and practical gains remain to be demonstrated against existing methods and simple pilot-training baselines.

## Documents

- [RESEARCH.md](RESEARCH.md) — motivation, hypotheses, experimental design, controls, evaluation criteria, limitations, and references.

Implementation, datasets, experiments, and results will be added in later stages.
