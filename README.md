# PHAROS

Ontology-based phenotype ranking of rare diseases with language model re-ranking.

Code, simulated patients, saved model outputs and result tables accompanying the manuscript
*How much does a language model add to ontology-based phenotype ranking of rare diseases?* (in preparation).

## Overview

Candidate diseases are ranked from a patient's Human Phenotype Ontology (HPO) terms with a rarity-weighted
semantic-similarity scorer. The top 20 candidates form a shortlist that a locally run language model
re-ranks under four prompt designs. Performance is evaluated on simulated patients scored against the full
knowledge base (held-out diseases) and, as a secondary analysis, on real published cases scored with each case's
own source publication removed from the annotations (leave-source-out). Diseases were divided into tuning (85) and
held-out (198) sets before held-out evaluation. The split and the checksums of the files frozen before that
evaluation are in `config/`.

## Repository layout

~~~
src/        analysis package
tests/      unit tests
slurm/      job scripts for the language model runs
config/     disease split; SHA-256 checksums of the frozen files
cache/      simulated patients, disease profiles, processed real cases, language model outputs
results/    evaluation outputs reported in the manuscript
figures/    figure scripts
data/       checksums of third-party inputs
~~~

## Requirements

Package versions are listed in `requirements-frozen.txt`. The language model was Qwen3.6-27B (Q8_0 GGUF) served with
llama.cpp build 11670 (commit b76b13379) on one NVIDIA H100 GPU, with temperature 0, seed 0, reasoning disabled and
grammar-constrained replies.

## Data

| Resource | Version | Source | In this repository |
|---|---|---|---|
| HPO ontology (`hp.obo`) | release 2026-09-01 | https://github.com/obophenotype/human-phenotype-ontology/releases | checksum |
| HPO annotations (`phenotype.hpoa`) | version 2026-09-02 | https://github.com/obophenotype/human-phenotype-ontology/releases | checksum |
| Phenopacket Store | 10,377 phenopacket files | https://github.com/monarch-initiative/phenopacket-store, doi:10.5281/zenodo.13168726 | processed cases (BSD 3-Clause) |
| Phenomizer syndrome tables | supplement to Köhler et al., Am J Hum Genet 2009;85:457–64 | publisher | checksum |

Licences for third-party material are in `NOTICE.md` and `THIRD_PARTY_LICENSES/`.

## Reproducing the analysis

Place `hp.obo` and `phenotype.hpoa` in the project root and the Phenopacket Store files in `phenopackets/`.

~~~
python -m src.annotations hp.obo phenotype.hpoa
python -m src.phenopackets
python -m src.profiles
python -m src.split
python -m src.simulate
python -m pytest -q tests/
~~~

Language model runs are submitted with the scripts in `slurm/` (`final.sh` for the held-out evaluation, with
`SET=primary` or `SET=ablation`; `real_llm.sh` for the real cases). Results are then evaluated with

~~~
python -m src.evaluate --which heldout --final
python -m src.real_eval
python -m src.evaluate_llm cache/llm_heldout_rag_ordev.jsonl cache/llm_heldout_model_alone.jsonl
python -m src.moves cache/llm_heldout_rag_ordev.jsonl
~~~

Figures are produced from the files in `results/` and `cache/`:

~~~
python figures/make_figures.py
python figures/make_supp_figures.py
~~~

## Verification

File integrity: `sha256sum -c MANIFEST.sha256`. Checksums of the code, simulated patients and split frozen before
held-out evaluation: `config/freeze_checksums.txt` and `config/freeze_checksums_real.txt`.

## Citation

Please cite the accompanying manuscript and release v1.0 of this repository (see `CITATION.cff`).

## Licence

Code: MIT (`LICENSE`). Data derived from third-party sources retain their original licences.

## Contact

Manan Vij ([@m-vij](https://github.com/m-vij))
