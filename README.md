# skatr-transfer

Cross-simulator transfer of self-supervised SKATR summaries for 21cm parameter inference.

Paper: [arXiv:2608.26354](https://arxiv.org/abs/2608.26354) — *Cross-simulator transfer with foundation model summaries: Towards robust SKA-era reionization inference*.

## Based on SKATR

This repository extends [heidelberg-hepml/skatr](https://github.com/heidelberg-hepml/skatr), a general-purpose summary transformer for lightcones observed by the Square Kilometre Array. SKATR is a 3D Vision Transformer pretrained label-free with a Joint Embedding Predictive Architecture (JEPA) objective; it maps a lightcone to a 360-dimensional summary that downstream models consume in place of the raw data.

> A. Ore, C. Heneka and T. Plehn, *SKATR: A self-supervised summary transformer for SKA*, [SciPost Phys. **18**, 155 (2025)](https://scipost.org/SciPostPhys.18.5.155).

The upstream repository documents the general experiment framework. Only what differs is described below.

## The transfer experiments

SKATR is pretrained once on cheap semi-numerical `21cmFAST` lightcones and then frozen. Here that frozen encoder is applied to lightcones from a physically distinct hydrodynamical radiative-transfer simulator (Loreli II) with a disjoint parameter set, first noiseless and then under SKA AA\* thermal noise with foreground-wedge removal. The encoder is shown no target data, no target parameters and no noise at any point.

This repository covers the encoders and the summaries they produce: JEPA pretraining, the supervised ViT baselines used for comparison, the Loreli data pipeline and preprocessing, and the dataset-embedding scripts in `transfer/`. Posterior inference on the resulting summaries is run with the conditional flow matching head of [EoRFlow](https://github.com/astro-ML/EoRFlow).

## Usage

```
conda env create -f env.yaml
python main.py -cn <config> data.dir=/path/to/lightcones
```

Experiments are configured with [Hydra](https://hydra.cc/docs/intro/). Defaults live in `config/default.yaml` and any setting can be overridden from the command line.

| Config | Pipeline |
| :--- | :--- |
| `pretraining` | SKATR, self-supervised JEPA pretraining on 21cmFAST |
| `supervised_21cmfast_vit` | ViT(21cmFAST), supervised source baseline |
| `supervised_loreli_vit` | ViT(Loreli), supervised in-domain baseline |
| `supervised_loreli_vit_RotateReflect` | ViT(Loreli, R+R), augmented in-domain baseline |
| `regression_loreli_transfer` | regression on frozen SKATR summaries |

## Models

Pretrained weights will be released on Zenodo and linked here.

## Citation

```bibtex
@article{pietschke2026transfer,
  title         = {Cross-simulator transfer with foundation model summaries:
                   Towards robust {SKA}-era reionization inference},
  author        = {Pietschke, Yannic and Heneka, Caroline and Ore, Ayodele
                   and Meriot, Romain},
  year          = {2026},
  eprint        = {2608.26354},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO},
}

@article{ore2025skatr,
  title   = {{SKATR}: A self-supervised summary transformer for {SKA}},
  author  = {Ore, Ayodele and Heneka, Caroline and Plehn, Tilman},
  journal = {SciPost Phys.},
  volume  = {18},
  pages   = {155},
  year    = {2025},
  doi     = {10.21468/SciPostPhys.18.5.155},
}
```
