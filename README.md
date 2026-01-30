<div align="center">

# Cognitive Prior-Guided Evidence Decoupling for Remote Sensing Image Change Captioning

</div>

## Welcome
This repository contains the PyTorch implementation of our RSICC method for the paper:

**Cognitive Prior-Guided Evidence Decoupling for Remote Sensing Image Change Captioning**.

---

## Overview
Remote Sensing Image Change Captioning (RSICC) aims to describe semantic changes between bi-temporal remote sensing images with natural language. Existing methods are often distracted by pseudo-changes (e.g., illumination/seasonal variations), leading to inaccurate change localization and hallucinated captions.

Our method establishes a coarse-to-fine prior-guided reasoning pathway:
- **CPPM (Cognitive Prior Prompt Module):** generates a global cognitive prior prompt indicating the macroscopic change state.
- **DEDM (Difference Evidence Decoupling Module):** decouples bi-temporal evidence into **Common Evidence** (stable background) and **Private Evidence** (semantic changes) using a prior-guided decoupling loss.
- **DEFM (Difference Evidence Fusion Module):** fuses structured cues to guide caption generation for accurate change descriptions.

This repository includes code for **training, inference, and evaluation**, as well as the **tokenization/word mapping** used in our experiments.

---

## Installation and Dependencies
Tested with Python 3.9. Please ensure your PyTorch/CUDA versions match your GPU driver if using CUDA.

```bash
conda create -n rsicc python=3.9
conda activate rsicc
pip install -r requirements.txt
```

## Data Preparation (LEVIR-CC)
Firstly, download the image pairs of LEVIR_CC dataset from the [[Repository](https://github.com/Chen-Yang-Liu/RSICC)]. Extract images pairs and put them in `./data/LEVIR_CC/` as follows:
```text
./data/LEVIR_CC/
  ├─LevirCCcaptions_v1.json
  └─images/
     ├─train/
     │  ├─A
     │  └─B
     ├─val/
     │  ├─A
     │  └─B
     └─test/
        ├─A
        └─B

```
Then preprocess the dataset:
```bash
python create_input_files.py
```

After that, cache files (e.g., .pkl) will be generated in ./data/LEVIR_CC/.

## Quick Start
```bash
python create_input_files.py
python train.py
python eval12.py
```

## Citation
For double-blind review, please cite as:
```bibtex
@article{anonymous2026rsicc,
  title={Cognitive Prior-Guided Evidence Decoupling for Remote Sensing Image Change Captioning},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
