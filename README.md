# NDGCNet

NDGCNet is a unified deep learning framework for predicting ncRNA–drug resistance associations across multiple ncRNA types.

## Introduction

Predicting ncRNA–drug resistance associations is important for understanding drug response mechanisms at the molecular level and promoting precision cancer treatment. Although many computational methods have been developed, most existing approaches are still fragmented. In particular, methods that use informative RNA features often build separate models on independent datasets for different ncRNA types, which limits their ability to jointly exploit shared drug resistance patterns.

To address this limitation, NDGCNet integrates multiple ncRNA types into a single learning framework for cross-type ncRNA–drug resistance association prediction.

NDGCNet contains three main components:

1. **CNN-based multi-modal feature extractor**

   This module learns feature representations of ncRNAs and drugs from multi-modal input data. Contrastive learning is introduced to enhance ncRNA type-aware discrimination and improve representation quality.

2. **Cross-attention fusion predictor**

   This module models deep interactions between ncRNA and drug representations. By using cross-attention, NDGCNet captures shared resistance patterns across different ncRNA types and improves association prediction.

3. **Graph-constrained space mapping module**

   This module regularizes the learned representations by preserving neighborhood relationships in ncRNA similarity networks and drug similarity networks. It helps the model maintain structural consistency in the learned latent space.

Overall, NDGCNet provides an integrated framework for jointly learning from multiple ncRNA types and predicting ncRNA–drug resistance associations in a unified manner.

---

## Availability



---

## Local running

python main.py

## Environment

The recommended running environment is listed below.

```text
python = 3.10
torch = 2.1.0+cu118
dgl = 2.2.1+cu118
torch-geometric = 2.6.1
```

Basic data handling and utility packages:

```text
pandas >= 1.5.0
numpy >= 1.24.0
scikit-learn >= 1.2.0
tqdm >= 4.65.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
```
