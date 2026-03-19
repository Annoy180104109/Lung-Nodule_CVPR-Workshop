# Lung-Nodule_CVPR-Workshop
Title : Towards A Few-Shot Segmentation and Classification Framework for Lung Nodule Analysis

Abstract : n this study, few-shot joint segmentation and classifi-001
cation of lung nodules in chest Computed Tomography002
(CT) was investigated under limited-annotation settings.003
A framework was developed to support both fully super-004
vised few-shot learning with pixel-level annotations and a005
mixed-supervision setting combining pixel-level labels with006
weaker image-level supervision. Using a self-supervised007
Self-DIstillation with NO labels (DINO) - Vision Trans-008
former (ViT) backbone (DINO-ViT), image, class, and query009
tokens were extracted, correlated through cosine similar-010
ity, and refined by a Vision Correlation Transformer to en-011
hance support and query representations for the two tasks.012
In the mixed-supervision setting, pseudo-masks were gen-013
erated using a lightweight self-supervised ViT model and014
incorporated as weak supervisory signals during train-015
ing. The framework was evaluated on the Lung Image016
Database Consortium and Image Database Resource Ini-017
tiative dataset (LIDC-IDRI) using patient-wise splits un-018
der multiple way/shot configurations and backbone settings.019
The strongest results were obtained for segmentation, with a020
maximum mean Intersection over Union (IoU) of 69.36% in021
the 1-way 5-shot setting, while the experiments also high-022
lighted the greater difficulty of low-shot nodule classifica-023
tion and the sensitivity of pseudo-label-based supervision.024
The findings establish a medically grounded benchmark for025
low-annotation joint lung nodule analysis and align with026
current interests in robust, data-efficient, and clinically rel-027
evant AI for medical imaging
