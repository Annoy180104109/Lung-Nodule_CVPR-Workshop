# Lung-Nodule_CVPR-Workshop
Title : Towards A Few-Shot Segmentation and Classification Framework for Lung Nodule Analysis

Abstract : In this study, few-shot joint segmentation and classification of lung nodules in chest Computed Tomography002
(CT) was investigated under limited-annotation settings.A framework was developed to support both fully supervised few-shot learning with pixel-level annotations and a mixed-supervision setting combining pixel-level labels with weaker image-level supervision. Using a self-supervised007
Self-DIstillation with NO labels (DINO) - Vision Transformer (ViT) backbone (DINO-ViT), image, class, and query009
tokens were extracted, correlated through cosine similarity, and refined by a Vision Correlation Transformer to en-hance support and query representations for the two tasks. In the mixed-supervision setting, pseudo-masks were generated using a lightweight self-supervised ViT model and
incorporated as weak supervisory signals during training. The framework was evaluated on the Lung Image Database Consortium and Image Database Resource Initiative dataset (LIDC-IDRI) using patient-wise splits under multiple way/shot configurations and backbone settings.
The strongest results were obtained for segmentation, with a maximum mean Intersection over Union (IoU) of 69.36% in
the 1-way 5-shot setting, while the experiments also highlighted the greater difficulty of low-shot nodule classifica-tion and the sensitivity of pseudo-label-based supervision. The findings establish a medically grounded benchmark for low-annotation joint lung nodule analysis and align with
current interests in robust, data-efficient, and clinically relevant AI for medical imaging
