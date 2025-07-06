---

# CausalCounterfactualVAE: Disentangling Gene Expression for Counterfactual Prediction

## Overview

This repository presents `CausalCounterfactualVAE`, a PyTorch-based Conditional Variational Autoencoder (CVAE) designed for disentangled representation learning and counterfactual gene expression prediction using the Genotype-Tissue Expression (GTEx) dataset. The model effectively disentangles individual-specific biological signatures from contextual factors like tissue type, sex, and age, enabling the prediction of hypothetical gene expression profiles under different conditions.

## Key Features

*   **GTEx Data Processing**: Robust pipeline for loading, merging, and filtering GTEx RNA-seq gene expression (TPM) and rich metadata (sample attributes, subject phenotypes).
*   **Contextual VAE Architecture**: A deep generative model that learns a latent representation of gene expression conditioned on multiple biological and demographic factors (tissue type, sex, age, individual ID).
*   **Counterfactual Gene Expression Prediction**: Ability to generate "what-if" gene expression profiles by preserving an individual's unique biological signature while changing contextual factors (e.g., predicting lung expression if a liver sample were from lung tissue).
*   **Disentanglement Verification**: Explicit control experiments demonstrating that the model successfully disentangles individual identity from contextual factors in the latent space, ensuring predictions are not generic but retain source-specific characteristics.
*   **Comprehensive Evaluation**: Quantitative assessment of reconstruction performance (Pearson Correlation, R-squared) and qualitative visualization of the learned latent space (t-SNE).

## Dataset

The project utilizes the **Genotype-Tissue Expression (GTEx) Project (V8)** dataset, specifically:
*   Gene Expression (TPM) data (`GTEX_Analysis_2017-06-05_V8_RNASeqCvl.1.9_gene_tpm.gct.gz`)
*   Sample Attributes (`GTEX_Analysis_2017-06-05_V8_Annotations_SampleAttributesDS.txt`)
*   Subject Phenotypes (`GTEX_Analysis_2017-06-05_V8_Annotations_SubjectPhenotypesDS.txt`)

Data is processed to include highly variable genes (2000 genes) and filtered for relevant tissue types (e.g., Lung, Heart, Liver, Brain Cortex, etc.), sex, and age.

## Model Architecture

The `CausalContextualVAE` is composed of:

*   **Conditional Encoder**: Maps gene expression input to the mean and log-variance of the latent space (`z`). It takes gene expression, tissue ID, sex ID, age (scaled), and individual ID as inputs. Contextual factors are embedded and fused before being concatenated with gene expression for the main encoder MLP.
*   **Conditional Decoder**: Reconstructs gene expression from the latent space (`z`) and the contextual vector.
*   **Reparameterization Trick**: Used to sample from the latent distribution during training.
*   **`predict_counterfactual` method**: Generates counterfactual expression by encoding a source sample to obtain its latent representation (`z`) and then decoding `z` with a *new* target context.

**Model Dimensions:**
*   **Genes**: 2000
*   **Tissue Types**: 16
*   **Sexes**: 2
*   **Individuals**: 940
*   **Latent Dimension**: 256
*   **Context Fusion Dimension**: 256
*   **Tissue Embedding Dimension**: 64
*   **Sex Embedding Dimension**: 16
*   **Individual Embedding Dimension**: 128
*   **Age Dimension**: 1

**Total Trainable Parameters**: 6,296,912

## Training and Evaluation

The model is trained using a composite VAE loss function (reconstruction loss + KL divergence loss) with the AdamW optimizer and a `ReduceLROnPlateau` scheduler. Early stopping is implemented based on validation loss improvement.

**Key Evaluation Metrics (Example):**
*   **Average Pearson Correlation Coefficient (per gene)**: 0.9515
*   **Average R-squared (per gene)**: 0.9030

## Counterfactual Prediction Example

The model can predict how a sample's gene expression would look if it were from a different tissue. For instance, transforming an `Original Lung Expression` profile to `Counterfactual Liver Expression` from the same individual. Visualizations demonstrate the shift in expression distribution while attempting to retain individual-specific characteristics.

## Disentanglement Verification

A critical experiment is performed to verify the influence of the source sample's unique identity on counterfactual predictions. By generating counterfactual liver expressions from *two different lung samples* (from different individuals) and comparing the resulting liver profiles, the model demonstrates that:

*   The two counterfactual liver profiles are distinct from each other.
*   The inherent differences between the two source individuals are preserved even after the counterfactual transformation.

This strong indication suggests that the model is **NOT** just generating a generic liver template, but correctly blending the unique biological signature (`z`) of each source lung sample with the 'liver' context.

## Setup and Usage (Google Colab Environment)

This project was developed in a Google Colab environment, leveraging Google Drive for data storage.

1.  **Mount Google Drive**: The initial setup involves mounting Google Drive to access raw data and save processed files/models.
2.  **Define Project Paths**: Configure `PROJECT_ROOT`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, and `ENCODERS_DIR`.
3.  **Download GTEx Data**: Ensure the necessary GTEx V8 files are downloaded and placed in the specified `RAW_DATA_DIR`. A link for `GTEX_Analysis_2017-06-05_V8_Annotations_SubjectPhenotypesDS.txt` is provided in the notebook.
4.  **Run Notebook Cells**: Execute the Jupyter notebook cells sequentially (Cells 1-8).
    *   **Cell 1**: Environment Setup and Data Path Configuration.
    *   **Cell 2**: Optimized Loading, Merging, and Aggressive Subsetting of GTEx metadata and gene expression. Conversion of sex to numerical ID using `LabelEncoder`.
    *   **Cell 3**: Custom PyTorch Dataset and DataLoader Setup for training.
    *   **Cell 4**: Deep Learning Model Architecture Definition (`CausalContextualVAE`).
    *   **Cell 5**: Training Loop Implementation (Reconstruction + KL Divergence Loss).
    *   **Cell 6**: Model Evaluation and Counterfactual Generation. Includes reconstruction performance metrics and visualization of latent space and counterfactual predictions.
    *   **Cell 7**: Direct Comparison of Counterfactual vs. Actual Gene Expression (if a matching actual liver sample exists for the source individual).
    *   **Cell 8**: Verifying the Influence of the Source Sample (disentanglement control experiment).

## Dependencies

*   `torch`
*   `numpy`
*   `pandas`
*   `scikit-learn` (LabelEncoder, StandardScaler, PCA, TSNE, r2_score, pearsonr)
*   `matplotlib`
*   `seaborn`
*   `tqdm`
*   `pickle`
*   `gzip`

## References and Inspiration

This project is built upon the foundational work in Variational Autoencoders and Conditional VAEs for biological data. Inspirations and similar approaches include:

*   [lovelyscientist/rna-seq-vae](https://github.com/lovelyscientist/rna-seq-vae) - A similar project using CVAE on GTEx data, conditioned on tissue and age.
*   Related research on disentangled representations and causal inference in generative models for omics data.

---
