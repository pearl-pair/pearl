# Entity-Aware Generative Retrieval for Personalized Contexts

This is the official code repository for **PEARL** (<u>**P**</u>ersonalized <u>**E**</u>ntity-<u>**A**</u>ware Generative <u>**R**</u>etrieva<u>**L**</u>), submitted to **CIKM 2025**.

---

## Code Information
All code is written in Python 3.8.20 and PyTorch 1.13.0.
This repository contains the implementation of 
**PEARL** (<u>**P**</u>ersonalized <u>**E**</u>ntity-<u>**A**</u>ware Generative <u>**R**</u>etrieva<u>**L**</u>), 
a novel generative retrieval framework designed to handle ambiguous, context-dependent queries in personalized IR scenarios.
**PEARL** integrates three key techniques: 1) entity-span mass regularization to reduce lexical sensitivity, 2) prefix-based contrastive learning to improve structural alignment, 
and 3) context diversification via transposition to enhance robustness against contextual variation. 
These components are jointly trained under a unified sequence-to-sequence objective using identifiers derived via hierarchical k-means.

## Installation

Install required packages by running:

```
pip install -r requirements.txt
```

> **Note**: This codebase has been tested on a GPU cluster equipped with four RTX 4090 GPUs.  
> We utilize [wandb](https://wandb.ai/site) as a cloud-based logger. You must first register and log in to use it.

---

## Data Preprocessing

Follow the instructions provided in `preprocessing.ipynb` and `ner_ft.ipynb`.

In `preprocessing.ipynb`, you can perform:

- Generation of training and evaluation datasets for fine-tuning a NER (FLAIR) model:  
  - Fine-tune the NER model using the datasets generated in `ner_ft.ipynb`.

- Generation of training and evaluation datasets for fine-tuning the DocT5query model:  
  - Use these datasets when setting the task to `docTquery`.

- Passage tokenization via hierarchical k-means clustering with verbalized contexts.
- Query-passage dataset augmentation through transposition.
- Refinement of query generation datasets.
- Annotation and assignment of IDs for the synthetic dataset.

The preprocessing pipeline uses the training file located at:

```
data/synthetics/syn_50k.json
```

After completing preprocessing, the output dataset will be located at:

```
data/synthetics/syn_50k_train_pearl
```

---

## Model Training

Execute the training script using the following command format:

```
./run.sh {output_log_file} {num_processes} {config_file}
```

Adjust training parameters in the configuration file located at `config/train.yml`. You can modify datasets and model configurations directly through command-line arguments for various experiments. Retrieval Hit scores on the development set will be logged to wandb during training.

### Training the PEARL Model

#### Step 1: Train the Query Generation Model

PEARL requires a DocT5query model to generate potentially relevant queries for each candidate passage:

```
./run.sh dTq_train.log 4 configs/docTquery.yml
```

#### Step 2: Generate Queries for All Passages

Run query generation for all passages in the training dataset:

```
./run.sh dTq_generation.log 4 configs/generation.yml
```

> **Note**: The example above runs with default configurations, which you can modify in `configs/generation.yml`. The parameter `--num_return_sequences` specifies the number of queries generated per passage. By default, this is set to 5.

#### Step 3: Train the PEARL Model

Finally, train the PEARL model using the prepared dataset `syn_50k_train_pearl`:

```
./run.sh pearl_train.log 4 configs/train.yml
```