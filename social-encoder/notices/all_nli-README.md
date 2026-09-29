---
language:
- en
multilinguality:
- monolingual
size_categories:
- 1M<n<10M
task_categories:
- feature-extraction
- sentence-similarity
pretty_name: AllNLI
tags:
- sentence-transformers
dataset_info:
- config_name: pair
  features:
  - name: anchor
    dtype: string
  - name: positive
    dtype: string
  splits:
  - name: train
    num_bytes: 43012118
    num_examples: 314315
  - name: dev
    num_bytes: 992955
    num_examples: 6808
  - name: test
    num_bytes: 1042254
    num_examples: 6831
  download_size: 27501136
  dataset_size: 45047327
- config_name: pair-class
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 138755142
    num_examples: 942069
  - name: dev
    num_bytes: 3034127
    num_examples: 19657
  - name: test
    num_bytes: 3142127
    num_examples: 19656
  download_size: 72651651
  dataset_size: 144931396
- config_name: pair-score
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: score
    dtype: float64
  splits:
  - name: train
    num_bytes: 138755142
    num_examples: 942069
  - name: dev
    num_bytes: 3034127
    num_examples: 19657
  - name: test
    num_bytes: 3142127
    num_examples: 19656
  download_size: 72653539
  dataset_size: 144931396
- config_name: triplet
  features:
  - name: anchor
    dtype: string
  - name: positive
    dtype: string
  - name: negative
    dtype: string
  splits:
  - name: train
    num_bytes: 98815977
    num_examples: 557850
  - name: dev
    num_bytes: 1272591
    num_examples: 6584
  - name: test
    num_bytes: 1341266
    num_examples: 6609
  download_size: 39988980
  dataset_size: 101429834
configs:
- config_name: pair
  data_files:
  - split: train
    path: pair/train-*
  - split: dev
    path: pair/dev-*
  - split: test
    path: pair/test-*
- config_name: pair-class
  data_files:
  - split: train
    path: pair-class/train-*
  - split: dev
    path: pair-class/dev-*
  - split: test
    path: pair-class/test-*
- config_name: pair-score
  data_files:
  - split: train
    path: pair-score/train-*
  - split: dev
    path: pair-score/dev-*
  - split: test
    path: pair-score/test-*
- config_name: triplet
  data_files:
  - split: train
    path: triplet/train-*
  - split: dev
    path: triplet/dev-*
  - split: test
    path: triplet/test-*
---

# Dataset Card for AllNLI

This dataset is a concatenation of the [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) datasets.
Despite originally being intended for Natural Language Inference (NLI), this dataset can be used for training/finetuning an embedding model for semantic textual similarity.

## Dataset Subsets

### `pair-class` subset

* Columns: "premise", "hypothesis", "label"
* Column types: `str`, `str`, `class` with `{"0": "entailment", "1": "neutral", "2", "contradiction"}`
* Examples:
    ```python
    {
      'premise': 'A person on a horse jumps over a broken down airplane.',
      'hypothesis': 'A person is training his horse for a competition.',
      'label': 1,
    }
    ```
* Collection strategy: Reading the premise, hypothesis and integer label from SNLI & MultiNLI datasets.
* Deduplified: Yes

### `pair-score` subset

* Columns: "sentence1", "sentence2", "score"
* Column types: `str`, `str`, `float`
* Examples:
    ```python
    {
      'sentence1': 'A person on a horse jumps over a broken down airplane.',
      'sentence2': 'A person is training his horse for a competition.',
      'score': 0.5,
    }
    ```
* Collection strategy: Taking the `pair-class` subset and remapping "entailment", "neutral" and "contradiction" to 1.0, 0.5 and 0.0, respectively.
* Deduplified: Yes

### `pair` subset

* Columns: "anchor", "positive"
* Column types: `str`, `str`
* Examples:
    ```python
    {
      'anchor': 'A person on a horse jumps over a broken down airplane.',
      'positive': 'A person is training his horse for a competition.',
    }
    ```
* Collection strategy: Reading the SNLI & MultiNLI datasets and considering the "premise" as the "anchor" and the "hypothesis" as the "positive" if the label is "entailment". The reverse ("entailment" as "anchor" and "premise" as "positive") is not included.
* Deduplified: Yes

### `triplet` subset

* Columns: "anchor", "positive", "negative"
* Column types: `str`, `str`, `str`
* Examples:
    ```python
    {
      'anchor': 'A person on a horse jumps over a broken down airplane.',
      'positive': 'A person is outdoors, on a horse.',
      'negative': 'A person is at a diner, ordering an omelette.',
    }
    ```
* Collection strategy: Reading the SNLI & MultiNLI datasets, for each "premise" making a list of entailing and contradictory sentences using the dataset labels. Then, considering all possible triplets out of these entailing and contradictory lists. The reverse ("entailment" as "anchor" and "premise" as "positive") is not included.
* Deduplified: Yes