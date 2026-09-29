---
language:
- en
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
task_categories:
- feature-extraction
- sentence-similarity
pretty_name: SQuAD
tags:
- sentence-transformers
dataset_info:
  config_name: pair
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 72106438
    num_examples: 87599
  download_size: 11823133
  dataset_size: 72106438
configs:
- config_name: pair
  data_files:
  - split: train
    path: pair/train-*
---

# Dataset Card for SQuAD

This dataset is a collection of question-answer pairs from the SQuAD dataset. See [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) for additional information.
This dataset can be used directly with Sentence Transformers to train embedding models.

## Dataset Subsets

### `pair` subset

* Columns: "question", "answer"
* Column types: `str`, `str`
* Examples:
    ```python
    {
      'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
      'answer': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
    }
    ```
* Collection strategy: Reading the SQuAD dataset from [embedding-training-data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data).
* Deduplified: No