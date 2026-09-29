---
dataset_info:
  features:
  - name: query
    dtype: string
  - name: positive
    dtype: string
  - name: negative
    dtype: string
  splits:
  - name: train
    num_bytes: 371861236.32572496
    num_examples: 500000
  download_size: 236711416
  dataset_size: 371861236.32572496
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
