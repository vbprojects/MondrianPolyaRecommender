---
license: apache-2.0
language:
- en
base_model:
- answerdotai/ModernBERT-base
base_model_relation: finetune
pipeline_tag: sentence-similarity
library_name: transformers
tags:
- sentence-transformers
- mteb
- embedding
- transformers.js
- text-embeddings-inference
---

# gte-modernbert-base

We are excited to introduce the `gte-modernbert` series of models, which are built upon the latest modernBERT pre-trained encoder-only foundation models. The `gte-modernbert` series models include both text embedding models and rerank models.

The `gte-modernbert` models demonstrates competitive performance in several text embedding and text retrieval evaluation tasks when compared to similar-scale models from the current open-source community. This includes assessments such as MTEB, LoCO, and COIR evaluation.

## Model Overview

- Developed by: Tongyi Lab, Alibaba Group
- Model Type: Text Embedding
- Primary Language: English
- Model Size: 149M
- Max Input Length: 8192 tokens
- Output Dimension: 768

### Model list


|                                         Models                                         | Language |       Model Type       | Model Size | Max Seq. Length | Dimension | MTEB-en | BEIR | LoCo | CoIR |
|:--------------------------------------------------------------------------------------:|:--------:|:----------------------:|:----------:|:---------------:|:---------:|:-------:|:----:|:----:|:----:|
|  [`gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)   | English  |     text embedding     |    149M    |      8192       |    768    |  64.38  | 55.33 | 87.57 | 79.31 | 
| [`gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)  | English  | text reranker     |    149M    |    8192    |     -     |  - | 56.19 | 90.68 | 79.99 |

## Usage

> [!TIP]
> For `transformers` and `sentence-transformers`, if your GPU supports it, the efficient Flash Attention 2 will be used automatically if you have `flash_attn` installed. It is not mandatory.
> 
> ```bash
> pip install flash_attn
> ```

Use with `transformers`

```python
# Requires transformers>=4.48.0

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

model_path = "Alibaba-NLP/gte-modernbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state[:, 0]
 
# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores.tolist())
# [[42.89073944091797, 71.30911254882812, 33.664554595947266]]
```

Use with `sentence-transformers`:

```python
# Requires transformers>=4.48.0
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
embeddings = model.encode(input_texts)
print(embeddings.shape)
# (4, 768)

similarities = cos_sim(embeddings[0], embeddings[1:])
print(similarities)
# tensor([[0.4289, 0.7131, 0.3366]])
```

Use with `transformers.js`:

```js
// npm i @huggingface/transformers
import { pipeline, matmul } from "@huggingface/transformers";

// Create a feature extraction pipeline
const extractor = await pipeline(
  "feature-extraction",
  "Alibaba-NLP/gte-modernbert-base",
  { dtype: "fp32" }, // Supported options: "fp32", "fp16", "q8", "q4", "q4f16"
);

// Embed queries and documents
const embeddings = await extractor(
  [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms",
  ],
  { pooling: "cls", normalize: true },
);

// Compute similarity scores
const similarities = (await matmul(embeddings.slice([0, 1]), embeddings.slice([1, null]).transpose(1, 0))).mul(100);
console.log(similarities.tolist()); // [[42.89077377319336, 71.30916595458984, 33.66455841064453]]
```

Additionally, you can also deploy `Alibaba-NLP/gte-modernbert-base` with [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) as follows:

- CPU

```bash
docker run --platform linux/amd64 \
  -p 8080:80 \
  -v $PWD/data:/data \
  --pull always \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 \
  --model-id Alibaba-NLP/gte-modernbert-base
```

- GPU

```bash
docker run --gpus all \
  -p 8080:80 \
  -v $PWD/data:/data \
  --pull always \
  ghcr.io/huggingface/text-embeddings-inference:1.7 \
  --model-id Alibaba-NLP/gte-modernbert-base
```

Then you can send requests to the deployed API via the OpenAI-compatible `v1/embeddings` route (more information about the [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)):

```bash
curl https://0.0.0.0:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "what is the capital of China?",
      "how to implement quick sort in python?",
      "Beijing",
      "sorting algorithms"
    ],
    "model": "Alibaba-NLP/gte-modernbert-base",
    "encoding_format": "float"
  }'
```

## Training Details

The `gte-modernbert` series of models follows the training scheme of the previous [GTE models](https://huggingface.co/collections/Alibaba-NLP/gte-models-6680f0b13f885cb431e6d469), with the only difference being that the pre-training language model base has been replaced from [GTE-MLM](https://huggingface.co/Alibaba-NLP/gte-en-mlm-base) to [ModernBert](https://huggingface.co/answerdotai/ModernBERT-base). For more training details, please refer to our paper: [mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval](https://aclanthology.org/2024.emnlp-industry.103/)

## Evaluation

### MTEB

The results of other models are retrieved from [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Given that all models in the `gte-modernbert` series have a size of less than 1B parameters, we focused exclusively on the results of models under 1B from the MTEB leaderboard.

|                                            Model Name                                            | Param Size (M) | Dimension | Sequence Length | Average (56) | Class. (12) | Clust. (11) | Pair Class. (3) | Reran. (4) | Retr. (15) |  STS (10)   | Summ. (1) |
|:------------------------------------------------------------------------------------------------:|:--------------:|:---------:|:---------------:|:------------:|:-----------:|:---:|:---:|:---:|:---:|:-----------:|:--------:|
|        [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)         |      335       |   1024    |       512       |    64.68     |    75.64    | 46.71 | 87.2 | 60.11 | 54.39 |     85      |   32.71  |
| [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) |      560       |   1024    |       514       |    64.41     |    77.56    | 47.1 | 86.19 | 58.58 | 52.47 |    84.78    |   30.39  |
|                [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)                |      335       |   1024    |       512       |    64.23     |    75.97    | 46.08 | 87.12 | 60.03 | 54.29 |    83.11    |   31.61  |
|             [gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)              |      137       |    768    |      8192       |  64.11   |    77.17    | 46.82 | 85.33 | 57.66 | 54.09 |    81.97    |   31.17  |
|                 [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)                 |      109       |    768    |       512       |    63.55     |    75.53    | 45.77 | 86.55 | 58.86 | 53.25 |    82.4     |   31.07  |
|            [gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)             |      409       |   1024    |      8192       |    65.39     |    77.75    | 47.95 | 84.63 | 58.50 | 57.91 |    81.43    |   30.91  |
| [modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base) |      149       |    768    |      8192       |    62.62     |    74.31    | 44.98 | 83.96 | 56.42 | 52.89 |    81.78    |   31.39  |
| [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |                |    768    |      8192       |    62.28     |   	73.55    |	43.93 |	84.61 |	55.78 | 53.01|    81.94    |   30.4   |
| [gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) |      305       |    768    |       8192      |     61.4     | 70.89 | 44.31 | 84.24 | 57.47 |51.08 |    82.11    |   30.58  | 
| [jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) | 572 |   1024    |      8192  |       65.51 | 82.58 |45.21 |84.01 |58.13 |53.88 | 85.81 |   29.71  | 
| [**gte-modernbert-base**](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 149 |   768    |      8192  |   **64.38** | **76.99** | **46.47** | **85.93** | **59.24** | **55.33** | **81.57** | **30.68** |


### LoCo (Long Document Retrieval)(NDCG@10)

| Model Name |  Dimension | Sequence Length | Average (5) | QsmsumRetrieval | SummScreenRetrieval | QasperAbastractRetrieval | QasperTitleRetrieval |  GovReportRetrieval |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [gte-qwen1.5-7b](https://huggingface.co/Alibaba-NLP/gte-qwen1.5-7b) | 4096 | 32768 |  87.57 | 49.37 | 93.10 | 99.67 | 97.54 | 98.21 | 
| [gte-large-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-v1.5) |1024 | 8192 | 86.71 | 44.55 | 92.61 | 99.82 | 97.81 | 98.74 |
| [gte-base-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-v1.5) | 768 | 8192 | 87.44 | 49.91  | 91.78 | 99.82 | 97.13 | 98.58 |
| [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 768 | 8192 | 88.88 | 54.45 | 93.00 | 99.82 | 98.03 | 98.70 |
| [gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | - | 8192 | 90.68 | 70.86 | 94.06 | 99.73 | 99.11 | 89.67 | 

### COIR (Code Retrieval Task)(NDCG@10)

| Model Name | Dimension | Sequence Length | Average(20) | CodeSearchNet-ccr-go | CodeSearchNet-ccr-java | CodeSearchNet-ccr-javascript | CodeSearchNet-ccr-php | CodeSearchNet-ccr-python | CodeSearchNet-ccr-ruby | CodeSearchNet-go | CodeSearchNet-java | CodeSearchNet-javascript | CodeSearchNet-php | CodeSearchNet-python | CodeSearchNet-ruby | apps | codefeedback-mt | codefeedback-st | codetrans-contest | codetrans-dl | cosqa | stackoverflow-qa | synthetic-text2sql |
|:----:|:---:|:---:|:---:|:---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 768 | 8192 | 79.31	| 94.15	| 93.57 |	94.27 |	91.51	| 93.93	| 90.63	| 88.32 |	83.27	| 76.05	| 85.12	| 88.16	| 77.59	| 57.54	| 82.34	| 85.95	| 71.89	 | 35.46	| 43.47	| 91.2	| 61.87 |
| [gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | - | 8192 | 79.99	| 96.43	| 96.88	| 98.32 | 91.81	| 97.7	| 91.96 |	88.81	| 79.71	| 76.27	| 89.39	| 98.37	| 84.11	| 47.57	| 83.37	| 88.91	| 49.66	| 36.36	| 44.37	| 89.58	| 64.21 |

### BEIR(NDCG@10)

| Model Name | Dimension | Sequence Length | Average(15) | ArguAna | ClimateFEVER | CQADupstackAndroidRetrieval | DBPedia | FEVER | FiQA2018 | HotpotQA | MSMARCO | NFCorpus | NQ | QuoraRetrieval | SCIDOCS | SciFact | Touche2020 | TRECCOVID |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 768 | 8192 | 55.33 | 72.68 | 37.74 | 42.63 | 41.79 | 91.03 | 48.81 | 69.47 | 40.9 | 36.44 | 57.62 | 88.55 | 21.29 | 77.4 | 21.68 | 81.95 |
| [gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | - | 8192 | 56.73 | 69.03 | 37.79 | 44.68 | 47.23 | 94.54 | 49.81 | 78.16 | 45.38 | 30.69 | 64.57 | 87.77 | 20.60 | 73.57 | 27.36 | 79.89 |



## Hiring

We have open positions for **Research Interns** and **Full-Time Researchers** to join our team at Tongyi Lab. 
We are seeking passionate individuals with expertise in representation learning, LLM-driven information retrieval, Retrieval-Augmented Generation (RAG), and agent-based systems. 
Our team is located in the vibrant cities of **Beijing** and **Hangzhou**.
If you are driven by curiosity and eager to make a meaningful impact through your work, we would love to hear from you. Please submit your resume along with a brief introduction to <a href="mailto:dingkun.ldk@alibaba-inc.com">dingkun.ldk@alibaba-inc.com</a>.


## Citation

If you find our paper or models helpful, feel free to give us a cite.

```
@inproceedings{zhang2024mgte,
  title={mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval},
  author={Zhang, Xin and Zhang, Yanzhao and Long, Dingkun and Xie, Wen and Dai, Ziqi and Tang, Jialong and Lin, Huan and Yang, Baosong and Xie, Pengjun and Huang, Fei and others},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  pages={1393--1412},
  year={2024}
}

@article{li2023towards,
  title={Towards general text embeddings with multi-stage contrastive learning},
  author={Li, Zehan and Zhang, Xin and Zhang, Yanzhao and Long, Dingkun and Xie, Pengjun and Zhang, Meishan},
  journal={arXiv preprint arXiv:2308.03281},
  year={2023}
}
```