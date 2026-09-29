---
dataset_info:
- config_name: bm25
  features:
  - name: query-id
    dtype: string
  - name: corpus-ids
    list: string
  splits:
  - name: NanoClimateFEVER
    num_bytes: 3922584
    num_examples: 50
  - name: NanoDBPedia
    num_bytes: 10592868
    num_examples: 50
  - name: NanoFEVER
    num_bytes: 5923377
    num_examples: 50
  - name: NanoFiQA2018
    num_bytes: 2254597
    num_examples: 50
  - name: NanoHotpotQA
    num_bytes: 2937600
    num_examples: 50
  - name: NanoMSMARCO
    num_bytes: 2745017
    num_examples: 50
  - name: NanoNFCorpus
    num_bytes: 1765636
    num_examples: 50
  - name: NanoNQ
    num_bytes: 3387630
    num_examples: 50
  - name: NanoQuoraRetrieval
    num_bytes: 2463039
    num_examples: 50
  - name: NanoSCIDOCS
    num_bytes: 4864400
    num_examples: 50
  - name: NanoArguAna
    num_bytes: 7017348
    num_examples: 50
  - name: NanoSciFact
    num_bytes: 1696812
    num_examples: 50
  - name: NanoTouche2020
    num_bytes: 12088242
    num_examples: 49
  download_size: 5192075
  dataset_size: 61659150
- config_name: corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: NanoClimateFEVER
    num_bytes: 5617105
    num_examples: 3408
  - name: NanoDBPedia
    num_bytes: 2280448
    num_examples: 6045
  - name: NanoFEVER
    num_bytes: 6285894
    num_examples: 4996
  - name: NanoFiQA2018
    num_bytes: 4201493
    num_examples: 4598
  - name: NanoHotpotQA
    num_bytes: 1868234
    num_examples: 5090
  - name: NanoMSMARCO
    num_bytes: 1745074
    num_examples: 5043
  - name: NanoNFCorpus
    num_bytes: 4521394
    num_examples: 2953
  - name: NanoNQ
    num_bytes: 2740852
    num_examples: 5035
  - name: NanoQuoraRetrieval
    num_bytes: 346228
    num_examples: 5046
  - name: NanoSCIDOCS
    num_bytes: 2149671
    num_examples: 2210
  - name: NanoArguAna
    num_bytes: 3854860
    num_examples: 3635
  - name: NanoSciFact
    num_bytes: 4227131
    num_examples: 2919
  - name: NanoTouche2020
    num_bytes: 12592148
    num_examples: 5745
  download_size: 31113108
  dataset_size: 52430532
- config_name: qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  splits:
  - name: NanoClimateFEVER
    num_bytes: 4361
    num_examples: 148
  - name: NanoDBPedia
    num_bytes: 60640
    num_examples: 1158
  - name: NanoFEVER
    num_bytes: 1630
    num_examples: 57
  - name: NanoFiQA2018
    num_bytes: 2200
    num_examples: 123
  - name: NanoHotpotQA
    num_bytes: 3885
    num_examples: 100
  - name: NanoMSMARCO
    num_bytes: 1065
    num_examples: 50
  - name: NanoNFCorpus
    num_bytes: 64851
    num_examples: 2518
  - name: NanoNQ
    num_bytes: 1340
    num_examples: 57
  - name: NanoQuoraRetrieval
    num_bytes: 1359
    num_examples: 70
  - name: NanoSCIDOCS
    num_bytes: 21472
    num_examples: 244
  - name: NanoArguAna
    num_bytes: 3496
    num_examples: 50
  - name: NanoSciFact
    num_bytes: 1054
    num_examples: 56
  - name: NanoTouche2020
    num_bytes: 45452
    num_examples: 932
  download_size: 91853
  dataset_size: 212805
- config_name: queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: NanoClimateFEVER
    num_bytes: 7044
    num_examples: 50
  - name: NanoDBPedia
    num_bytes: 2723
    num_examples: 50
  - name: NanoFEVER
    num_bytes: 2948
    num_examples: 50
  - name: NanoFiQA2018
    num_bytes: 3531
    num_examples: 50
  - name: NanoHotpotQA
    num_bytes: 6019
    num_examples: 50
  - name: NanoMSMARCO
    num_bytes: 2328
    num_examples: 50
  - name: NanoNFCorpus
    num_bytes: 1939
    num_examples: 50
  - name: NanoNQ
    num_bytes: 3132
    num_examples: 50
  - name: NanoQuoraRetrieval
    num_bytes: 3087
    num_examples: 50
  - name: NanoSCIDOCS
    num_bytes: 6041
    num_examples: 50
  - name: NanoArguAna
    num_bytes: 62331
    num_examples: 50
  - name: NanoSciFact
    num_bytes: 5352
    num_examples: 50
  - name: NanoTouche2020
    num_bytes: 2609
    num_examples: 49
  download_size: 97820
  dataset_size: 109084
configs:
- config_name: bm25
  data_files:
  - split: NanoClimateFEVER
    path: bm25/NanoClimateFEVER-*
  - split: NanoDBPedia
    path: bm25/NanoDBPedia-*
  - split: NanoFEVER
    path: bm25/NanoFEVER-*
  - split: NanoFiQA2018
    path: bm25/NanoFiQA2018-*
  - split: NanoHotpotQA
    path: bm25/NanoHotpotQA-*
  - split: NanoMSMARCO
    path: bm25/NanoMSMARCO-*
  - split: NanoNFCorpus
    path: bm25/NanoNFCorpus-*
  - split: NanoNQ
    path: bm25/NanoNQ-*
  - split: NanoQuoraRetrieval
    path: bm25/NanoQuoraRetrieval-*
  - split: NanoSCIDOCS
    path: bm25/NanoSCIDOCS-*
  - split: NanoArguAna
    path: bm25/NanoArguAna-*
  - split: NanoSciFact
    path: bm25/NanoSciFact-*
  - split: NanoTouche2020
    path: bm25/NanoTouche2020-*
- config_name: corpus
  data_files:
  - split: NanoClimateFEVER
    path: corpus/NanoClimateFEVER-*
  - split: NanoDBPedia
    path: corpus/NanoDBPedia-*
  - split: NanoFEVER
    path: corpus/NanoFEVER-*
  - split: NanoFiQA2018
    path: corpus/NanoFiQA2018-*
  - split: NanoHotpotQA
    path: corpus/NanoHotpotQA-*
  - split: NanoMSMARCO
    path: corpus/NanoMSMARCO-*
  - split: NanoNFCorpus
    path: corpus/NanoNFCorpus-*
  - split: NanoNQ
    path: corpus/NanoNQ-*
  - split: NanoQuoraRetrieval
    path: corpus/NanoQuoraRetrieval-*
  - split: NanoSCIDOCS
    path: corpus/NanoSCIDOCS-*
  - split: NanoArguAna
    path: corpus/NanoArguAna-*
  - split: NanoSciFact
    path: corpus/NanoSciFact-*
  - split: NanoTouche2020
    path: corpus/NanoTouche2020-*
- config_name: qrels
  data_files:
  - split: NanoClimateFEVER
    path: qrels/NanoClimateFEVER-*
  - split: NanoDBPedia
    path: qrels/NanoDBPedia-*
  - split: NanoFEVER
    path: qrels/NanoFEVER-*
  - split: NanoFiQA2018
    path: qrels/NanoFiQA2018-*
  - split: NanoHotpotQA
    path: qrels/NanoHotpotQA-*
  - split: NanoMSMARCO
    path: qrels/NanoMSMARCO-*
  - split: NanoNFCorpus
    path: qrels/NanoNFCorpus-*
  - split: NanoNQ
    path: qrels/NanoNQ-*
  - split: NanoQuoraRetrieval
    path: qrels/NanoQuoraRetrieval-*
  - split: NanoSCIDOCS
    path: qrels/NanoSCIDOCS-*
  - split: NanoArguAna
    path: qrels/NanoArguAna-*
  - split: NanoSciFact
    path: qrels/NanoSciFact-*
  - split: NanoTouche2020
    path: qrels/NanoTouche2020-*
- config_name: queries
  data_files:
  - split: NanoClimateFEVER
    path: queries/NanoClimateFEVER-*
  - split: NanoDBPedia
    path: queries/NanoDBPedia-*
  - split: NanoFEVER
    path: queries/NanoFEVER-*
  - split: NanoFiQA2018
    path: queries/NanoFiQA2018-*
  - split: NanoHotpotQA
    path: queries/NanoHotpotQA-*
  - split: NanoMSMARCO
    path: queries/NanoMSMARCO-*
  - split: NanoNFCorpus
    path: queries/NanoNFCorpus-*
  - split: NanoNQ
    path: queries/NanoNQ-*
  - split: NanoQuoraRetrieval
    path: queries/NanoQuoraRetrieval-*
  - split: NanoSCIDOCS
    path: queries/NanoSCIDOCS-*
  - split: NanoArguAna
    path: queries/NanoArguAna-*
  - split: NanoSciFact
    path: queries/NanoSciFact-*
  - split: NanoTouche2020
    path: queries/NanoTouche2020-*
  default: true
---
