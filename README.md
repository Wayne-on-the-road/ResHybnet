# Introduction
The whole experiment is based on a sample graph dataset from CERT4.2. The sampled user-day nodes and train-test splitting are the same as our previous work (B-CITD) . The code and data here consists of three parts:
## 0. Sequence pre-processing
All the pre-processing work of sequential activities for user-days are included in file directory (*code_for_sequence_process*). To try this procedure out, one needs to download the four log files used from original CERT4.2 dataset, since it is too big for uploading them here. The other alternative for file downloading is from my personal cloud link:
Put downloaded .csv file in cert4.2_data directory and run the two .py files one by one will generate the activity sequence for each user-day sample in our sample dataset, which can be used as input dataset for LSTM feature extraction in the next stage, and will be stored in *sample_data* directory. 

This directory is just for trial and demonstration, and it is not necessary for later steps, since all the sample data needed have already been included in another separated directory. 

## 1. Use LSTM auto-encoder to extract features from sequential activity
To try this part, one only needs to include the *sample_data* directory from main branch, and run the [lstm_feature_extraction.py](https://github.com/Wayne-on-the-road/ResHybnet/blob/main/lstm_feature_extraction.py "lstm_feature_extraction.py").

## 2. Try ResHybnet model
