## Social-Media-Toxic-Comment-Filter

The model can filter user comments based on the degree of harmfulness of the language through these steps:
* _Preprocess the text by eliminating the set of tokens that do not contribute significantly to the semantic level;_
* _Transform the text corpus into sequences;_
* _Build a deep learning model including recurring layers for a multilabel classification task;_
* _In prediction time, the model must return a vector containing a 1 or a 0 in correspondence with each label present in the dataset (toxic, severe_toxic, obscene, threat, insult, identity_hate)._

The used dataset is heavily unbalanced (the vast majority of comments is non-toxic, so different cases are studied to rebalance and boost.

For each case the Deep Learning model hyperparameters do not substancially change (just small adjustments) and is characterized by: 
Embedding, Bidirectional, TimeDistributed, Flatten Dense Droput and Dense layers.

The analyzed cases are the following:

1. Downsampling non-toxic train dataset;
2. Oversampling toxic train dataset;
3. Label sensitive oversampling toxic train dataset;
4. Words Embedding;
5. Oversampling oxic train dataset plus Words Embedding;
6. abel sensitive oversampling toxic train dataset plus Words Embedding.
