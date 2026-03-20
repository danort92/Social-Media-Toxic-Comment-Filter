# Comment Structure — Social Media Toxic Comment Filter

This document recreates and documents the full comment structure used throughout the `Toxic_Comments_Filter.ipynb` notebook.

---

## Overall Notebook Organization

The notebook follows a repeating **section-based** architecture with **6 experimental cases**, each sharing a common pipeline structure. The hierarchy is:

```
## Project Title (markdown)
   Project Description (markdown, Italian)
   # DATASET IMPORT & EDA (markdown section header)
      [code cells with inline comments]
   # PREPROCESSING (markdown section header)
      [code cells with inline comments]
   # N° CASE: CASE TITLE (markdown section header, repeated x6)
      [data preparation code cells]
      # MODEL CONSTRUCTION (markdown section header)
      [model + training code cells]
      [evaluation code cells]
      # CONCLUSIONS (markdown section header)
      [analysis markdown cell]
```

---

## 1. Markdown Cell Comment Structure

### 1.1 Project Header
```markdown
## Project: Toxic Comment Filter
```

### 1.2 Project Description (Italian)
```markdown
Costruire un modello in grado di filtrare i commenti degli utenti in base al grado di dannosità del linguaggio:
1. Preprocessare il testo eliminando l'insieme di token che non danno contributo significativo a livello semantico
2. Trasformare il corpus testuale in sequenze
3. Costruire un modello di Deep Learning comprendente dei layer ricorrenti per un task di classificazione multilabel
4. In prediction time, il modello deve ritornare un vettore contenente un 1 o uno 0 in corrispondenza di ogni label presente nel dataset (toxic, severe_toxic, obscene, threat, insult, identity_hate).
```

### 1.3 Major Section Headers
Use `#` with italic formatting (`*...*`) or bold (`**...**`):

```markdown
# *DATASET IMPORT & EDA*
# *PREPROCESSING*
# *MODEL CONSTRUCTION*
# *CONCLUSIONS*
```

### 1.4 Case Headers
Use `#` with bold and numbering:

```markdown
# **1° CASE: DOWNSAMPLING CLEAN TRAIN DATA**
# **2° CASE: OVERSAMPLING TOXIC TRAIN DATA**
# **3° CASE: OVERSAMPLING TOXIC TRAIN DATA - LABEL SENSITIVE**
# **4° CASE: WORDS EMBEDDING**
# **5° CASE: OVERSAMPLING + WORDS EMBEDDING**
## **6° CASE: LABEL SENSITIVE OVERSAMPLING + WORDS EMBEDDING**
```

### 1.5 Conclusion Markdown Cells
Free-form analysis text discussing Precision, Recall, Accuracy trade-offs. Example pattern:
```markdown
Oversampling the minority class (toxic comments) the model reaches a higher value of Precision, but a lower value of Recall. Not so good. Same problem on less populated labels is evident too. Let's try to oversample only these specific labels...
```

---

## 2. Code Cell Comment Structure

### 2.1 Import Section — Category Headers
Imports are grouped by category using `#category` comments (no space after `#`, lowercase):

```python
#basics
import pandas as pd
import numpy as np
import pickle

#nlp
import spacy
import string
import nltk
import re

#viz
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns

#ml
from sklearn.model_selection import train_test_split
from keras.models import Sequential
```

### 2.2 Single-Line Descriptive Comments
Short descriptive comments at the top of code cells, prefixed with `#` and a space:

```python
#check for N/A values to drop
df.isna().sum()

#define the comments texts
df_comments=df["comment_text"]

#define the labels
df_toxic_comments_labels=df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

#number of toxic comments per label
df_toxic_comments_labels.sum()

#dataset split
X_train,X_test,Y_train,Y_test=train_test_split(...)

#creation of train dataframe
data_train=pd.DataFrame({...})

#creation of clean comments and toxic comments dataframe
data_train_clean=data_train[data_train["sum_injurious"]==0]

#tokenization of train and test sets
tokenizer=Tokenizer(num_words=10000)

#padding of train and test sets
pad_X_train = pad_sequences(tok_X_train, maxlen=100, padding="post")

#number of non toxic comments: severly imbalanced dataset
non_toxic_comments_counts=len(...)

#examples...
output_word=""
```

### 2.3 Inline Code Comments
Comments explaining specific lines within functions:

```python
sentence=sentence.lower() # remove capital letters
sentence=sentence.replace(c," ") # replace punctuations with a space

sentence = re.sub(r'http[s]?://...', '', sentence) # clean url
sentence = re.sub(r'#(\w+)', '', sentence)   # clean hashes
sentence = re.sub(r'@(\w+)', '', sentence)   # clean @
sentence = re.sub(r'<[^>]+>', '', sentence)  # clean tags
sentence = re.sub(r'\d+', '', sentence)      # clean digits

sentence=" ".join(token.lemma_ for token in document) # lemmatization
sentence = " ".join(word for word in sentence.split() if len(word)>1 and word not in english_stopwords) # clean short words and stopwords
```

### 2.4 Embedding and Model Comments

```python
#Embedding length based on selected model - we are using 50d here.
embedding_vector_length = 300

#Initialize embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_vector_length))

# Load pretrained Glove model (in word2vec form)
word2vec_model=gensim.downloader.load('word2vec-google-news-300')

#Reading word's embedding from Glove model for a given word
embedding_vector = word2vec_model[word]
```

### 2.5 Rationale / Design Decision Comments

```python
#recall is the most important, I want to minimize the flase negatives (toxic comments that are classified as non toxic)
```

### 2.6 Commented-Out Code Blocks
Used for alternative approaches or previously-run cells:

```python
#comments=[data_cleaner(comment) for comment in comments]

#import pickle
#with open('/content/drive/MyDrive/Colab Notebooks/cleaned_comments.pickle', "wb") as f:
#  pickle.dump(comments, f)

#class MyCallBack(Callback):
#    def on_epoch_end(self, epoch, logs, **kwargs):
#        if logs["val_recall"] > rec_threshold and logs["val_precision"] > pre_threshold:
#            print(f"Reached precision > {pre_threshold} and recall > {rec_threshold} on test.")
#            self.model.stop_training=True
#rec_threshold=0.8
#pre_threshold=0.7
#my_cb=MyCallBack()
```

---

## 3. Repeating Pipeline Structure Per Case

Each of the 6 cases follows this code cell pattern:

| Step | Comment Pattern | Description |
|------|----------------|-------------|
| 1 | `#dataset split` | Train/test split |
| 2 | `#creation of train dataframe` | Build labeled DataFrame |
| 3 | `#creation of clean comments and toxic comments dataframe` | Separate toxic/clean |
| 4 | _(case-specific resampling — no comment or descriptive)_ | Resample strategy |
| 5 | `#tokenization of train and test sets` | Tokenize text |
| 6 | `#padding of train and test sets` | Pad sequences |
| 7 | _(MODEL CONSTRUCTION markdown header)_ | — |
| 8 | _(no comment)_ | Model architecture definition |
| 9 | _(no comment or rationale comment)_ | Compile + fit |
| 10 | _(no comment)_ | Plot training curves |
| 11 | _(no comment)_ | `model.predict()` |
| 12 | _(no comment)_ | Confusion matrix |
| 13 | _(no comment)_ | Classification report |
| 14 | _(no comment)_ | `get_labels()` helper |
| 15 | _(no comment)_ | Round predictions |
| 16 | `#examples...` | Interactive prediction display |
| 17 | _(CONCLUSIONS markdown header)_ | — |
| 18 | _(conclusion text)_ | Analysis of results |

---

## 4. Comment Style Summary

| Pattern | Example | Usage |
|---------|---------|-------|
| Category grouping | `#basics`, `#nlp`, `#viz`, `#ml` | Import sections only |
| Cell-level descriptor | `#dataset split` | Top of most code cells |
| Inline explanation | `# remove capital letters` | After code on same line |
| Rationale comment | `#recall is the most important...` | Design decisions |
| Commented-out code | `#comments=[data_cleaner(...)]` | Disabled alternatives |
| Markdown section header | `# *SECTION TITLE*` | Major sections |
| Markdown case header | `# **N° CASE: TITLE**` | Experiment cases |
| Markdown conclusion | Free-form analysis text | After each case |

### Conventions
- Comments use `#` with a space for descriptive comments (`#comment text`)
- Import category comments omit the space (`#basics`)
- No docstrings are used anywhere in the notebook
- Function definitions have no comments — the code is self-explanatory
- Markdown headers alternate between italic (`*...*`) and bold (`**...**`) styling
- Conclusion cells are plain text without bullet points (except the first conclusion)
