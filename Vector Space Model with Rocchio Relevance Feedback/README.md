# Vector Space Model with Rocchio Relevance Feedback

## 0. 

## 0. Information

### Introduction

* Implement an information retrieval system.
* Given a pool of Chinese news articles and several queries in NTCIR format, retrieve relevant documents among these articles for each query.
* Use a Vector Space Model (VSM) with Rocchio Relevance Feedback (pseudo version)

### Link
- Kaggle: [WM 2019 - VSM Model](https://www.kaggle.com/c/wm-2019-vsm-model/leaderboard)
- Data: [Google Drive](https://drive.google.com/drive/folders/1fUzce4SMaPAQLik0m3vhZpmCUdfhqfu5?usp=sharing)
- Slide: [Google Drive](https://drive.google.com/file/d/1CuxmcLR6SRloSARAKrVPgTHiCDktdjWu/view?usp=sharing)

### Environment
Language: Python 3.5.2
System: Ubuntu 16.04.6 LTS

### Execution
Usage:
```
python3 vsm.py [-r] -i INPUT_FILE -o OUTPUT_FILE -m MODEL_DIR -d NTCIR_DIR
```

Example:
```
python3 vsm.py -r -i data/queries/query-test.xml -o result.csv -m data/mode
l -d data/CIRB010
```

## 1. Vector Space Model

### Formula:

$$\sum\limits_{t \in Q,D} \ln \frac{N-df+0.5}{df+0.5} \cdot \frac{(k_1+1)tf}{(k_1(1-b+b\frac{dl}{avdl}))+tf}$$

### Variables:
$$Q: query$$
$$D: document$$
$$N:corpus\ length$$
$$df: number\ of\ documents\ contain\ the\ term$$
$$tf: term's\ frequency\ in\ document$$ 
$$dl: document\ length$$
$$avdl: average\ document\ length$$


## 2. Rocchio Relevance Feedback

### Formula:

$$\vec{Q_m} = \vec{Q_o} + \biggl(br \cdot {\tfrac{1}{|D_r|}} \cdot \sum\limits_{\vec{D_j} \in D_r} \vec{D_j}\biggr)$$

### Variables:
$$\vec{Q_o}: original\ query\ vector$$
$$\vec{Q_m}: modified\ query\ vector$$
$$D_r: related\ documents$$
$$\vec{D_j}: related\ document\ vector$$


## 3. Experiments

### 0. Default setting:
```
use_stopwords = True
k1 = 1.6
b = 0.9
threshold = 0.25
feedback_times = 3
b_r = 0.5
max_related = 10
```

### 1. Rocchio Relevance Feedback

| Feedback Times       | Kaggle Public | Kaggle Private |
| -------------------- | ------------- | -------------- |
| 0 (Without Feedback) | 0.78851       | 0.72793        |
| 1                    | 0.80914       | 0.74494        |
| 2                    | 0.81197       | 0.68697        |
| 3                    | 0.81313       | 0.73668        |
| 4                    | 0.81125       | 0.67217        |
| 5                    | 0.80939       | 0.70694        |

### 2. $$k_1$$

| $$k_1$$ | Kaggle Public | Kaggle Private |
| ------- | ------------- | -------------- |
| 1       | 0.78149       | 0.65126        |
| 1.2     | 0.77644       | 0.72572        |
| 1.4     | 0.81165       | 0.73417        |
| 1.6     | 0.81313       | 0.73668        |
| 1.8     | 0.81407       | 0.67620        |
| 2       | 0.81614       | 0.67556        |

### 3. $$b$$

| $$b$$ | Kaggle Public | Kaggle Private |
| ----- | ------------- | -------------- |
| 0.70  | 0.81232       | 0.65700        |
| 0.75  | 0.81172       | 0.67715        |
| 0.80  | 0.81211       | 0.67762        |
| 0.85  | 0.81239       | 0.73630        |
| 0.90  | 0.81313       | 0.73668        |
| 0.95  | 0.81347       | 0.73799        |
| 1.00  | 0.81365       | 0.68414        |

### 4. $$Stopwords$$

|                   | Kaggle Public | Kaggle Private |
| ----------------- | ------------- | -------------- |
| Without Stopwords | 0.78545       | 0.67161        |
| With Stopwords    | 0.81347       | 0.73799        |
