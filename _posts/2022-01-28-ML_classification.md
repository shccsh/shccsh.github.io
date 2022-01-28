---
layout: post
title:  "[머신러닝] 분류 - 평가 및 임곗값 설정 함수"
---


# 머신러닝분류_실제활용함수



```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix
from sklearn.metrics import f1_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# 경고 메시지 무시
import warnings
warnings.filterwarnings(action='ignore')
```

## 머신러닝 분류 평가지표
분류의 평가 지표 
- 정확도 (Accuracy) : 예측 결과가 동일한 데이터 건수 / 전체 예측 데이터 건수
- 오차행렬 (Confusion Matrix) : 예측 클래스와 실제 클래스의 예측 정확도를 보여주는 행렬
- 정밀도 (Precision) : 예측을 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율
- 재현율 (Recall) : 실제 값이 positive인 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율
- F1 Score : 정밀도와 재현율을 결합한 지표로 어느 한 쪽으로 치우치지 않는 수치일 때 상대적으로 높은 값을 갖는다
- ROC AUC : ROC 곡선은 FPR(False Positive Rate)이 변할 때 TPR(True Positive Rate)이 어떻게 변하는지를 나타내는 곡선으로 해당 곡선의 아래 면적을 AUC라 하고, AUC 값은 1에 가까울수록 좋은 수치

```python
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
          F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc), '\n')
```
학습을 통해 예측을 한 후, 검증을 할 때는 위의 함수를 불러오면 된다.
get_clf_eval(y_test, pred=None, pred_proba=None)
## 임계값 변화 함수

### Binarizer : 요소들이 기준값보다 큰지 작은지를 알려주는 함수
<Binarize>
요소가 기준값(threshold)과 비교해서,
- 같거나 작으면 0을 반환
- 크면 1을 반환

```python
X = [[ 1, -1, 2],
     [ 2, 0, 0],
     [ 0, 1.1, 1.2]]
```


```python
from sklearn.preprocessing import Binarizer

# Binarizer의 threshold를 1.1로 세팅.
binarizer = Binarizer(threshold = 1.1)

# array X의 값들이 1.1보다 작거나 같으면 0, 크면 1을 반환한다.
binarizer.fit_transform(X)
```




    array([[0., 0., 1.],
           [1., 0., 0.],
           [0., 0., 1.]])



### 분류 임곗값 변화에 따른 예측값 변환 함수


```python
thresholds = [0.4, 0.45, 0.50, 0.55, 0.60]
```


```python
def get_eval_by_threshold(y_test , pred_proba_c1, thresholds):
    # 임계값을 차례로 돌면서 Evaluation 수행.
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1) 
        custom_predict = binarizer.transform(pred_proba_c1)
        
        # 임계값에 따른 결과들이 출력된다.
        print('임곗값:', custom_threshold)
        get_clf_eval(y_test , custom_predict)
```
thresholds 리스트 안에 변화해 볼 임계값 수치를 적고,
아래 형태로 get_eval_by_threshold 함수를 사용하면 된다.
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)
