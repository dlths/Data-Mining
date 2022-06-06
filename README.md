# data-mining
# Datamining-Team-Project
## 주제 : 한국인의 고혈압 발생 요인 분석 및 예측

### 분석 배경
우리나라의 고혈압 유병률(만 30세 이상)은 2019년 기준 27.2%로 성인 인구 100명 당 27명 꼴로 고통받고 있다.(질병관리청, 2019)\
인구의 평균수명이 증가함에 따라 고혈압의 유병 기간도 길어질 것으로 예상됨에 따라 고혈압은 지속적으로 국가의 중요한 보건문제가 될 것이다.(OECD, 2013; 윤희숙, 2013)

연령이 높이질수록 복합만성질환을 많이 보유하고 있는데 고혈압은 그 자체보다는 합병증인 관상동맥 심장질환, 만성심부전, 뇌졸중과 같은 질환이 발생 시
환자에게 많은 고통을 줄 뿐만 아니라 경제적 부담까지 가중시키기 때문에 국가적인 관리가 요구되는 매우 중요한 질환이다.

### 분석 목적
1. 머신러닝 기법 중 White Box에 속하는 Decision Tree와, feature importance를 볼 수 있는 Random forest를 사용하여 어떤 요인들이 고혈압을 발생시키는지 알아본다.
   또한 파악된 요인들을 어떻게 개선시킬 것인지 방안을 제안해 보고자 한다.

2. SVM(support vector machine)과 KNN(K-Nearest Neighbors) 머신러닝을 사용하여 고혈압 발생환자를 예측하고, 현재는 고혈압 환자가 아니지만 추후에 고혈압이 발생될 가능성이 있는지 판별할 수 있도록 예측모델을 만들어보고자 한다.

### 데이터 획득
KoGES 자료를 Kaggle에서 데이터를 획득
한국인유전체조사사업(Korean Genome and Epidemiology Study; KoGES)은 한국인에게 흔한 만성 질환의 위험요인을 규명하여 맞춤 예방의학 구현의 과학적 근거를 마련하고자 질병관리청 국립보건 연구원이 수행하는 코호트 사업이다.\
2001년부터 일반 인구집단을 대상으로 참여자를 모집하여 건강 및 생활습관 관련 설문조사와 검진을 수행하고, 혈액, 소변, DNA 등의 인체유래물을 수집하였고, 새로운 질병발생과 생활습관 변화에 대한 추적조사를 진행하였다.\
KoGES 출처 : https://www.kdca.go.kr/contents.es?mid=a40504010000 \
Kaggle 출처 : https://www.kaggle.com/datasets/junsoopablo/korean-genome-and-epidemiology-study-koges

### 분석과정
``` python
import mglearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# follow_01_data 불러오기
data1 = pd.read_csv("follow_01_data.csv")

# follow_01_data 중 사용할 feature만 걸러내기
data1_1 = data1[['T00_ID', 'T00_SEX', 'T01_AGE', 'T01_DRINK', 'T01_DRDU', 'T01_SMOKE', 'T01_SMDU', 'T01_EXER', 'T01_FMFHT', 'T01_FMMHT',
             'T01_FMFDM', 'T01_FMMDM', 'T01_HEIGHT', 'T01_WEIGHT', 'T01_WAIST', 'T01_HIP', 'T01_PULSE', 'T01_SBP', 'T01_DBP',
              'T01_HBA1C', 'T01_GLU0', 'T01_CREATININE', 'T01_AST', 'T01_ALT', 'T01_TCHL', 'T01_HDL', 'T01_TG', 'T01_INS0', 'T01_HTN']]
###
변수 설명

(설명 변수 - T00_ID, T01_HTN 제외)
T00_ID : 개인식별자
T00_SEX : 성별 (1 = 남자, 2 = 여자)
T01_AGE : 만 나이 (세)
T01_DRINK : 음주여부 (1 = 예(비음주), 2 = 아니오(과거음주), 3 = 아니요(현재 음주))
T01_DRDU : 총 음주기간 (1 = 5년 이하, 2 = 6 ~ 10년, 3 = 11~ 20년, 4 = 21년 이상)
T01_SMOKE : 흡연여부 (1 = 아니오(비흡연), 2 = 예(과거흡연), 3 = 예(현재흡연))
T01_SMDU : 총 흡연기간 (년)
T01_EXER : 몸에 땀이 날 정도의 운동여부 (1 = 아니오, 2 = 예)
T01_FMFHT : 고혈압 가족력 (부) 진단 여부 (1 = 아니오, 2 = 예)
T01_FMMHT : 고혈압 가족력 (모) 진단 여부 (1 = 아니오, 2 = 예)
T01_FMFDM : 당뇨병 가족력 (부) 진단 여부 (1 = 아니오, 2 = 예)
T01_FMMDM : 당뇨병 가족력 (모) 진단 여부 (1 = 아니오, 2 = 예)
T01_HEIGHT : 신장 (cm)
T01_WEIGHT : 체중 (kg)
T01_WAIST : 허리둘레 평균 (cm)
T01_HIP : 엉덩이둘레 평균 (cm)
T01_PULSE : 맥박 평균 (회/분)
T01_SBP : 수축기혈압 (mmHg)
T01_DBP : 이완기혈압 (mmHg)
T01_HBA1C : HbA1C (%)
T01_GLU0 : Fasting Glucose (mg/dL)
T01_CREATININE : Creatinine (mg/dL)
T01_AST : AST (IU/L)
T01_ALT : ALT (IU/L)
T01_TCHL : Total Cholesterol (mg/dL)
T01_HDL : HDL-CHolesterol (mg/dL)
T01_TG : Triglyceride (mg/dL)
T01_INS0 : Fasting Insulin (uIU/ml)
T01_HTN : 고혈압 진단 여부 (1 = 아니오, 2 = 예) 기반조사의 경우 : 일생동안 의사로부터 질병을 진단 받은 적이 있는지?
###


# follow_02_data 불러오기
data2 = pd.read_csv("follow_02_data.csv")

# follow_02_data 중 사용할 feature만 걸러내기
data2_1 = data2[['T00_ID', 'T02_HTN']]
###
변수 설명

(타겟 변수  T00_ID 제외)
T00_ID : 개인식별자
T02_HTN : 고혈압 진단 여부 (1 = 아니오, 2 = 예) 추적조사의 경우 : 지난 코호트 검진  의사로부터 질병을 진단 받은 적이 있는지?
###

# ID를 기준으로 두 데이터 자료 합치기
df = pd.merge(data1_1, data2_1)

# T01HTN = T02HTN = 2인 행 제거 (기존에도 고혈압이었는데 추적조사에도 고혈압인 대상)
df1 = df.drop(index= df[(df['T01_HTN']==2) & (df['T02_HTN']==2)].index)

# T01HTN = 2 T02HTN = 1인 행 제거 (기존에는 고혈압이었는데 추적조사에는 정상으로 대상)
df2 = df1.drop(index= df1[(df1['T01_HTN']==2) & (df1['T02_HTN']==1)].index)

# T01HTN 열 제거
df3 = df2.drop('T01_HTN', axis='columns')

# 해당없음 77777을 0으로 convert
replace_column=['T00_ID', 'T00_SEX', 'T01_AGE', 'T01_DRINK', 'T01_DRDU', 'T01_SMOKE', 'T01_SMDU', 'T01_EXER', 'T01_FMFHT', 'T01_FMMHT',
             'T01_FMFDM', 'T01_FMMDM', 'T01_HEIGHT', 'T01_WEIGHT', 'T01_WAIST', 'T01_HIP', 'T01_PULSE', 'T01_SBP', 'T01_DBP',
              'T01_HBA1C', 'T01_GLU0', 'T01_CREATININE', 'T01_AST', 'T01_ALT', 'T01_TCHL', 'T01_HDL', 'T01_TG', 'T01_INS0', 'T02_HTN']

for col in replace_column:
    df3[col].replace(77777.0,0,inplace=True)
    
# Yes or no가 1 또는 2를 0 또는 1로 convert
categorical=['T00_SEX','T01_EXER','T01_FMFHT','T01_FMMHT','T01_FMFDM','T01_FMMDM', 'T02_HTN']
df3[categorical]=df3[categorical].replace({1.0:0,1:0,2.0:1,2:1})

# 미상, 무응답, 미측정 99999가 포함된 행 제거
mask = df3[['T00_ID', 'T00_SEX', 'T01_AGE', 'T01_DRINK', 'T01_DRDU', 'T01_SMOKE', 'T01_SMDU', 'T01_EXER', 'T01_FMFHT', 'T01_FMMHT',
             'T01_FMFDM', 'T01_FMMDM', 'T01_HEIGHT', 'T01_WEIGHT', 'T01_WAIST', 'T01_HIP', 'T01_PULSE', 'T01_SBP', 'T01_DBP',
              'T01_HBA1C', 'T01_GLU0', 'T01_CREATININE', 'T01_AST', 'T01_ALT', 'T01_TCHL', 'T01_HDL', 'T01_TG', 'T01_INS0', 'T02_HTN']].isin([99999.0])

df4 = df3[~mask]

df5 = df4.dropna()

# 조사안함 66666가 포함된 값은 평균값으로 대체 'T01_PULSE', 'T01_HBA1C', 'T01_INS0'
df6 = df5.replace(66666.0, np.NAN)

df6.loc[df6['T01_PULSE'] != df6['T01_PULSE'], 'T01_PULSE'] = df6['T01_PULSE'].mean()
df6.loc[df6['T01_HBA1C'] != df6['T01_HBA1C'], 'T01_HBA1C'] = df6['T01_HBA1C'].mean()
df6.loc[df6['T01_INS0'] != df6['T01_INS0'], 'T01_INS0'] = df6['T01_INS0'].mean()
```
#### 전처리 전 data의 EDA
```python
h1 = df.iloc[1:,:].hist(figsize=(30,20))
```
![image](https://user-images.githubusercontent.com/105834310/170257259-29e3eb2e-bb27-420f-a77d-2d3c28331730.png)
#### 전처리 후 data의 EDA
```python
h2 = df6.iloc[1:,:].hist(figsize=(30,20))
```
![image](https://user-images.githubusercontent.com/105834310/170257648-2ecd2c31-d514-4fba-b07e-686582680365.png)
```

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df5[['T00_SEX', 'T01_AGE', 'T01_DRINK', 'T01_DRDU', 'T01_SMOKE', 'T01_SMDU', 'T01_EXER', 'T01_FMFHT', 'T01_FMMHT',
             'T01_FMFDM', 'T01_FMMDM', 'T01_HEIGHT', 'T01_WEIGHT', 'T01_WAIST', 'T01_HIP', 'T01_PULSE', 'T01_SBP', 'T01_DBP',
              'T01_HBA1C', 'T01_GLU0', 'T01_CREATININE', 'T01_AST', 'T01_ALT', 'T01_TCHL', 'T01_HDL', 'T01_TG', 'T01_INS0']], df5['T02_HTN'], random_state=0)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
```
(648, 27) (648,) (216, 27) (216,)
### Oversampling
#### 정상인과 고혈압 환자의 class 비율이 불균형하여, 이는 질병 예측에 있어 모델이 대부분 정상인으로 판별할 가능성이 있어 Oversampling을 수행
```python
print("counts of label '1': {}".format(sum(y_train==1)))
print("counts of label '0': {}".format(sum(y_train==0)))
```
counts of label '1': 27\
counts of label '0': 513
```python
print("counts of label '1': {}".format(sum(y_test==1)))
print("counts of label '0': {}".format(sum(y_test==0)))
```
counts of label '1': 10\
counts of label '0': 171
```python
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, sampling_strategy=0.5)

X_resampled, y_resampled = sm.fit_resample(X_train,y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_resampled.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_resampled.shape))
```
After OverSampling, the shape of train_X: (939, 27)\
After OverSampling, the shape of train_y: (939,) 
```python
print("After OverSampling, counts of label '1': {}".format(sum(y_resampled==1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_resampled==0)))
```
After OverSampling, counts of label '1': 313\
After OverSampling, counts of label '0': 626
```python
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=1)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
```
(615, 27) (615,) (154, 27) (154,)
### 고혈압 발생 환자들의 주요 원인(특징) 파악하고 차후에 고혈압이 발생할 수 있는 사람을 판별할 수 있는 예측모델
#### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

training_accuracy = []
validation_accuracy = []
cross_val_accuracy = []

md_settings = [1, 2, 5, 7, 10, 20]
mss_settings = [2, 3, 5, 7, 10, 30]

for md in md_settings:
    for mss in mss_settings:
        # model training with labeled training data
        clf = DecisionTreeClassifier(max_depth= md, min_samples_split =mss, random_state=0)
        clf.fit(X_train, y_train)
        
        # prediction
        y_train_hat = clf.predict(X_train)
        y_val_hat = clf.predict(X_val)

        # evaluation    
        training_accuracy.append(accuracy_score(y_train, y_train_hat))
        validation_accuracy.append(accuracy_score(y_val, y_val_hat))
        
        # K-fold cross_validation (k = 10)
        kf = KFold(n_splits = 10)
        scores = cross_val_score(clf, X_train, y_train, cv = kf)
        cross_val_accuracy.append(scores.mean())
        
result1 = pd.DataFrame({"max_depth":sorted(md_settings*6), "min_sample_split":mss_settings*6, "training accuracy": training_accuracy, "validation accuracy": validation_accuracy, "cross_val_accuracy":cross_val_accuracy})validation_accuracy})

result1.loc[result1['cross_val_accuracy'].idxmax()]
```
max_depth              10.000000\
min_sample_split        5.000000\
training accuracy       0.991870\
validation accuracy     0.902597\
cross_val_accuracy      0.910603\
Name: 26, dtype: float64
```python
# model test
clf = DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state=0)
clf.fit(X_train_val, y_train_val)

# prediction
y_test_hat = clf.predict(X_test)

# evaluation
test_accuracy = accuracy_score(y_test, y_test_hat)

print("Accuracy on test set: ", test_accuracy)
```
Accuracy on test set:  0.9060773480662984
```python
matrix = confusion_matrix(y_test, y_test_hat)
sns.heatmap(matrix, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
print(classification_report(y_test, y_test_hat))
```
![image](https://user-images.githubusercontent.com/105834310/170264455-53484905-01c8-4412-a6c2-ee78111bfba7.png)
```python
import graphviz

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['T00_SEX', 'T01_AGE', 'T01_DRINK', 'T01_DRDU', 'T01_SMOKE', 'T01_SMDU', 'T01_EXER', 'T01_FMFHT', 'T01_FMMHT',
                                               'T01_FMFDM', 'T01_FMMDM', 'T01_HEIGHT', 'T01_WEIGHT', 'T01_WAIST', 'T01_HIP', 'T01_PULSE', 'T01_SBP', 'T01_DBP',
                                               'T01_HBA1C', 'T01_GLU0', 'T01_CREATININE', 'T01_AST', 'T01_ALT', 'T01_TCHL', 'T01_HDL', 'T01_TG', 'T01_INS0'],  
                                class_names= 'T02_HTN',
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph
```
![image](https://user-images.githubusercontent.com/105834310/170284422-5ce14801-867f-4012-9d0c-69f8fc3826fe.png)
```python
f_i=pd.Series(clf.feature_importances_, index=X_train.columns)
f_i.sort_values(ascending=False)
import matplotlib.pyplot as plt
f_i.sort_values().plot(kind='barh',figsize=(10,6))
plt.show()
```
![image](https://user-images.githubusercontent.com/105834310/170264511-269189df-d938-40aa-8634-6fe60789ca53.png)
#### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

training_accuracy= []
validation_accuracy = []
cross_val_accuracy = []

n_settings = [1, 2, 5, 10, 20, 30, 50]
max_depth = [2, 3, 5, 7, 10]

for n in n_settings:
    for depth in max_depth:
        rf = RandomForestClassifier(n_estimators=n, max_depth = depth, random_state=0)
        rf.fit(X_train, y_train)
    
        # prediction
        y_train_hat = rf.predict(X_train)
        y_val_hat = rf.predict(X_val)
    
        # evluation
        training_accuracy.append(accuracy_score(y_train, y_train_hat))
        validation_accuracy.append(accuracy_score(y_val, y_val_hat))
        
        # K-fold cross_validation (k = 10)
        kf = KFold(n_splits = 10)
        scores = cross_val_score(rf, X_train, y_train, cv = kf)
        cross_val_accuracy.append(scores.mean())
    
result2 = pd.DataFrame({'n_estimators':n_settings * 5, 'max_depth':max_depth * 7, 'training accuracy': training_accuracy, 'validation accuracy': validation_accuracy, "cross_val_accuracy":cross_val_accuracy})

result2.loc[result2['cross_val_accuracy'].idxmax()]
```
n_estimators           50.000000\
max_depth              10.000000\
training accuracy       1.000000\
validation accuracy     0.948052\
cross_val_accuracy      0.970730\
Name: 34, dtype: float64
```python
# model test
rf = RandomForestClassifier(n_estimators= 50, max_depth = 10, random_state = 0)
rf.fit(X_train_val, y_train_val)

# prediction
y_test_hat = rf.predict(X_test)

# evaluation
test_accuracy = accuracy_score(y_test, y_test_hat)

print("Accuracy on test set: ", test_accuracy)
```
Accuracy on test set:  0.9337016574585635
```python
matrix = confusion_matrix(y_test, y_test_hat)
sns.heatmap(matrix, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
print(classification_report(y_test, y_test_hat))
```
![image](https://user-images.githubusercontent.com/105834310/170259992-6e2b45d8-abe3-4827-aaae-828dbeb5e46e.png)
```python
f_i=pd.Series(rf.feature_importances_, index=X_train.columns)
f_i.sort_values(ascending=False)
f_i.sort_values().plot(kind='barh',figsize=(10,6))
plt.show()
```
![image](https://user-images.githubusercontent.com/105834310/170260047-a6edd019-e661-4232-aee7-3dadf27e4602.png)
#### Support Vector Machine
```python
from sklearn.svm import SVC

training_accuracy = []
validation_accuracy = []
cross_val_accuracy = []

C_settings = [0.01, 0.1, 1, 10, 100]      
gamma_settings = [0.01, 0.1, 0.5, 0.7, 1]

for C in C_settings:
    for gamma in gamma_settings:
        svc = SVC(C = C, kernel = 'rbf', gamma = gamma)
        svc.fit(X_train, y_train)

        # prediction
        y_train_hat = svc.predict(X_train)
        y_val_hat = svc.predict(X_val)
    
        # evluation
        training_accuracy.append(accuracy_score(y_train, y_train_hat))
        validation_accuracy.append(accuracy_score(y_val, y_val_hat))
        
        # K-fold cross_validation (k = 10)
        kf = KFold(n_splits = 10)
        scores = cross_val_score(svc, X_train, y_train, cv = kf)
        cross_val_accuracy.append(scores.mean())
        
result3 = pd.DataFrame({'C':sorted(C_settings*5), 'gamma':gamma_settings*5, 'training accuracy': training_accuracy, 'validation accuracy': validation_accuracy, "cross_val_accuracy":cross_val_accuracy})

result3.loc[result3['cross_val_accuracy'].idxmax()]
```
C                      10.000000\
gamma                   0.010000\
training accuracy       1.000000\
validation accuracy     0.896104\
cross_val_accuracy      0.905500\
Name: 15, dtype: float64
```python
# model test
svc = SVC(C = 10, kernel = 'rbf', gamma = 0.01)
svc.fit(X_train_val, y_train_val)

# prediction
y_test_hat = svc.predict(X_test)

# evaluation
test_accuracy = accuracy_score(y_test, y_test_hat)

print("Accuracy on test set: ", test_accuracy)
```
Accuracy on test set:  0.9447513812154696
```python
matrix = confusion_matrix(y_test, y_test_hat)
sns.heatmap(matrix, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
print(classification_report(y_test, y_test_hat))
```
![image](https://user-images.githubusercontent.com/105834310/170260401-fa727f18-d1e7-446e-b3c6-5b07986625bf.png)
#### K Nearest Neighbor
```python
from sklearn.neighbors import KNeighborsClassifier

score_test = []

n_neighbor = [1, 2, 3, 4, 5]      

for n in n_neighbor:

    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train_val, y_train_val)

    y_pred = knn.predict(X_test)
    
    score_test.append(accuracy_score(y_test, y_pred))
    
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_val, y_train_val)
```
KNeighborsClassifier(n_neighbors=2)
```python
y_pred = knn.predict(X_test) #=y_pred=y_hat
print("Test set predictions:\n", y_pred)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
```
Test set score: 0.88
```python
#Confusion matrix and classification report
matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
print(classification_report(y_test, y_pred))
```
![image](https://user-images.githubusercontent.com/105834310/170260747-4544ae12-9fdb-49ca-b3f7-58f5b84a115f.png)
### 결과 & 기대효과
결과
Decision Tree, Random Forest, KNN, SVM 총 4개의 모델을 통해 생활습관과 인체유래물 정보를 바탕으로 향후 고혈압이 발생하는 환자를 예측해 보았다.
질병 예측 분야에서는 예측모델에 있어 환자인데 환자로 판별해야 하는 것이 매우 중요하므로 recall(재현율)을 중요하게 바라본다
따라서 4개의 모델 중 고혈압 환자 recall 값이 가장 높은 Decision Tree를 최종 모델로 선정하여 추가적으로 해석하였다

decision tree 결과 분석
(1) BP <= 125.5 HEIGHT <=150.5  WEIGHT <= 58.5  ALT <= 20.0  AST <= 25.0

(2) SBP > 125.5 PULSE <= 63.151 TCHL > 154.5 PULSE > 60.029 HBA1C > 5.852

(3) SBP > 125.5 PULSE > 63.151 CREATININE > 0.999 AGE > 60.5 ALT >13.5  GLU > 107.344 HDL > 32  HBA1C > 6.084  FMFHT > 0.5  TG > 67.5

(4) SBP > 125.5 PULSE > 63.151 CREATININE > 0.999 AGE > 60.5 TG > 193.16 PULSE > 71.633 WEIGHT > 51

(1)번의 경우를 제외하고 나머지(2), (3), (4)의 경우는 수축기 혈압이 125.5보다 높은것을 알 수 있다
특히 가장 많은 샘플 결과인 (3)은 수축기 혈압 변수 말고 다른 변수들은 정상에 가까운데 이는 FMFHT 가족력 즉 아버지가 고혈압일 경우이므로 유전일 확률이 높은것으로 해석된다

고혈압을 결정짓는 가장 영향력 있는 변수 SBP(수축기 혈압)과, PULSE(맥박)이다

1. 젊었을 때 말랑말랑하던 혈관은 점차 탄력을 잃고 딱딱해진다.
쉽게 넓어지거나 좁아지지는 않는다. 혈액량이 늘어났음에도 혈관이 적절히 넓어지지 않기 때문에 혈관벽에 가해지는 압력이 높아지고 이 때문에 수축기 혈압이 높게 나타난다.\

2. 심장박동수가 빨라지는 주요 원인은 부정맥이다.
부정맥은 동빈맥, 상심실성 빈맥, 심실성 빈맥 등이 있고 부정맥 원인으로는 각종 심장 질환, 폐질환, 전신질환, 심근의 전기적 현상 이상, 전해질 대사 이상, 약물 중독 등을 꼽을 수 있다.\
정상인이라도 흥분하거나 과도한 스트레스를 받으면 맥박이 100회 이상으로 증가할 수 있다.

이를 예방할 수 있는 방법으로는 생활 습관을 바꾸는 것이다. 특히 건강한 식습관과 운동을 꾸준히 하는 습관을 가져야한다.
1. 흡연과 음주는 최대한 지양해야 한다.
2. 음식은 나트륨 함량이 적게, 채소와 생선을 골고루 섭취해야한다.
3. 가능한 한 매일 30분 이상 적절한 운동을 한다.
4. 적정 체중과 허리둘레를 유지한다.
5. 스트레스를 줄이고, 즐거운 마음으로 생활한다.
8. 정기적으로 혈압, 혈당, 콜레스테롤을 측정한다.
9. 고혈압, 당뇨병, 이상지질혈증(고지혈증)을 꾸준히 치료한다.
10. 뇌졸중, 심근경색증의 응급 증상을 숙지하고 발생 즉시 병원에 간다.

최종적으로 이 프로젝트를 통해서 생활습관 및 병원에서 검사를 통한 인체유래물을 바탕으로 향후에 고혈압이 걸릴 확률이 높은 대상자를 찾아 미리 예방할 수 있을 것으로 기대된다.

### 한계점 및 추후 개선 방안
한계점
1. Data의 개수가 충분하지 않고, 조사를 안 한 부분에 대해서는 전처리과정을 거쳤지만 그 수가 얼추 되기 때문에 모델을 100% 신뢰하기 어렵다
2. 각 feature들의 값의 범위가 다양해서 모델 분석에 영향을 끼쳤을 것으로 예상된다.

추후 개선 방안
1. 꾸준한 Data 수집을 통해 정확성을 높인다.
2. 각 feature들이 모델에 공평하게 영향ㅇ르 줄 수 있는 전처리 방안을 추가로 적용하여 적용하지 않은 데이터와의 결과를 비교해 볼 필요가 있다
