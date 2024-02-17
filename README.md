프로젝트 주제 : 음성 인식 데이터로 성별 판별하기

프로젝트 동기 : 이전부터 하고 싶었던 주제였는데 방학 동안 프로젝트를 만들 수 있는 기회가 생겨서 선정하게 되었다.


프로젝트 설명

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection, sklearn.linear_model, sklearn.svm, sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
import os
```


필요한 라이브러리들을 불러와준다. 여러 모델들을 학습시키기 위해 불러와준다.

```python
df = pd.read_csv('voice.csv')
```

필요한 데이터를 가져온다.

```python
df.head()
```

데이터를 확인하여 준다.

![image](https://github.com/SUNRINEmotion/7th-taegyun_project/assets/110581786/b556bd99-39c0-4340-b908-1b51c7f3d631)

```python
df.describe()
```

데이터의 통계량을 확인해준다.

![image](https://github.com/SUNRINEmotion/7th-taegyun_project/assets/110581786/6d3a3b0d-0ae1-4a29-a08b-bb9564d12deb)

```python
df.info()
```


데이터의 전반적인 정보도 확인하여준다.

![image](https://github.com/SUNRINEmotion/7th-taegyun_project/assets/110581786/c98b44b5-a0d6-4db0-8928-bae5c6840a6f)


마지막으로 데이터에 결측치가 있는지 확인해 준다.

```python
df.isnull().sum()
```

![image](https://github.com/SUNRINEmotion/7th-taegyun_project/assets/110581786/112e79fe-3cf4-4965-b2b4-8da97a47f66e)

결측치가 없으므로 진행해준다.

```python
df.replace(to_replace="male", value=1, inplace=True)
df.replace(to_replace="female", value=0, inplace=True)
df.label.unique()
```

이진 분류를 위해 성별을 나타내는 데이터 라벨을 0과 1로 바꿔준다.

```python
xData=df.iloc[:,:-1]
yData=df.iloc[:,-1]
xData.shape, yData.shape
```

모델을 학습시키기 위해 라벨과 입력 데이터를 분리해준다.

```python
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
df.label.value_counts().plot(kind="pie",fontsize=16,labels=["Male", "Female"],ylabel="Male vs Female",autopct='%1.1f%%');
plt.subplot(1, 2, 2)
sns.countplot(x="label",data=df, palette="pastel")
plt.show()
```

성별의 분포를 그래프로 확인해준다.

![image](https://github.com/SUNRINEmotion/7th-taegyun_project/assets/110581786/923971eb-9ecc-42d0-9bea-ae96246ca8cc)

```python
plt.figure(figsize=(12,8))
data = df.corr()["label"].sort_values(ascending=False)
indices = data.index
labels = []
corr = []
for i in range(1, len(indices)):
    labels.append(indices[i])
    corr.append(data[i])
sns.barplot(x=corr, y=labels, palette='pastel')
plt.title('Correlation coefficient between different features and Label')
plt.show()
```

성별과 각 특성 데이터 사이의 상관관계를 나타내어 구분해준다.

성별의 차이에 따라 특성에서 나타나는 상관관계가 달라지는 것을 볼 수 있다.

![image](https://github.com/SUNRINEmotion/7th-taegyun_project/assets/110581786/df28fe6c-af6d-494e-b7b4-1f0d1cf668a6)

이 데이터들을 가지고 여러 모델을 학습시켜준다. 내가 사용한 모델들은 로지스틱 회귀, KNN, 가우시안 프로세스 분류 모델들이다. 이들을 사용한 이유는 데이터를 분류할 때 주로 사용되는 모델들이라서 사용하게 되었다.

```python
regressionModel = LogisticRegression(solver='liblinear')
regressionModel.fit(xTrain,yTrain)
regressionModel.score(xTrain,yTrain)
KNNModel = KNeighborsClassifier(n_neighbors=3)
KNNModel.fit(xTrain,yTrain)
KNNModel.score(xTrain,yTrain)
gpcModel = GaussianProcessClassifier()
gpcModel.fit(xTrain, yTrain)
gpcModel.score(xTrain, yTrain)
```

그 후 모델들의 성능을 그래프로 비교해준다.

![image](https://github.com/SUNRINEmotion/7th-taegyun_project/assets/110581786/76ca10ae-f4a5-4611-b17e-00b69d378da4)

![image](https://github.com/SUNRINEmotion/7th-taegyun_project/assets/110581786/4eb18433-632b-42ec-8836-5a2e25e215e6)

로지스틱 회귀 모델의 정확도가 가장 높은 것을 확인할 수 있다. 즉, 분류를 가장 잘 해냈다는 것을 알 수 있다.

