from sklearn.datasets import load_iris #데이터셋 임포트

iris_dataset = load_iris()
print("Iris_data set key {}".format(iris_dataset.keys()))

from sklearn.model_selection import train_test_split

train_inpiut, test_input, train_label, test_label = train_test_split(
    iris_dataset['data'],
    iris_dataset['target'],
    test_size=0.25,
    random_state=42
)

# 위 함수는 데이터값과 정답을 받아들이고
# test size 인자값으로 테스트 데이터를
# 전체 데이터의 25%로 설정하여 이 비율로 데이터를 나눔

from sklearn.neighbors import KNeighborsClassifier
# 나이브베이지 추론
knn = KNeighborsClassifier(n_neighbors=1)
# knn 객체 생성, 하이퍼 파라미터로k 값.

knn.fit(train_inpiut, train_label)
# fit 메서드를 이용하여 학습 시킨다.

predict_label = knn.predict(test_input)
print(predict_label)

import numpy as np
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
result_array = knn.predict(new_input)

print(result_array) 