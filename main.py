import numpy as np
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, StandardScaler, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

import streamlit as st
import seaborn as sns
import pandas as pd
 #                                                                                                                                                                            streamlit run h:/ML/main.py

def load_data():
    """
    Загрузка данных
    """
    data = pd.read_csv('H:\ML\insurance.csv')
    return data


@st.cache
def preprocess_data(data_in):
    """
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    """

    data_out = data_in.copy()
    # Числовые колонки для масштабирования
    scale_cols = ['age', 'bmi']
    new_cols = []
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        new_cols.append(new_col_name)
        data_out[new_col_name] = sc1_data[:, i]

    scale_cols = ['charges']
    new_cols2 = []
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        new_cols2.append(new_col_name)
        data_out[new_col_name] = sc1_data[:, i]

    X = data_out[new_cols]
    Y = data_out[new_cols2]
    # Чтобы в тесте получилось низкое качество используем только 0,5% данных для обучения
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test, X, Y


data = load_data()



st.sidebar.header('Случайный лес')
n_estimators_1 = st.sidebar.slider('Количество фолдов:', min_value=3, max_value=10, value=3, step=1)

st.subheader('RandomForestRegressor')


X_train, X_test, Y_train, Y_test, X, Y = preprocess_data(data)
forest_1 = RandomForestRegressor(n_estimators = n_estimators_1, oob_score=True, random_state=10)
forest_1.fit(X_train, Y_train)
Y_predict = forest_1.predict(X_test)

st.subheader('Первые 5 значений')
st.write(X_train.head())

fig1 = plt.figure(figsize=(5, 4))
ax = plt.scatter(X_test['age_scaled'], Y_test, marker='o', label='Тестовая выборка')
plt.scatter(X_test['age_scaled'], Y_predict, marker='.', label='Предсказанные данные')
plt.legend(loc='lower right')
plt.xlabel('age_scaled')
plt.ylabel('charges')
plt.plot(n_estimators_1)
st.pyplot(fig1)
