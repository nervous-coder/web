import streamlit as st
import pickle
import base64
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from collections import Counter

st.write("""# Использование методов машинного обучения для планов заготовительного производства""")

st.header('Нажмите на кнопку ниже чтобы узнать с какой точностью работает модель')

if st.button('Узнать как работает модель'):
    test = pd.read_csv('data.csv')

    a = []
    for i in test['workload']:
        a.append(math.ceil(i))
    test['workload'] = a

    b = []
    for i in test['pf_time']:
        b.append(math.ceil(i))
    test['pf_time'] = b

    st.write('Ниже изображена тепловая карта матрицы взаимной корреляции. Она наглядно показывает по каким признакам будет производиться прогнозирование. Визуализирует зависимость признаков между собой.')

    k = 15
    cols = test.corr().nlargest(k, 'workload')['workload'].index
    cm = np.corrcoef(test[cols].values.T)
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cm, ax=ax, cmap="YlGnBu", linewidths=0.1, yticklabels=cols.values, xticklabels = cols.values, annot=True, fmt=".2f")
    st.pyplot()

    test_data = test
    X = test_data.drop(['workload', 'id'], axis = 1)
    y = test_data['workload']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)

    model = RandomForestRegressor(n_estimators = 100, min_samples_split = 10, min_samples_leaf = 1, max_features = 'auto', n_jobs = -1)
    model.fit(X_train, y_train)
    model_prediction = model.predict(X_train)
    model.score(X_train, y_train)
    rez_model = round(model.score(X_train, y_train) * 100, 2)
    rez_model_to_str = "Точность прогнозирования: {}%".format(rez_model)
    st.write(rez_model_to_str)

st.header('Ввод и загрузка данных')

uploaded_file = st.file_uploader('Загрузите CSV файл', type=["csv"])
plane_id = st.text_input(label = 'Введите идентификатор плана')
article = st.text_input(label = 'Введите артикул')
batch_size = st.text_input(label = 'Введите размер партии')
pf_time = st.text_input(label = 'Введите подготовительно-заключительное время')

def user_data():
    plane_id.isdigit()
    article.isdigit()
    batch_size.isdigit()
    pf_time_int = math.ceil(pf_time.isdigit())
    data = {
        'plane_id': plane_id,
        'article': article,
        'batch_size': batch_size,
        'pf_time': pf_time_int,
    }

    custom_data = pd.DataFrame(data, index = [0])
    return custom_data

if st.button('Осуществить прогноз'):
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        df = input_df
        df.drop(['id'], axis=1, inplace = True)
        source_df = df
    else:
        df = user_data()
        source_df = user_data()

    st.subheader('Исходные данные')
    st.write(df)

    a = []
    for i in df['pf_time']:
        a.append(math.ceil(i))
    df['pf_time'] = a

    try:
        model = pickle.load(open('model.pkl', 'rb'))
        predict = model.predict(df)
        st.subheader('Прогноз')
        st.write(predict)

        source_df['workload'] = predict
        st.subheader('Предпросмотр выгружаемых данных')
        st.write(source_df)

        generate_csv = source_df.to_csv(index = False)
        convert_to_base64 = base64.b64encode(generate_csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{convert_to_base64}" download="output.csv">Скачать данные в формате CSV</a>'

        st.subheader('Для скачивания прогноза нажмите на ссылку ниже')
        st.markdown(href, unsafe_allow_html = True)

        total_plane = source_df

        total_plane_list = total_plane['plane_id']
        counter_array = dict(Counter(total_plane_list))
        total_plane_ids = []
        for i in counter_array:
            total_plane_ids.append(i)

        total_plane_sum = total_plane.groupby(['plane_id']).sum()
        total_plane_sum['plane_id'] = total_plane_ids
        st.subheader('Примеры планов')
        if uploaded_file is not None:
            total_plane_sum.drop(['article', 'party_size'], axis=1, inplace = True)
        st.write(total_plane_sum)

        generate_csv_sum = total_plane_sum.to_csv(index = False)
        convert_to_base64_sum = base64.b64encode(generate_csv_sum.encode()).decode()
        href_sum = f'<a href="data:file/csv;base64,{convert_to_base64_sum}" download="output_sum.csv">Скачать данные в формате CSV</a>'

        st.subheader('Для скачивания примера планаов нажмите на ссылку ниже')
        st.markdown(href_sum, unsafe_allow_html = True)

    except:
        st.error('Что-то пошло не так. Возможно неверно введены данные, пожалуйста перепроверьте')
