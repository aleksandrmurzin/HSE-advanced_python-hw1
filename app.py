"""
[x] построение графиков распределений признаков
* построение матрицы корреляций
* построение графиков зависимостей целевой переменной и признаков
[x] вычисление числовых характеристик распределения числовых столбцов (среднее, min, max, медиана и так далее)
"""


import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

class Data:
    def __init__(self, path: str = "data/full_dataset.csv", data = None):
        self._path = path
        self._data = self._load_data() if data is None else data
        self._data_info = {}
    
    def _load_data(self):
        return pd.read_csv(self._path)

    def save_data(self):
        self.data.to_csv(f"updated_dataset_{datetime.today().strftime('%m/%d/%Y')}.csv")
        
    @property
    def data_info(self):
        info = self._update_info()
        return info
    
    @property
    def data(self):
        return self._data
    
    def _update_info(self):
        self._data_info["num_rows"] = self._data.shape[0]
        self._data_info["num_columns"] = self._data.shape[1]
        self._data_info["columns"] = self._data.columns.to_list()
        self._data_info["numerical_columns"] = self._data.select_dtypes(include=[int, float]).columns.to_list()
        self._data_info["catigorical_columns"] = self._data.select_dtypes(include=[object, bool]).columns.to_list()
        self._data_info["target_columns"] = ["TARGET"]
        return self._data_info

data_handler = Data()


st.time_input("current time", disabled=True)
st.title('EDA for banking data')


data_load_state = st.text('Loading data...')
st.subheader('Initial data')
st.write(data_handler.data.head(5))
st.write(data_handler.data.tail(5))
data_load_state.text('Loading data...done!')


st.subheader('Data description')
st.write(data_handler.data_info)


# add selection-box widget
msg = "Which column do you want to explore?"
selected_num_col = st.selectbox(msg, sorted(data_handler.data_info["columns"]))

# show histograms
if selected_num_col in data_handler.data_info["numerical_columns"]:
    st.header(f"{selected_num_col} - historgram")
    hist_values = np.histogram(data_handler.data[selected_num_col])
    hist_values = pd.DataFrame(hist_values).T
    hist_values = hist_values.rename(columns={0:"count", 1: f"range of {selected_num_col.lower()}"})
    st.bar_chart(data=hist_values, y="count")
          
# show statistics for columns     
col_info = {}

col_info["Number of Unique Values"] = len(data_handler.data[selected_num_col].unique())
col_info["Number of Rows with Missing Values"] = data_handler.data[selected_num_col].isnull().sum()
col_info["Number of Rows with 0"] = data_handler.data[selected_num_col].eq(0).sum()
if selected_num_col in data_handler.data_info["numerical_columns"]:
    col_info["Number of Rows with Negative Values"] = data_handler.data[selected_num_col].lt(0).sum()
    col_info["Average Value"] = data_handler.data[selected_num_col].mean()
    col_info["Standard Deviation Value"] = data_handler.data[selected_num_col].std()
    col_info["Minimum Value"] = data_handler.data[selected_num_col].min()
    col_info["Maximum Value"] = data_handler.data[selected_num_col].max()
    col_info["Median Value"] = data_handler.data[selected_num_col].median()
 
info_df = pd.DataFrame(list(col_info.items()), columns=['Description', 'Value'])
st.dataframe(info_df) # display dataframe as a markdown table





    
    