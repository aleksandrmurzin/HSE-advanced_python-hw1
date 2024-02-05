import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


def plot_correlation_with_target(df, save_path=None):
    """
    Plots the correlation of each variable in the dataframe with the 'demand' column.

    Args:
    - df (pd.DataFrame): DataFrame containing the data, including a 'demand' column.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - None (Displays the plot on a Jupyter window)
    """

    # Compute correlations between all variables and 'demand'
    correlations = df.corr()["TARGET"].drop("TARGET").sort_values()

    # Generate a color palette from red to green
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    # Set Seaborn style
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )  # Light grey background and thicker grid lines

    # Create bar plot
    fig = plt.figure(figsize=(12, 8))
    bars = plt.barh(correlations.index, correlations.values, color=color_mapped)

    # Set labels and title with increased font size
    plt.title("Correlation with TARGET", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # prevent matplotlib from displaying the chart every time we call this function
    plt.close(fig)

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    return fig

def plot_correlation_matrix(df, save_path=None):
    """
    Plots the correlation matrix.

    Args:
    - df (pd.DataFrame): DataFrame containing the data, including a 'demand' column.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - None (Displays the plot on a Jupyter window)
    """

    # Compute correlations between all variables and 'demand'
    correlations = df.corr()

    # Set Seaborn style
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )  # Light grey background and thicker grid lines

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlations, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))
    color_mapped = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlations, mask=mask, cmap=color_mapped, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Set labels and title with increased font size
    plt.title("Correlation", fontsize=18)
    plt.xlabel("Variable", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # prevent matplotlib from displaying the chart every time we call this function
    plt.close(fig)

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    return fig


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


    def get_stats(self, column):
        stats = self._update_stats(column)
        return stats        

    def _update_stats(self, column, stats={}):
        stats["Number of Unique Values"] = len(self._data[column].unique())
        stats["Number of Rows with Missing Values"] = self._data[column].isnull().sum()
        stats["Number of Rows with 0"] = self._data[column].eq(0).sum()
        if column in self._data_info["numerical_columns"]:
            stats["Number of Rows with Negative Values"] = self._data[column].lt(0).sum()
            stats["Average Value"] = self._data[column].mean()
            stats["Standard Deviation Value"] = self._data[column].std()
            stats["Minimum Value"] = self._data[column].min()
            stats["Maximum Value"] = self._data[column].max()
            stats["Median Value"] = self._data[column].median()
        return stats

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
st.divider()
msg = "Which column do you want to explore?"
selected_num_col = st.selectbox(msg, sorted(data_handler.data_info["columns"]))

# show histograms
if selected_num_col in data_handler.data_info["numerical_columns"] and selected_num_col != "TARGET":
    st.subheader(f"{selected_num_col} - historgram")
    hist_values = np.histogram(data_handler.data[selected_num_col])
    hist_values = pd.DataFrame(hist_values).T
    hist_values = hist_values.rename(columns={0:"count", 1: f"range of {selected_num_col.lower()}"})
    st.bar_chart(data=hist_values, y="count")
    
    st.subheader(f"{selected_num_col} vs TARGET")    
    st.scatter_chart(
    data=data_handler.data[[selected_num_col, "TARGET"]].dropna(),
    x=selected_num_col,
    y="TARGET")

# show statistics
stats = data_handler.get_stats(column=selected_num_col)         
info_df = pd.DataFrame(list(stats.items()), columns=['Description', 'Value'])
st.subheader(f"{selected_num_col} - stats")
st.dataframe(info_df) # display dataframe as a markdown table


# Plot correlation
st.divider()
st.subheader('General Correlation plot')
fig = plot_correlation_with_target(data_handler.data[data_handler.data_info["numerical_columns"]])
st.write(fig)


# Plot correlation matrix
st.subheader('Correlation matrix')
fig = plot_correlation_matrix(data_handler.data[data_handler.data_info["numerical_columns"]])
st.write(fig)

    
    