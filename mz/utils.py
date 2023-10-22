from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import folium
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# constants
state_geo = 'datasets/us-states.json'

########## Preprocessing ##########
onehot_encoders = {}

def convert_category_to_num_label(df: pd.DataFrame, col_name: str):
    """
        params:
            df: pandas dataframe
            col_name: the name of a column to convert

        return: a new data frame with the onehot encoding
    """
    encoder = LabelEncoder()
    store_label = encoder.fit_transform(df[col_name])
    df[col_name] = store_label

def get_new_df_with_onehot_encoding(df: pd.DataFrame, col_name: str, is_train_data: bool):
    """
        params:
            df: pandas dataframe
            col_name: the name of a column to convert
            is_train_data: whether or not it is train data

        return: a new data frame with the onehot encoding
    """
    encoder = OneHotEncoder(sparse=False)
    if is_train_data:
        cat = encoder.fit_transform(df[[col_name]])
        one_hot_features = pd.DataFrame(cat, columns=[col_name + "_" + col for col in encoder.categories_[0]])
        new_df = pd.concat([df.drop(columns=[col_name]), one_hot_features], axis=1)
        onehot_encoders[col_name] = encoder
        return new_df
    else:
        assert(onehot_encoders.get(col_name))
        encoder = onehot_encoders[col_name]
        cat = encoder.transform(df[[col_name]])
        one_hot_features = pd.DataFrame(cat, columns=[col_name + "_" + col for col in encoder.categories_[0]])
        new_df = pd.concat([df.drop(columns=[col_name]), one_hot_features], axis=1)
        return new_df
###################################



########## Statistics #############
def get_vif(df: pd.DataFrame):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    target_df = df.select_dtypes(include=numerics)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(target_df.values, i) for i in range(target_df.shape[1])]
    vif["Feature Name"] = target_df.columns
    vif = vif.sort_values("VIF Factor").reset_index(drop=True)
    return vif
###################################



########## Visualization ##########
def draw_heatmap(df: pd.DataFrame):
    """
        params
            df: a pandas dataframe to use to draw a heatmap of it
    """
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=2.2)
    sns.heatmap(df.corr(numeric_only=True), cmap='vlag', annot=True, annot_kws={"size": 24})
    plt.show()

def set_plot_labels(ax, title: Tuple[str, int]=None, xlab: Tuple[str, int]=None, ylab: Tuple[str, int]=None, legend:Tuple[List, int]=None):
    """
        params
            title: title tuple containing a string of its name and its font size
            xlab: x label tuple containing a string of its name and its font size
            ylab: y label tuple containing a string of its name and its font size
            legend: legend tuple containing a string of its name and its font size
    """
    if title:
        plt.title(title[0], fontsize=title[1])
    if xlab:
        ax.set_xlabel(xlab[0], size=xlab[1])
    if ylab:
        ax.set_ylabel(ylab[0], size=ylab[1])
    if legend:
        ax.legend(labels=legend[0], fontsize=legend[1])

def get_map_with_markers(df: pd.DataFrame, lat: str, long: str, marker_color:str = '#E75858', marker_rad:float = 0.02):
    init_map_lat = df[lat].mean()
    init_map_long = df[long].mean()

    fol_map = folium.Map(location=[init_map_lat, init_map_long], zoom_start=3)
    folium.Choropleth(geo_data=state_geo, fill_color='white').add_to(fol_map)
    for row in range(len(df)):
        cm = folium.CircleMarker([df.loc[row, lat], df.loc[row, long]],
                                 color=marker_color,
                                 radius=marker_rad)
        cm.add_to(fol_map)

    return fol_map

def draw_feature_importance_plot(feat_names: np.array, feat_importance: np.array):
    """
        Draw a bar graph of feature importance

        params
            feat_names: feature names
            feat_importance: feature importance
    """
    data = {'feature_names': feat_names, 'feature_importance': feat_importance * 100}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(20, 20))
    ax = sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    set_plot_labels(ax, ("Feature Importance", 40), ("feature importance (%)", 30), ("feature names", 30))
    plt.show()
###################################


############ Evaluation ###########
def print_eval_metrics(y_test, y_pred):
    p = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    p.plot()
    plt.show()
    print(classification_report(y_test, y_pred))
###################################