import streamlit as st
import plotly.express as px
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save_model
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as clf_save_model
import pandas as pd
import os

# Cache the data loading and setup operations
@st.cache_resource
def load_data(file):
    return pd.read_csv(file, index_col=None)

@st.cache_resource
def run_regression_setup(df, target):
    reg_setup(df, target=target, verbose=False)
    return reg_pull()

@st.cache_resource
def run_classification_setup(df, target):
    clf_setup(df, target=target, verbose=False)
    return clf_pull()

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://img.pikbest.com/wp/202347/ai-brain-blurred-wallpaper-with-3d-rendered-blue-polygons-a-hologram-of-the-representing-and-machine-learning-concepts_9760017.jpg!w700wp")
    st.title("AutoMachina")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Regression", "Classification", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = load_data(file)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")

    st.header('Data Summary')
    st.write(df.describe())

    st.header('Missing Values')
    st.write(df.isnull().sum())

    st.header('Correlation Matrix')
    corr_matrix = df.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr)

    st.header('Distribution of Numerical Features')
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_features:
        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
        st.plotly_chart(fig)

    st.header('Distribution of Categorical Features')
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
        st.plotly_chart(fig)

    st.header('Scatter Plots')
    chosen_target = st.selectbox('Choose the Target Column for Scatter Plots', numerical_features)
    for col in numerical_features:
        if col != chosen_target:
            fig = px.scatter(df, x=col, y=chosen_target, title=f'{col} vs {chosen_target}')
            st.plotly_chart(fig)

if choice == "Regression":
    st.title("Regression Modelling")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Regression Modelling'):
        setup_df = run_regression_setup(df, chosen_target)
        st.dataframe(setup_df)
        best_model = reg_compare()
        compare_df = reg_pull()
        st.dataframe(compare_df)

        # Adding percentage columns to the comparison dataframe
        compare_df['Percentage'] = (compare_df['R2'] / compare_df['R2'].sum()) * 100
        st.dataframe(compare_df)

        # Plotting the performance of different models
        fig = px.bar(compare_df, x=compare_df.index, y='R2', title='Model Comparison')
        st.plotly_chart(fig)
        
        reg_save_model(best_model, 'best_model_regression')

if choice == "Classification":
    st.title("Classification Modelling")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Classification Modelling'):
        setup_df = run_classification_setup(df, chosen_target)
        st.dataframe(setup_df)
        best_model = clf_compare()
        compare_df = clf_pull()
        st.dataframe(compare_df)

        # Adding percentage columns to the comparison dataframe
        compare_df['Accuracy'] = [x*100 for x in compare_df['Accuracy']]
        st.dataframe(compare_df)

        # Plotting the performance of different models
        fig = px.bar(compare_df, x=compare_df.index, y='Accuracy', title='Model Comparison')
        st.plotly_chart(fig)
        
        clf_save_model(best_model, 'best_model_classification')

if choice == "Download":
    st.title("Download Your Model")
    model_type = st.selectbox("Choose Model Type", ["Regression", "Classification"])
    model_file = f'best_model_{model_type.lower()}.pkl'
    with open(model_file, 'rb') as f:
        st.download_button('Download Model', f, file_name=model_file)
