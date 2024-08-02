import streamlit as st
import plotly.express as px
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save_model
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as clf_save_model
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Cache the data loading and setup operations
@st.cache_resource
def load_data(file):
    return pd.read_csv(file, index_col=None)

@st.cache_resource
def preprocess_data(df):
    # Handling missing values for numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        imputer = SimpleImputer(strategy='mean')
        df[numeric_df.columns] = imputer.fit_transform(numeric_df)

    # Standardizing the data
    scaler = StandardScaler()
    if not numeric_df.empty:
        df[numeric_df.columns] = scaler.fit_transform(df[numeric_df.columns])

    # Handling missing values for categorical columns
    cat_df = df.select_dtypes(include=['object'])
    if not cat_df.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_df.columns] = cat_imputer.fit_transform(cat_df)
    
    return df

@st.cache_resource
def run_regression_setup(df, target):
    reg_setup(df, target=target, preprocess=False, verbose=False)
    return reg_pull()

@st.cache_resource
def run_classification_setup(df, target):
    clf_setup(df, target=target, preprocess=False, verbose=False)
    return clf_pull()

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
    df = preprocess_data(df)

with st.sidebar:
    st.image("https://img.pikbest.com/wp/202347/ai-brain-blurred-wallpaper-with-3d-rendered-blue-polygons-a-hologram-of-the-representing-and-machine-learning-concepts_9760017.jpg!w700wp")
    st.title("AutoMachina")
    choice = st.radio("Navigation", ["Home", "Upload", "Profiling", "Regression", "Classification", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Home":
    st.title("Welcome to AutoMachina")
    st.write("""
    AutoMachina is a powerful tool for data exploration, regression, and classification tasks. With this app, you can:
    - Upload your dataset
    - Perform exploratory data analysis
    - Run regression and classification models
    - Compare model performances
    - Download the best models for your use
    
    Use the sidebar to navigate through the steps and get started with your data analysis journey!
    """)

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = load_data(file)
        df = preprocess_data(df)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        st.success("Dataset uploaded and preprocessed successfully! Now, proceed to the Profiling section.")

if choice == "Profiling":
    if 'df' not in locals():
        st.warning("Please upload a dataset first.")
    else:
        st.title("Exploratory Data Analysis")
        st.header('Data Summary')
        st.write(df.describe())
        st.header('Missing Values')
        st.write(df.isnull().sum())
        st.header('Correlation Matrix')
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr_matrix = numeric_df.corr()
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
        st.success("Profiling completed! Proceed to Regression or Classification.")

if choice == "Regression":
    if 'df' not in locals():
        st.warning("Please upload a dataset first.")
    else:
        st.title("Regression Modelling")
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Regression Modelling'):
            setup_df = run_regression_setup(df, chosen_target)
            st.dataframe(setup_df)
            best_model = reg_compare()
            compare_df = reg_pull()
            
            # Converting R2 values to percentages
            compare_df['R2'] = compare_df['R2'] * 100
            compare_df['Percentage'] = compare_df['R2'] / compare_df['R2'].sum() * 100
            
            st.dataframe(compare_df)
            
            fig = px.bar(compare_df, x=compare_df.index, y='R2', title='Model Comparison')
            st.plotly_chart(fig)
            
            reg_save_model(best_model, 'best_model_regression')
            st.success("Regression modelling completed! Proceed to Download to get your best model.")

if choice == "Classification":
    if 'df' not in locals():
        st.warning("Please upload a dataset first.")
    else:
        st.title("Classification Modelling")
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Classification Modelling'):
            setup_df = run_classification_setup(df, chosen_target)
            st.dataframe(setup_df)
            best_model = clf_compare()
            compare_df = clf_pull()
            
            # Converting Accuracy values to percentages
            compare_df['Accuracy'] = compare_df['Accuracy'] * 100
            
            st.dataframe(compare_df)
            
            fig = px.bar(compare_df, x=compare_df.index, y='Accuracy', title='Model Comparison')
            st.plotly_chart(fig)
            
            clf_save_model(best_model, 'best_model_classification')
            st.success("Classification modelling completed! Proceed to Download to get your best model.")


if choice == "Download":
    st.title("Download Your Model")
    model_type = st.selectbox("Choose Model Type", ["Regression", "Classification"])
    model_file = f'best_model_{model_type.lower()}.pkl'
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            st.download_button('Download Model', f, file_name=model_file)
        st.success(f"{model_type} model is ready for download!")
    else:
        st.error(f"No {model_type} model available for download. Please run the {model_type} modelling first.")
