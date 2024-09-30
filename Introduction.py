import streamlit as st
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly.express as px
import streamlit as st
import os
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata
import joblib
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def main():
    st.set_page_config(page_title="GenAI - Synthetic Data", page_icon="🧊")
    st.title("Welcome to the Data Imputation App")
    st.write("""
    ## Overview
    This application allows you to:
    - **Explore**: Get an introduction to data generation and imputation techniques.
    - **Generate Data**: Use advanced methods like GANs to perform data imputation.

    Use the sidebar to navigate between different sections of the app.
    """)

    st.header("Introduction")
    # Placeholder items for the carousel
    # Add image selection options
    st.subheader("Our Features")
    options = ["Flowchart", "GANBLR", "RealTabFormer" , "Tab-VAE","Evaluation", 'Appendix']
    selected_option = st.selectbox("Select a feature to display:", options)  # Changed to single select
    
    # Mapping of options to image paths and captions
    feature_mapping = {
        "Flowchart": {"path": "images/process-flowchart.png", "caption": "Flowchart of the Data Generation Process  "},
        "GANBLR": {"path": "images/ganblr-framework.png", "caption": "GANBLR"},
        "RealTabFormer": {"path": "images/realtabformer-framework.png", "caption": "RealTabFormer"},
        "Tab-VAE": {"path": "images/tab-vae-framework.png", "caption": "Tab-VAE"},
        "Evaluation": {"path": "images/data-comparison.png", "caption": "Evaluation"}
    }
    
    if selected_option != 'Appendix':
        st.image(feature_mapping[selected_option]["path"], caption=feature_mapping[selected_option]["caption"])
    else:
        st.write("""
        ## Appendix
        - [GANBLR](https://www.researchgate.net/publication/356159733_GANBLR_A_Tabular_Data_Generation_Model)
        - [RealTabFormer](https://arxiv.org/pdf/2302.02041)
        - [Tab-VAE](https://www.scitepress.org/Papers/2024/123024/123024.pdf)
        - [Diffusion Models](https://www.semanticscholar.org/reader/701f7b7de38436b219ceae61bb06ec406d408a5e)
        - [Language Models](https://arxiv.org/pdf/2210.06280v2)     
        """)

if __name__ == "__main__":
    main()