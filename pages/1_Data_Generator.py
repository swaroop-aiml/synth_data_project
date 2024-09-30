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

# Function to generate synthetic data
def generate_synthetic_data(data, selected_model, num_samples):
    
    model_mapping = {
        "RealTab": os.path.join("model", "copula_synthesizer.pkl"),
        "GANBLR": os.path.join("model", "ctgan_synthesizer.pkl"),
        "Tab-VAE": os.path.join("model", "tvae_synthesizer.pkl")
    }
    synthetic_data = pd.DataFrame()
    model_path = model_mapping[selected_model]

    if model_path and os.path.exists(model_path):
        if selected_model == 'Tab-VAE':
            synthesizer = TVAESynthesizer.load(model_path)
            synthetic_data = synthesizer.sample(num_samples)
        elif selected_model == 'GANBLR':
            synthesizer = CTGANSynthesizer.load(model_path)
            synthetic_data = synthesizer.sample(num_samples)
        elif selected_model == 'RealTab':
            synthesizer = GaussianCopulaSynthesizer.load(model_path)
            synthetic_data = synthesizer.sample(num_samples)
        else:
            synthesizer = joblib.load(model_path)
            synthetic_data = synthesizer.sample(num_samples)
    else:
        st.error(f"Model file for {selected_model} not found.")
    

    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
            # Label encode categorical columns
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Handle missing values (e.g., fill with mean for numerical columns)
    df_numeric = data.select_dtypes(include=[np.number])
    df_numeric = df_numeric.fillna(df_numeric.mean())
    data[df_numeric.columns] = df_numeric
    # Apply RandomOverSampler
    random_over_sampler = RandomOverSampler(random_state=42)
    X_resampled, _ = random_over_sampler.fit_resample(data, data.iloc[:,0])
    resampled_data = pd.DataFrame(X_resampled, columns=data.columns)
    
    # Decode the label encoded columns
    for col, le in label_encoders.items():
        try:    
            resampled_data[col] = le.inverse_transform(resampled_data[col])
            data[col] = le.inverse_transform(data[col])
        except Exception as e:
            st.warning(f"Decoding issue with column '{col}': {e}")
    
    return synthetic_data, resampled_data

# Main function
def main():
    st.header("Synthetic Data Generator")
    st.write("""
    This section allows you to:
    - **Train Model**: Train model using sample data.
    - **Generate Synthetic Data**: Trained model is used to generate synthetic data.
    - **Evaluate Data Quality**: Data Quality is evaluated using the synthetic data.
    - **Compare Distributions**: Actual data distribution is compared with synthetic data distribution.

    Use the sidebar to navigate between different sections of the app.
    """)

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

        model_options = ["All", "RealTab", "GANBLR", "Tab-VAE"]
        selected_model = st.selectbox("Select a synthesis model (leave empty to run all models)", model_options)

        num_samples = None
        if selected_model:
            num_samples = st.number_input(
                "Enter the number of synthetic samples (must be greater than the number of records)",
                min_value=50000,
                step=5000
            )

    if uploaded_file is not None:
        try:
            # Reset session state if a new file is uploaded
            if 'file_uploaded' not in st.session_state or st.session_state.file_uploaded != uploaded_file.name:
                st.session_state.file_uploaded = uploaded_file.name
                st.session_state.synthetic_data = None
                st.session_state.resampled_data = None

            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, delimiter=',')
            else:
                df = pd.read_excel(uploaded_file)

            # Identify categorical columns
            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
            st.subheader("Uploaded Data")
            st.write(df)    
            data = df       

            # Generate metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=df)

            synthetic_data = st.session_state.get('synthetic_data', None)
            resampled_data = st.session_state.get('resampled_data', None)

            if selected_model != "All":
                if num_samples is not None and num_samples > df.shape[0]:
                    synthesis_button = st.button("Generate Synthetic Data")

                    if synthesis_button:
                        with st.spinner("Generating synthetic data..."):
                            synthetic_data, resampled_data = generate_synthetic_data(data, selected_model, num_samples)
                            if synthetic_data is not None:
                                st.session_state.synthetic_data = synthetic_data
                                st.session_state.resampled_data = resampled_data
                                st.success("Synthetic data generation completed.")

                                # Evaluate quality
                                with st.spinner("Evaluating data quality..."):
                                    quality = evaluate_quality(df, synthetic_data, metadata)
                                    st.write("**Quality Evaluation:**")
                                    st.write(np.round(quality.get_score()*100,2),"%")
                else:
                    if num_samples is not None:
                        st.error("Number of samples must be greater than the number of records.")
            else:
                run_all_button = st.button("Run All Models and Select Best")

                if run_all_button:
                    with st.spinner("Generating synthetic data with all models..."):
                        model_mapping = {
                            "RealTab": os.path.join("model", "copula_synthesizer.pkl"),
                            "GANBLR": os.path.join("model", "ctgan_synthesizer.pkl"),
                            "Tab-VAE": os.path.join("model", "tvae_synthesizer.pkl")
                        }

                        all_synthetic = {}
                        quality_scores = {}

                        for model_name, model_path in model_mapping.items():
                            if os.path.exists(model_path):
                                if model_name == 'Tab-VAE':
                                    synthesizer = TVAESynthesizer.load(model_path)
                                elif model_name == 'GANBLR':
                                    synthesizer = CTGANSynthesizer.load(model_path)
                                elif model_name == 'RealTab':
                                    synthesizer = GaussianCopulaSynthesizer.load(model_path)
                                else:
                                    synthesizer = joblib.load(model_path)

                                synthetic = synthesizer.sample(num_samples)
                                synthetic_data_generated, resampled_data_generated = generate_synthetic_data(
                                    data, model_name, num_samples)
                                if synthetic_data_generated is not None:
                                    all_synthetic[model_name] = synthetic_data_generated
                                    quality = evaluate_quality(df, synthetic_data_generated, metadata)
                                    quality_scores[model_name] = quality.get_score()
                            else:
                                st.warning(f"Model file for {model_name} not found. Skipping.")

                        if quality_scores:
                            # Select the best model based on quality scores
                            best_model = max(quality_scores, key=quality_scores.get)
                            synthetic_data, resampled_data = generate_synthetic_data(
                                df, best_model, num_samples )
                            if synthetic_data is not None:
                                st.session_state.synthetic_data = synthetic_data
                                st.session_state.resampled_data = resampled_data
                                st.success(f"Synthetic data generation completed using {best_model} model.")

                                # Evaluate quality
                                with st.spinner("Evaluating data quality..."):
                                    quality = evaluate_quality(df, synthetic_data, metadata)
                                    st.write("**Quality Evaluation:**")
                                    st.write(np.round(quality.get_score()*100,2),"%")
                        else:
                            st.error("No models were successfully loaded.")

            # Distribution comparison
            st.subheader("Distribution Comparison")
            column_options = df.columns.tolist()
            dist_column = st.selectbox("Select a column to compare distributions", column_options, key="dist_compare")

            if st.session_state.synthetic_data is not None and st.session_state.resampled_data is not None:
                synthetic_data = st.session_state.synthetic_data
                resampled_data = st.session_state.resampled_data

                # Combine all datasets into one DataFrame with a 'Data_Type' column
                df_actual = df[[dist_column]].copy()
                df_actual['Data_Type'] = 'Actual'
                df_synthetic = synthetic_data[[dist_column]].copy()
                df_synthetic['Data_Type'] = 'Synthetic'
                df_resampled = resampled_data[[dist_column]].copy()
                df_resampled['Data_Type'] = 'Resampled'

                combined_df = pd.concat([df_actual, df_synthetic, df_resampled])

                if dist_column in categorical_features:
                    # Bar Chart for Categorical Data
                    fig = px.bar(
                        combined_df,
                        x=dist_column,
                        color='Data_Type',
                        barmode='group',
                        opacity=0.7,
                        color_discrete_map={
                            'Actual': 'blue',
                            'Synthetic': 'red',
                            'Resampled': 'green'
                        },
                        labels={dist_column: dist_column, 'Data_Type': 'Data Type'}
                    )
                else:
                    # Histogram for Numerical Data
                    fig = px.histogram(
                        combined_df,
                        x=dist_column,
                        color='Data_Type',
                        nbins=30,
                        opacity=0.6,
                        barmode='overlay',
                        color_discrete_map={
                            'Actual': 'blue',
                            'Synthetic': 'red',
                            'Resampled': 'green'
                        },
                        labels={dist_column: dist_column, 'Data_Type': 'Data Type'}
                    )
                    fig.update_traces(opacity=0.6)

                fig.update_layout(
                    title=f'Distribution Comparison for {dist_column}',
                    legend_title_text='Data Type',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig)

                # Export synthetic data
                with st.expander("Download Synthetic Data"):
                    st.download_button(
                        label="Download Synthetic Data as CSV",
                        data=synthetic_data.to_csv(index=False).encode('utf-8'),
                        file_name='synthetic_data.csv',
                        mime='text/csv'
                    )
            else:
                st.info("Generate synthetic data to compare distributions.")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()