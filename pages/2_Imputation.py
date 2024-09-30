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

def build_generator(input_dim, noise_dim):
    """Builds the generator model."""
    input_layer = Input(shape=(noise_dim,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(input_dim, activation='tanh')(x)
    generator = Model(inputs=input_layer, outputs=x, name='Generator')
    return generator

def build_discriminator(input_dim):
    """Builds the discriminator model."""
    input_layer = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(input_layer)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=input_layer, outputs=x, name='Discriminator')
    return discriminator

def build_gan(generator, discriminator):
    """Builds the GAN by combining generator and discriminator."""
    discriminator.trainable = False
    gan_input = Input(shape=(generator.input_shape[1],))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    gan = Model(inputs=gan_input, outputs=gan_output, name='GAN')
    return gan

def preprocess_data(df):
    """Preprocesses the data: encodes categorical variables and scales numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    transformers = []
    if len(numerical_cols) > 0:
        transformers.append(
            ('num', MinMaxScaler(), numerical_cols)
        )
    if len(categorical_cols) > 0:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        )
    
    preprocessor = ColumnTransformer(transformers=transformers)
    df_processed = preprocessor.fit_transform(df)
    feature_names = []
    if len(numerical_cols) > 0:
        feature_names.extend(numerical_cols)
    if len(categorical_cols) > 0:
        feature_names.extend(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
    
    return df_processed, feature_names, preprocessor

def inverse_transform_data(df_imputed, preprocessor, feature_names):
    """Inverse transforms the processed data to original format."""
    df_imputed = pd.DataFrame(df_imputed, columns=feature_names)
    
    numerical_cols = []
    categorical_cols = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            numerical_cols.extend(columns)
        elif name == 'cat':
            categorical_cols.extend(columns)
    
    df_original = pd.DataFrame()
    if len(numerical_cols) > 0:
        df_original[numerical_cols] = df_imputed[numerical_cols]
    if len(categorical_cols) > 0:
        cat_data = df_imputed.drop(columns=numerical_cols, errors='ignore')
        encoder = preprocessor.named_transformers_['cat']
        if hasattr(encoder, 'inverse_transform'):
            cat_original = encoder.inverse_transform(cat_data)
            df_original[categorical_cols] = cat_original
        else:
            st.warning("The encoder does not support inverse transformation.")
    
    return df_original

def train_gan(df_processed, generator, discriminator, gan, epochs=1000, batch_size=64, noise_dim=100):
    """Trains the GAN for data imputation."""
    progress_bar = st.progress(0)  # Initialize the progress bar
    status_text = st.empty()        # Placeholder for status updates

    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        idx = np.random.randint(0, df_processed.shape[0], batch_size)
        real_data = df_processed[idx]
        
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_data = generator.predict(noise)
        
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)
        
        # Update progress bar
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        
        # Update status text periodically
        if epoch % 100 == 0 or epoch == epochs - 1:
            try:
                d_loss_value = float(d_loss[0])
                d_accuracy = float(d_loss[1])
                g_loss_value = float(g_loss)
                
                status_text.text(f"Epoch {epoch + 1}/{epochs}")
            except (IndexError, TypeError, ValueError) as e:
                pass
                # st.error(f"Error during formatting losses: {e}")
                # st.write(f"d_loss: {d_loss}, g_loss: {g_loss}")
    
    progress_bar.empty()      # Remove the progress bar when done
    status_text.empty()       # Clear the status text
    return generator

def main():
    st.header("Data Generator for Missing Value Imputation Using GANs")
    
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.subheader("First 5 Rows of the Dataset")
            st.dataframe(df.head())
            
            # Visualize Missing Values
            st.subheader("Missing Values Frequency")
            missing_freq = df.isnull().sum()
            missing_freq = missing_freq[missing_freq > 0].sort_values(ascending=False)
            
            if not missing_freq.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=missing_freq.values, y=missing_freq.index, ax=ax, palette="viridis")
                ax.set_xlabel("Number of Missing Values")
                ax.set_ylabel("Features")
                ax.set_title("Missing Values per Feature")
                st.pyplot(fig)
            else:
                st.info("No missing values found in the dataset.")
                return  # No imputation needed
            
            # Preprocess the data
            st.subheader("Preprocessing Data for GAN Imputation")
            df_processed, feature_names, preprocessor = preprocess_data(df)
            st.write("Data has been preprocessed (encoded and scaled).")
            
            # Initialize GAN components
            input_dim = df_processed.shape[1]
            noise_dim = 100
            generator = build_generator(input_dim, noise_dim)
            discriminator = build_discriminator(input_dim)
            gan = build_gan(generator, discriminator)
            
            # Compile the discriminator
            discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
            
            # Compile the GAN
            gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
            
            st.write("GAN models have been initialized.")
            
            # Training parameters
            st.subheader("Training GAN for Imputation")
            epochs = st.number_input("Number of Training Epochs", min_value=100, max_value=10000, value=1000, step=100)
            batch_size = st.number_input("Batch Size", min_value=16, max_value=512, value=64, step=16)
            noise_dim_input = st.number_input("Noise Dimension", min_value=50, max_value=500, value=100, step=50)
            
            if st.button("Start Imputation"):
                with st.spinner("Training GAN. This may take a while..."):
                    generator = train_gan(df_processed, generator, discriminator, gan, epochs=int(epochs), batch_size=int(batch_size), noise_dim=int(noise_dim_input))
                st.success("GAN training completed!")
                
                # Generate imputed data
                st.subheader("Imputing Missing Values")
                noise = np.random.normal(0, 1, (df_processed.shape[0], noise_dim))
                imputed_data = generator.predict(noise)
                
                # Inverse scaling
                scaler = preprocessor.named_transformers_['num'] if 'num' in preprocessor.named_transformers_ else None
                if scaler is not None:
                    num_cols = preprocessor.transformers_[0][2]
                    imputed_data[:, :len(num_cols)] = scaler.inverse_transform(imputed_data[:, :len(num_cols)])
                
                # Inverse transform to original data
                imputed_df = inverse_transform_data(imputed_data, preprocessor, feature_names)
                
                # Combine with original missing data locations
                df_original = df.copy()
                df_original = df_original.reset_index(drop=True)
                imputed_df = imputed_df.reset_index(drop=True)
                
                # Replace missing values with imputed data
                for column in df_original.columns:
                    if df_original[column].isnull().any():
                        df_original.loc[df_original[column].isnull(), column] = imputed_df.loc[df_original[column].isnull(), column]
                
                st.success("Missing values imputed successfully!")

                st.subheader("Missing Values Frequency in Imputed Data")
                missing_freq = imputed_df.isnull().sum()
                missing_freq = missing_freq[missing_freq > 0].sort_values(ascending=False)
                
                if not missing_freq.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=missing_freq.values, y=missing_freq.index, ax=ax, palette="viridis")
                    ax.set_xlabel("Number of Missing Values")
                    ax.set_ylabel("Features")
                    ax.set_title("Missing Values per Feature")
                    st.pyplot(fig)
                else:
                    st.info("No missing values found in the dataset.")
                    

                st.subheader("Imputed Dataset (First 5 Rows)")
                st.dataframe(df_original.head())
                
                # Option to download the imputed dataset
                csv = df_original.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Imputed Data as CSV",
                    data=csv,
                    file_name='imputed_data_gan.csv',
                    mime='text/csv',
                )
                    
        except Exception as e:
            st.error(f"Error loading the file: {e}")
    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()