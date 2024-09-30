import streamlit as st

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
    options = ["Feature 1", "Feature 2", "Feature 3"]
    selected_option = st.selectbox("Select a feature to display:", options)  # Changed to single select
    
    # Mapping of options to image paths and captions
    feature_mapping = {
        "Feature 1": {"path": "images/electronics-11-00002-g001.png", "caption": "Flowchart of the Data Generation Process  "},
        "Feature 2": {"path": "images/feature2.png", "caption": "Feature 2"},
        "Feature 3": {"path": "images/feature3.png", "caption": "Feature 3"}
    }
    
    if selected_option:
        st.image(feature_mapping[selected_option]["path"], caption=feature_mapping[selected_option]["caption"])
    else:
        st.info("Select a feature to display the corresponding image.")

if __name__ == "__main__":
    main()