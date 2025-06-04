import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Try to load the trained model
try:
    model = load_model('skinmodel.h5')
    model_loaded = True
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    model_loaded = False

# Define the label map
label_map = {0: 'Psoriasis', 1: 'Seborrheic Dermatitis', 2: 'Atopic Dermatitis'}

# Define image size
IMG_SIZE = 150

# CSS to inject
page_bg_color = """
<style>
body {
    background-color: #ADD8E6; /* Blue background color */
    color: #333333; /* Text color */
}
.stButton>button {
    background-color: #3498db; /* Button background color */
    color: white; /* Button text color */
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    border: 2px solid #2980b9;
}
.stButton>button:hover {
    background-color: #2980b9; /* Hover color for button */
}
.st-image {
    border-radius: 10px; /* Rounded corners for image */
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); /* Box shadow for image */
}
</style>
"""

st.markdown(page_bg_color, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Detection", "📄 About"])

# Main content based on selected page
if page == "🏠 Home":
    st.title("Home Page")
    st.write("""
    ## Welcome to the Skin Disease Detection System!

    This web application is designed to help you detect skin diseases using a machine learning model. The model has been trained to identify three types of skin conditions:
    - Psoriasis
    - Seborrheic Dermatitis
    - Atopic Dermatitis

    ### How to Use the Web App:
    1. Navigate to the **🔍 Detection** page using the sidebar.
    2. Upload an image of the affected skin area.
    3. Click the **Detect** button to get the model's prediction.
    4. The app will display the predicted condition and the confidence level of the prediction.

    We hope this tool helps you in identifying skin conditions accurately and promptly. Please consult a healthcare professional for a comprehensive diagnosis and treatment plan.
    """)

elif page == "🔍 Detection":
    st.title("Detection Page")
    st.write("""
    ## Skin Disease Detection
    Upload an image and the model will predict the type of skin disease.
    """)
    
    if not model_loaded:
        st.error("Model is not loaded. Please check the model file.")
    else:
        # Image upload functionality
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                # Read the uploaded image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is not None:
                    # Create columns for layout
                    col1, col2 = st.columns([2, 1])

                    # Display the uploaded image in the first column
                    with col1:
                        st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

                    # Process and prepare the image for prediction
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
                    img_resized = img_resized.astype('float32') / 255.0
                    img_resized = np.expand_dims(img_resized, axis=-1)
                    img_resized = np.expand_dims(img_resized, axis=0)

                    # Display the button and prediction results in the second column
                    with col2:
                        if st.button('Detect'):
                            # Make predictions
                            prediction = model.predict(img_resized)
                            predicted_class = np.argmax(prediction, axis=1)[0]
                            predicted_label = label_map[predicted_class]

                            # Display the condition and confidence
                            st.write(f"**Condition:** {predicted_label}")
                            st.write(f"**Confidence:** {prediction[0][predicted_class] * 100:.2f}%")
                else:
                    st.error("Error in reading the image. Please upload a valid image file.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.write("Please upload an image file to proceed.")

elif page == "📄 About":
    st.title("About Page")
    st.write("""
    ## About the Skin Disease Detection System
    
    This web application utilizes a machine learning model to detect skin diseases from uploaded images.
    
    ### Project Overview
    
    - **Purpose**: The primary goal of this project is to assist healthcare professionals at Edo state University Teaching Hospital in the early detection and identification of common skin diseases.
    
    - **Model Development**: The machine learning model used in this application has been trained to identify three specific types of skin conditions:
      - Psoriasis
      - Seborrheic Dermatitis
      - Atopic Dermatitis
    
    ### Contact
    
    For inquiries about this project, please contact:
    - **Praise Jefferson**
    - Email: [ipraise1a2a@gmail.com](mailto:ipraise1a2a@gmail.com)
    - Phone: +234816397906
    
    ### Note
    
    This tool is intended to assist and support healthcare providers. It does not replace professional medical advice. Always consult with a healthcare provider for a comprehensive diagnosis and treatment plan.
    """)

