import streamlit as st
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.datasets import cifar10
from model_utils import create_model, class_names
from PIL import Image

# Add this at the top of your file after imports
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    /* Modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Container styling */
    .main {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
    }

    /* Enhanced button styling */
    .stButton button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        transform: translateY(0);
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #45a049, #4CAF50);
    }

    .stButton button:active {
        transform: translateY(1px);
    }

    /* Headings */
    h1 {
        color: white;
        text-align: center;
        padding-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    h3 {
        color: white;
        margin-top: 2rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }

    /* Alerts */
    .stAlert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        backdrop-filter: blur(5px);
    }

    /* Images */
    .element-container img {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .element-container img:hover {
        transform: scale(1.08);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }

    /* Markdown text */
    .markdown-text-container {
        color: white;
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 8px;
        backdrop-filter: blur(5px);
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = 'static/saved_model/cifar10_model.keras'

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'x_test' not in st.session_state:
    st.session_state.x_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'random_indices' not in st.session_state:
    st.session_state.random_indices = None

def load_saved_model():
    try:
        st.session_state.model = tf.keras.models.load_model(MODEL_PATH)
        return True
    except:
        return False

def train_model():
    try:
        # Create necessary directories
        model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)

        # Load and preprocess CIFAR-10 dataset
        with st.spinner('Loading CIFAR-10 dataset...'):
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            st.session_state.x_test = x_test

            y_train = tf.keras.utils.to_categorical(y_train, 10)
            y_test = tf.keras.utils.to_categorical(y_test, 10)

        # Create and train model
        with st.spinner('Training model...'):
            st.session_state.model = create_model()
            st.session_state.model.fit(x_train, y_train, epochs=1, batch_size=128, validation_data=(x_test, y_test))
            st.session_state.model.save(MODEL_PATH)
            st.success('Model trained and saved successfully!')

    except Exception as e:
        st.error(f"Error during training: {str(e)}")

def main():
    st.markdown("<h1>🖼️ CIFAR-10 Image Classifier</h1>", unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Model Status")
        model_exists = os.path.exists(MODEL_PATH)
        if model_exists:
            st.success('✅ Model loaded and ready')
            if st.session_state.model is None:
                load_saved_model()
        else:
            st.error('❌ No model found')
        
        if st.button('🚀 Train Model'):
            train_model()
    
    with col2:
        st.markdown("### How it works")
        st.markdown("""
        This app uses a Convolutional Neural Network (CNN) to classify images into 10 categories:
        - ✈️ Airplane
        - 🚗 Automobile
        - 🐦 Bird
        - 🐱 Cat
        - 🦌 Deer
        - 🐕 Dog
        - 🐸 Frog
        - 🐎 Horse
        - 🚢 Ship
        - 🚛 Truck
        """)

    # Load test data if not already loaded
    if st.session_state.x_test is None:
        _, (x_test, _) = cifar10.load_data()
        st.session_state.x_test = x_test.astype('float32') / 255.0

    # Generate random indices only once when app starts or they don't exist
    if st.session_state.random_indices is None:
        num_images = 10
        st.session_state.random_indices = np.random.randint(0, len(st.session_state.x_test), size=num_images)

    st.markdown("---")
    st.markdown("### 🎯 Test the Model")
    st.markdown("Click on any image below to get a prediction")
    
    # Display random images using stored indices
    num_images = 10
    
    # Create columns for images
    cols = st.columns(5)
    for i, idx in enumerate(st.session_state.random_indices):
        col_idx = i % 5
        img = (st.session_state.x_test[idx] * 255).astype(np.uint8)
        
        # Display image and create button
        with cols[col_idx]:
            st.image(img, width=64)
            if st.button(f'Predict {i+1}', key=f'pred_{i}'):
                if st.session_state.model is not None:
                    img_processed = np.expand_dims(st.session_state.x_test[idx], axis=0)
                    pred = st.session_state.model.predict(img_processed)
                    pred_class = class_names[np.argmax(pred)]
                    st.success(f'Prediction: {pred_class}')
                else:
                    st.error('Please train the model first!')

if __name__ == '__main__':
    main()