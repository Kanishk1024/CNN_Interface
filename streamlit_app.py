import streamlit as st
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.datasets import cifar10
from model_utils import create_model, class_names
from PIL import Image

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
    st.title('CIFAR-10 Classifier')

    # Check if model exists
    model_exists = os.path.exists(MODEL_PATH)
    if model_exists:
        st.info('A trained model is loaded and ready for predictions.')
        if st.session_state.model is None:
            load_saved_model()
    else:
        st.warning('No trained model found. Please click "Train Model" to begin.')

    # Train model button
    if st.button('Train Model'):
        train_model()

    # Load test data if not already loaded
    if st.session_state.x_test is None:
        _, (x_test, _) = cifar10.load_data()
        st.session_state.x_test = x_test.astype('float32') / 255.0

    # Generate random indices only once when app starts or they don't exist
    if st.session_state.random_indices is None:
        num_images = 20
        st.session_state.random_indices = np.random.randint(0, len(st.session_state.x_test), size=num_images)
    
    # Display random images using stored indices
    st.subheader('Click an image to predict:')
    num_images = 20
    
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