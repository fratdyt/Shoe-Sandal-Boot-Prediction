import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image

# Load Model
model = tf.keras.models.load_model('model.h5')

# Title Page
st.title('Shoes, Sandals, And Boots Prediction')

# Upload Image
upload_file = st.file_uploader('Upload Image:', type=['jpg', 'png', 'jpeg'])

if upload_file is not None:
    img = Image.open(upload_file).convert('L')  # convert('L'): change to grayscale
    
    st.image(img, caption='Image Uploaded', use_container_width=True)

    # Prepro Img (samakan ukuran dgn model kita)
    img = img.resize((136, 102))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(a=img_array, axis=0) / 255.0

    # prediksi
    class_labels = {
        0: 'Boot',
        1: 'Sandal',
        2: 'Shoe'
    }

    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    class_name = class_labels.get(pred_class, "Unknown")

    st.write(f'Predict Result: {class_name}')
    st.write(f'Acc Predict: {pred[0][pred_class] * 100:.2f}%')