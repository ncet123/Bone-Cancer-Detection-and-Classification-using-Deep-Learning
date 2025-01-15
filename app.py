import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
hide_st_style = """
<style>
MainMenu {visibility:hidden;}
footer{visibility: hidden;}
.st-emotion-cache-1wbqy5l.e17vllj40 {visibility:hidden;}
</style>
"""

model = tf.keras.models.load_model('BONE_CANCER_MICRO_GPT_NEW.h5')
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title("Bone Cancer Detection And Classification Using Deep Learning")

uploaded_image = st.file_uploader("Upload an image to detect bone cancer", type=["jpg", "jpeg", "png"])
try:
    if uploaded_image is not None:
        imagee = Image.open(uploaded_image)
        st.image(imagee, caption="Uploaded Image", width=400)
        imagee = imagee.resize((256, 256))
        img_array = np.array(imagee) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        if prediction[0][1] > 0.80:
            st.success("NORMAL BONE")   
        elif prediction[0][0] > 0.85:
            st.success("CANCER BONE")
        else:
            st.error("UNABLE TO DETECT")
    else:
        st.info("Please upload an image")
except Exception as e:
    st.error("An Unexcepted Error Occured")
