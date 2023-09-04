import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image,ImageChops
import streamlit_drawable_canvas
from streamlit_drawable_canvas import st_canvas
import cv2


def load_model():
    #model_path='/Users/yassir2/code/Yassirbenj/amazigh_text/models/amazighmodel3.h5'
    model=tf.keras.models.load_model("interface/amazighmodel2.h5")
    return model

def predict(model,image):
    yhat=model.predict(np.expand_dims(image/255,0))
    labels=['ya','YAB','yach','yad','yadd','yae','yaf','yag',
            'yagh','yagw','yah','yahh','yaj','yak','yakw','yal',
            'yam','yan','yaq','yar','yarr','yas','yass','yat',
            'yatt','yaw','yax','yay','yaz','yazz','yey','yi','yu']
    proba=np.max(yhat)*100
    y = "{:.2f}".format(proba)
    result=labels[np.argmax(yhat)]
    return result,y

def trim(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    box = diff.getbbox()
    if box:
        image=image.crop(box)
    return image

def preprocess(image):
    results=[]
    image_rescale=image[:,:,3]
    #st.image(image[:,:,0])
    #st.image(image[:,:,1])
    #st.image(image[:,:,2])
    st.image(image[:,:,3])
    st.text(np.unique(image_rescale,return_counts=True))
    #gray_image = cv2.cvtColor(image_rescale, cv2.COLOR_RGB2GRAY)
    #st.image(gray_image)
    thresh = cv2.threshold(image_rescale, 128, 255, cv2.THRESH_BINARY)[1]
    st.image(thresh)
    st.text(np.unique(thresh,return_counts=True))
    contours,hierarchy=cv2.findContours(image_rescale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        item = image[y:y+h, x:x+w]
        img = Image.fromarray(item)
        new_image=img.resize((64,64))
        img_array = np.array(new_image)
        img_array=img_array[:,:,1]
        #img_array=np.reshape(img_array,(64,64,1))
        img_tensor=tf.convert_to_tensor(img_array)
        results.append(img_tensor)
    return results

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw"))

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)


realtime_update = st.sidebar.checkbox("Update in realtime", True)

with st.form("input_form",clear_on_submit=True):
    st.write("<h3>Upload your image for the magic ✨</h3>", unsafe_allow_html=True)
    canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    #stroke_color=stroke_color,
                    #background_color=bg_color,
                    #background_image=Image.open(bg_image) if bg_image else None,
                    update_streamlit=realtime_update,
                    height=150,
                    drawing_mode=drawing_mode,
                    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                    key="canvas",
                    )
    input_img=canvas_result.image_data

    #input_img = st.file_uploader('character image',type=['png', 'jpg','jpeg'])
    if st.form_submit_button("Predict"):
        if input_img is not None:
            st.text(input_img.shape)
            result=[]
            result_proba=1
            imgs=preprocess(input_img)
            loaded_model = load_model()
            for img in imgs:
                st.image(img.numpy())
                prediction = predict(loaded_model,img)
                letter=prediction[0]
                proba= prediction[1]
                result.append(letter)
                result_proba*=proba
            st.write(f"<h3>The prediction is: {result} with probability of {result_proba}% </h3>", unsafe_allow_html=True)
