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

def pad_image(image,desired_size):
    height, width, channels = image.shape

    # Create a new blank square canvas
    square_image = np.zeros((desired_size, desired_size, channels), dtype=np.uint8)

    # Calculate the padding values to center the original image
    top_pad = (desired_size - height) // 2
    bottom_pad = desired_size - height - top_pad
    left_pad = (desired_size - width) // 2
    st.text(left_pad)
    right_pad = desired_size - width - left_pad
    st.text(right_pad)

    # Copy the original image into the center of the square canvas
    square_image[top_pad:top_pad+height, left_pad:left_pad+width] = image

    return square_image

def preprocess(image):
    results=[]
    image_rescale=image[:,:,3]
    thresh = cv2.threshold(image_rescale, 0, 255, cv2.THRESH_BINARY)[1]
    st.image(thresh)
    contours,hierarchy=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_min=1000
    y_max=0
    h_max=0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if y<y_min:
            y_min=y
        if y>y_max:
            y_max=y
        if h>h_max:
            h_max=h

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        item = image[y_min:y+h, x:x+w]
        st.image(item)
        st.text(item.shape)
        desired_size=y+h-y_min
        st.text(desired_size)
        resized_array=pad_image(item,desired_size)
        st.image(resized_array)
        st.text(resized_array.shape)
        resized_array = cv2.resize(item, (64, 64))
        st.image(resized_array)
        resized_array = resized_array[:,:,3]
        resized_array = np.expand_dims(resized_array, axis=-1)
        img_tensor=tf.convert_to_tensor(resized_array)
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
            result_proba=100
            imgs=preprocess(input_img)
            loaded_model = load_model()
            for img in imgs:
                st.image(img.numpy())
                prediction = predict(loaded_model,img)
                letter=prediction[0]
                proba= prediction[1]
                result.append(letter)
                result_proba*=float(proba)/100
            st.write(f"<h3>The prediction is: {result} with probability of {result_proba}% </h3>", unsafe_allow_html=True)
