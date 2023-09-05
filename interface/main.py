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
    right_pad = desired_size - width - left_pad

    # Copy the original image into the center of the square canvas
    square_image[top_pad:top_pad+height, left_pad:left_pad+width] = image

    return square_image

def resize_image(image,desired_size):
    original_height, original_width, channels = image.shape

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / original_height

    # Determine which dimension (width or height) should match the desired size
    if aspect_ratio > 1.0:  # Width is larger
        new_width = desired_size[0]
        new_height = int(new_width / aspect_ratio)
    else:  # Height is larger or equal
        new_height = desired_size[1]
        new_width = int(new_height * aspect_ratio)

    # Resize the image while preserving the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a blank canvas of the desired size
    final_image = np.zeros((desired_size[1], desired_size[0], channels), dtype=np.uint8)

    # Calculate the position to place the resized image in the center of the canvas
    x_offset = (desired_size[0] - new_width) // 2
    y_offset = (desired_size[1] - new_height) // 2

    # Place the resized image on the canvas
    final_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return final_image

def preprocess(image):
    results=[]
    image_rescale=image[:,:,3]
    thresh = cv2.threshold(image_rescale, 0, 255, cv2.THRESH_BINARY)[1]
    contours,hierarchy=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_min=1000
    x_list={}
    for i,contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if y<y_min:
            y_min=y
        x_list[i]=x

    st.text(x_list)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        item = image[y_min:y+h, x:x+w]
        st.image(item)
        st.text(f"x={x}")
        desired_size=y+h-y_min
        resized_array=pad_image(item,desired_size)
        st.image(resized_array)
        resized_array=resize_image(resized_array,(64,64))
        resized_array = resized_array[:,:,3]
        img_tensor=tf.convert_to_tensor(resized_array)
        results.append(img_tensor)
    return results



with st.form("input_form",clear_on_submit=True):
    st.write("<h3>Upload your image for the magic ✨</h3>", unsafe_allow_html=True)
    canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=5,
                    update_streamlit=True,
                    height=250,
                    drawing_mode="freedraw",
                    key="canvas",
                    )
    input_img=canvas_result.image_data

    #input_img = st.file_uploader('character image',type=['png', 'jpg','jpeg'])
    if st.form_submit_button("Predict"):
        if input_img is not None:
            st.text(input_img.shape)
            result=[]
            proba_list=[]
            result_proba=100
            imgs=preprocess(input_img)
            loaded_model = load_model()
            for img in imgs:
                st.image(img.numpy())
                prediction = predict(loaded_model,img)
                letter=prediction[0]
                proba= prediction[1]
                result.append(letter)
                proba_list.append(proba)
                result_proba*=float(proba)/100
            st.write(f"<h3>The prediction is: {result} with probability of {result_proba}% </h3>", unsafe_allow_html=True)
            st.write(proba_list)
