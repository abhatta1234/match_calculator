import streamlit as st
import skimage_blur_effect
import skimage.io
import skimage.measure
from skimage import color,measure
import skimage_blur_effect
import numpy
import argparse

# Note: make sure skimage_blur_effect.py is in same directory level
# It overrides the original skimage_blur_effect

def estimate_blur_perceptual(image: numpy.array):
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return skimage_blur_effect.blur_effect(image, h_size=11)

def return_blur_estimate(imagepath):
    loaded_skimage = skimage.io.imread(imagepath)
    return estimate_blur_perceptual(loaded_skimage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Return the blur estimation for a single image")
    parser.add_argument("--source", "-s", help="path to the image")
    args = parser.parse_args()

    estimated_blur_value = return_blur_estimate(args.source)

    print(f"The estimated Blur Value of Image at {args.source} = {estimated_blur_value}"                                                                                  

header = st.container()
col1,col2 = st.columns(2)
predictions= st.container()


with header:
    st.markdown("<h1 style='text-align: center; color: skyblue;'>Image Match Score calculator</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: skyblue;'>Model: ArcFace trained on Glint360K</h3>",
                unsafe_allow_html=True)

with col1:
   st.markdown("<h3 style='text-align: center; color: grey;'>Image 1</h3>", unsafe_allow_html=True)
   image1 = st.file_uploader("Select Image 1 here",label_visibility="hidden")
   #st.write(np.load(image1).shape)
   if image1:
       pil_image1 = Image.open(image1)
       st.image(pil_image1)
       np_image1 = np.array(pil_image1)
       #val1= np.expand_dims(extract_features(np_image1),axis=0)

with col2:
   st.markdown("<h3 style='text-align: center; color: grey;'>Image 2</h3>", unsafe_allow_html=True)
   image2 = st.file_uploader("Select Image 2 here",label_visibility="hidden")
   if image2:
       pil_image2 = Image.open(image2)
       st.image(pil_image2)
       np_image2 = np.array(pil_image2)
       #val2 = np.expand_dims(extract_features(np_image2),axis=0)

with predictions:
    if image1 and image2:
        match_score = np.random.randint(10,size=2)#cosine_score(val1,val2).flatten()
        #variable_output = st.text_input("Match Score", value=match_score)
        html_str = f"""
        <style>
        p.a {{
          font: bold 30px Courier;
          text-align:justify;
        }}
        </style>
        <p class="a">Match Score = {match_score}</p>
        """

        st.markdown(html_str, unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: red;'>Need to Select both images for Match Score!</h3>", unsafe_allow_html=True)
