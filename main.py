import streamlit as st
import numpy as np
from PIL import Image
import cv2
import numpy as np
import torch
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

@st.cache
def load_onnx_model(modelpath="r100_glint360k.onnx"):
    # Run the model on the backend
    # session = ort.InferenceSession(onnx_model_path)
    if torch.cuda.is_available():
        ort_sess = ort.InferenceSession(modelpath, None, providers=["CUDAExecutionProvider"])
    else:
        ort_sess = ort.InferenceSession(modelpath, None)
        # get the name of the first input of the model
    input_name = ort_sess.get_inputs()[0].name
    return ort_sess,input_name

#def extract_features(image):
    ort_sess,input_name = load_onnx_model()
    #some preprocessing before inferencing
    img = cv2.resize(image, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)) # passing image as channel,length,width
    img = torch.from_numpy(img).unsqueeze(0).float()#unsqueeze(0) adds the batch size of 1 to array
    img.div_(255).sub_(0.5).div_(0.5)
    outputs_ort = ort_sess.run(None, {'{}'.format(input_name): img.numpy()})
    feature = np.array(outputs_ort[0]).flatten()
    return feature

def cosine_score(feature1,feature2):
    return cosine_similarity(feature1,feature2)


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
