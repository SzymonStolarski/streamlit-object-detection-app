import base64
import io
import random
import requests

import streamlit as st

from src.api_configuration import API_PATH
from src.callbacks import detect_objects_callback

available_models_list = requests.get(
    f"{API_PATH}/available_models").json()['available_models']
available_labels_list = requests.get(
    f"{API_PATH}/available_labels").json()['available_labels']


@st.cache
def generate_rand_number():
    return random.randint(0, len(available_labels_list)-1)


st.title('Image object detection app')

st.sidebar.markdown("# Adjust model parameters")
selected_model = st.sidebar.selectbox(
    'Select an object detection model',
    available_models_list
)
selected_min_score = st.sidebar.slider(
    'Select minimum confidence score',
    float(0), float(1), 0.2, 0.05
)
filtered_predictions = st.sidebar.multiselect(
    'Filter predicted objects',
    sorted(available_labels_list),
    available_labels_list[generate_rand_number()]
)

uploaded_images = st.file_uploader(
    'Upload images for object detection',
    ['jpg', 'png', 'bmp'], accept_multiple_files=True
)
detect_objects_button = st.button('Detect objects')

# if button is clicked:
if detect_objects_button:
    # Get prediction response from api
    prediction_response = detect_objects_callback(uploaded_images,
                                                  selected_min_score,
                                                  selected_model,
                                                  filtered_predictions)
    container_with_output = st.container()
    container_with_output.markdown("### Predicted images:")
    # print each image with predicted boxes
    for key in prediction_response:
        col1, col2 = container_with_output.columns(2)
        img_base_64 = prediction_response[key]['img']
        base64_bytes = img_base_64.encode("utf-8")
        base64_bytes = base64.b64decode(base64_bytes)
        bytes_object = io.BytesIO(base64_bytes)
        col1.image(bytes_object, channels="RGB")

        df_output_data = {
                          'label': list(
                              prediction_response[key]['labels'].values()),
                          'score': list(
                              prediction_response[key]['scores'].values())}
        col2.dataframe(df_output_data)
    # expand to see json response
    with container_with_output.expander('Show JSON output'):
        st.json(prediction_response)
