import streamlit as st


st.title('Image object detection app')

st.sidebar.markdown("# Adjust model parameters")
selected_model = st.sidebar.selectbox(
    'Select an object detection model',
    ('tensorflow1', 'tensorflow2')
)
selected_min_score = st.sidebar.slider(
    'Select minimum confidence score',
    float(0), float(1), 0.2, 0.05
)
filtered_predictions = st.sidebar.multiselect(
    'Filter predicted objects',
    ('dupa1', 'dupa2', 'dupa3'),
    ('dupa2')
)

uploaded_images = st.file_uploader(
    'Upload images for object detection',
    ['jpg', 'png', 'bmp']
)
st.button('Detect objects')

# if button is clicked:
# get api response with predictions...
container_with_output = st.container()
container_with_output.markdown("### Predicted images:")
# print each image with predicted boxes
container_with_output.write('images with predictions')
# table with prediction results
container_with_output.dataframe()
# expand to see json response
with container_with_output.expander('Show JSON output'):
    st.write('dupa')
