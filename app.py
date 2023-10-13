import streamlit as st
from PIL import Image
import tempfile
import asyncio
from helpercode import get_blip2_caption, extract_text_from_image, main_generater, normal_main_generater

# Streamlit title
st.title('InstaCaption Magic: Spark Creativity, One Post at a Time! ✨')

# Sidebar for OpenAI key input
with st.sidebar:
    password = st.text_input("OPEN_AI_KEY", type="password")
    submit = st.button('Submit')

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image-uploader")

post_type = st.radio(
    "Select the type of post you want to create:",
    ["Personal :selfie:", "Brand Collaboration :handshake:"],
    captions=["For sharing your personal moments.", "For promoting a brand."])



if uploaded_file is not None and password is not None and post_type is not  None and st.button("submit"):
    image = Image.open(uploaded_file).convert("RGB")  # Open uploaded file with PIL

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
        image.save(temp_image.name)  # Save PIL image to temporary file

    col1, col2 = st.columns(2)

    with col1:
        st.write("Displaying the image:")
        st.image(image, caption="Uploaded Image", use_column_width=True)  # Display PIL image directly

    with col2:
        st.write("Generated Caption")
        with st.spinner('Loading...'):
            # Extract text from the image and generate captions
            if post_type == 'Personal :selfie:':
                cap = asyncio.run(get_blip2_caption(image))  # Pass PIL image directly
                main_caption = asyncio.run(normal_main_generater(cap, password))
                st.write(main_caption)
            else:
                text = asyncio.run(extract_text_from_image(temp_image.name))
                cap = asyncio.run(get_blip2_caption(image))  # Pass PIL image directly
                main_caption = asyncio.run(main_generater(cap, text, password))
                st.write(main_caption)
