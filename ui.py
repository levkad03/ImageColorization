import io

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from unet import UNET
from utils import load_checkpoint

st.set_page_config(page_title="Image Colorizer", page_icon=":camera:", layout="wide")


@st.cache_resource
def load_model():
    model = UNET(in_channels=1, out_channels=3)
    load_checkpoint(torch.load("unet_checkpoint.pth.tar"), model)
    model.eval()
    return model


model = load_model()


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def colorize_image(image):
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(image_tensor)

    output = output.squeeze(0).permute(1, 2, 0).numpy()
    output = (output * 255).astype(np.uint8)
    return Image.fromarray(output)


st.title("Image Colorizer")
st.write("Upload an image and click the 'Colorize' button to colorize it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Colorize"):
        colorized_image = colorize_image(image)
        st.image(colorized_image, caption="Colorized Image", use_container_width=True)

        image_bytes = io.BytesIO()
        colorized_image.save(image_bytes, format="PNG")
        img_bytes = image_bytes.getvalue()

        st.download_button(
            label="Download image",
            data=image_bytes,
            file_name="colorized_image.png",
            mime="image/png",
        )
