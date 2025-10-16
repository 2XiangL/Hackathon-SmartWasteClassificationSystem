import streamlit as st
import os
from PIL import Image
from model import load_yolo_model, init_llm
from template import chat_template, image_template
from encode_image import encode_image


def init_st():

    st.set_page_config(page_title="WasteClassificationByYOLOv8")

    st.sidebar.title("API settings")

    st.title("This is a waste classification application by YOLOv8 and Qwen")

    st.markdown("You can upload an image for waste classification and recognition")


if __name__ == "__main__":

    model = load_yolo_model()

    init_st()

    api_key = st.sidebar.text_input("Please input your Qwen API Key: ", type="password")

    upload_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if upload_file is not None:
        image = Image.open(upload_file)
        filename = upload_file.name
        ext = os.path.splitext(filename)[1].lstrip('.')
        ext = ext.lower()

        st.markdown("## YOLO detection")
        
        with st.spinner("Detecting ......"):
            res = model(image)

        result_image = res[0].plot()
        result_image = Image.fromarray(result_image[..., ::-1])


        st.image(image, caption="Origin image")

        st.image(result_image, caption="Results")

        boxes = res[0].boxes
        unique_classes = set()
        unique_classes_text = ""
        if boxes is not None and len(boxes) > 0:
            class_names = model.names
            detected_classes = [class_names[int(cls)] for cls in boxes.cls]
            unique_classes = set(detected_classes)


            for item in unique_classes:
                unique_classes_text = unique_classes_text +  "Kind: " + item + "\n"


            st.write("Detected types:：", ", ".join(detected_classes))
            st.write(f"A total of {len(detected_classes)} were detected")

            st.markdown("---")

            st.markdown("## Category analysis")
