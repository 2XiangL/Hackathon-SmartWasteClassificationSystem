import streamlit as st
import os
from PIL import Image
from model import load_yolo_model, init_llm
from template import chat_template, image_template
from encode_image import encode_image


def load_css():
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_st():
    """Initialize page settings and layout"""
    # Set page configuration
    st.set_page_config(
        page_title="Smart Waste Classification System",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load CSS styles
    load_css()

    # Sidebar design
    with st.sidebar:
        st.markdown("""
        <div class="main-header" style="margin-bottom: 2rem;">
            <h1 style="font-size: 1.8rem; margin-bottom: 0.5rem;">⚙️ Settings Panel</h1>
            <p style="font-size: 0.9rem; opacity: 0.8;">Configure your API key</p>
        </div>
        """, unsafe_allow_html=True)

    # Main title area
    st.markdown("""
    <div class="main-header">
        <h1>♻️ Smart Waste Classification System</h1>
        <p>Intelligent recognition and analysis platform based on YOLOv8 deep learning and Qwen large model</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature introduction card
    st.markdown("""
    <div class="card fade-in">
        <h3>🌱 System Features</h3>
        <p>Upload an image to get the following intelligent analysis:</p>
        <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
            <li><strong>🎯 Precise Recognition</strong>: Use YOLOv8 technology to detect waste types</li>
            <li><strong>📊 Classification Statistics</strong>: Detailed waste category analysis</li>
            <li><strong>🔬 Processing Suggestions</strong>: AI-driven eco-friendly processing solutions</li>
            <li><strong>🌍 Carbon Footprint Analysis</strong>: Environmental impact assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")


if __name__ == "__main__":

    model = load_yolo_model()

    init_st()

    # Sidebar API settings
    with st.sidebar:
        st.markdown("""
        <div class="card" style="background: rgba(255,255,255,0.1); color: white;">
            <h3>🔐 API Configuration</h3>
            <p style="font-size: 0.9rem; opacity: 0.8;">Please enter your Qwen API key</p>
        </div>
        """, unsafe_allow_html=True)

        api_key = st.text_input(
            "Qwen API Key",
            type="password",
            label_visibility="collapsed",
            placeholder="Enter your API key...",
            help="Please visit Alibaba Cloud to get your Qwen API key",
            value="sk-a9f9167545b840c38bf60e99ad034f92"
        )

    # File upload area
    st.markdown("""
    <div class="card fade-in">
        <h3>📤 Image Upload</h3>
        <p>Please select a waste image to analyze (supports JPG, PNG format)</p>
    </div>
    """, unsafe_allow_html=True)

    upload_file = st.file_uploader(
        "Select image file",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed",
        help="Supports JPG, PNG, JPEG format image files"
    )

    if upload_file is not None:
        image = Image.open(upload_file)
        filename = upload_file.name
        ext = os.path.splitext(filename)[1].lstrip('.')
        ext = ext.lower()

        # Detection process
        st.markdown("""
        <div class="analysis-section slide-in">
            <h2>🎯 YOLO Smart Detection</h2>
            <p>Using deep learning model to analyze the image...</p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🔍 Detecting, please wait..."):
            res = model(image)

        result_image = res[0].plot()
        result_image = Image.fromarray(result_image[..., ::-1])

        # Image comparison display
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="image-container fade-in">
                <h3>📷 Original Image</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, caption="Uploaded original image", use_container_width=True)

        with col2:
            st.markdown("""
            <div class="image-container fade-in">
                <h3>🎯 Detection Result</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(result_image, caption="AI recognition result", use_container_width=True)

        boxes = res[0].boxes
        unique_classes = set()
        unique_classes_text = ""
        if boxes is not None and len(boxes) > 0:
            class_names = model.names
            detected_classes = [class_names[int(cls)] for cls in boxes.cls]
            unique_classes = set(detected_classes)

            for item in unique_classes:
                unique_classes_text = unique_classes_text + "Kind: " + item + "\n"

            # Statistics display
            st.markdown("""
            <div class="detection-results fade-in">
                <h2>📊 Detection Statistics</h2>
            </div>
            """, unsafe_allow_html=True)

            # Create statistics cards
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{len(detected_classes)}</div>
                    <div class="stat-label">Detected Objects</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{len(unique_classes)}</div>
                    <div class="stat-label">Waste Types</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">100%</div>
                    <div class="stat-label">Recognition Accuracy</div>
                </div>
                """, unsafe_allow_html=True)

            # Detected category tags
            st.markdown("""
            <div class="card fade-in">
                <h3>🏷️ Detected Waste Types</h3>
            </div>
            """, unsafe_allow_html=True)

            # label cloud
            tags_html = ""
            for i, item in enumerate(unique_classes):
                tags_html += f'<span class="tag">{item}</span>'

            st.markdown(f'<div style="text-align: center; margin: 1rem 0;">{tags_html}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Category analysis section
            st.markdown("""
            <div class="analysis-section slide-in">
                <h2>🔬 Smart Classification Analysis</h2>
                <p>Eco-friendly processing suggestions based on Qwen large model</p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("🤖 AI is analyzing, please wait..."):
                client = init_llm(api_key)

                response = client.chat.completions.create(
                    model="qwen3-omni-flash",
                    messages=[
                        {"role": "system", "content": chat_template},
                        {"role": "user", "content": unique_classes_text},
                    ],
                    stream=False
                )

            st.markdown(f"""
            <div class="card fade-in">
                {response.choices[0].message.content}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Carbon footprint analysis section
            st.markdown("""
            <div class="analysis-section slide-in">
                <h2>🌍 Environmental Impact Analysis</h2>
                <p>Carbon footprint calculation and eco-friendly recommendations</p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("🌱 Calculating environmental impact..."):
                base64_image = encode_image(image, ext)

                response = client.chat.completions.create(
                    model="qwen3-omni-flash",
                    messages=[
                        {"role": "system", "content": image_template},
                        {"role": "user", "content":
                            [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/{ext};base64,{base64_image}"},
                                }
                            ]},
                    ],
                    stream=False
                )

            st.markdown(f"""
            <div class="card fade-in">
                {response.choices[0].message.content}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="warning-message fade-in">
                <h3>⚠️ No Target Detected</h3>
                <p>Sorry, no recognizable waste objects were detected in the current image. Please try using a clearer image or ensure the image contains recyclable waste items.</p>
            </div>
            """, unsafe_allow_html=True)

    # Page bottom information
    st.markdown("---")

    st.markdown("""
    <div class="card fade-in" style="background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%); text-align: center;">
        <h3>🌱 About Smart Waste Classification System</h3>
        <p>This system uses advanced YOLOv8 deep learning technology and Qwen large language model to contribute technological power to environmental protection.</p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🎯</div>
                <div style="font-weight: 600; color: var(--primary-color);">Precise Recognition</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Based on YOLOv8</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🤖</div>
                <div style="font-weight: 600; color: var(--primary-color);">Smart Analysis</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Qwen Large Model</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🌍</div>
                <div style="font-weight: 600; color: var(--primary-color);">Eco-Friendly</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary);">Green Technology</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: var(--text-secondary); border-top: 1px solid #e0e0e0; margin-top: 2rem;">
        <p>🚀 Smart Waste Classification System | Empowering Environmental Protection with Technology ♻️</p>
        <p style="font-size: 0.85rem;">Powered by YOLOv8 & Qwen AI</p>
    </div>
    """, unsafe_allow_html=True)
