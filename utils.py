import streamlit as st
from PIL import Image
import io
import base64


def display_image_with_zoom(image, title="Image Preview", use_column_width=True):
    """
    Display image with zoom functionality
    """
    st.markdown(f"""
    <div class="image-container fade-in">
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)
    st.image(image, use_column_width=use_column_width)


def create_download_link(image, filename="processed_image.jpg", link_text="Download Processed Image"):
    """
    Create image download link
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}" style="background: linear-gradient(135deg, var(--primary-color), var(--primary-light)); color: white; padding: 0.75rem 1.5rem; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block; margin: 0.5rem 0; transition: all 0.3s ease;">{link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)


def display_progress_bar(percentage, label="Processing Progress"):
    """
    Display beautiful progress bar
    """
    st.markdown(f"""
    <div class="card fade-in">
        <h3>{label}</h3>
        <div style="background: #e0e0e0; border-radius: 10px; height: 20px; margin: 1rem 0; overflow: hidden;">
            <div style="background: linear-gradient(90deg, var(--primary-color), var(--primary-light)); width: {percentage}%; height: 100%; transition: width 0.3s ease; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.85rem;">
                {percentage}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_info_card(title, content, icon="ℹ️", card_type="info"):
    """
    Create information card
    """
    card_classes = {
        "info": "success-message",
        "warning": "warning-message",
        "error": "error-message"
    }

    css_class = card_classes.get(card_type, "card")

    st.markdown(f"""
    <div class="{css_class} fade-in">
        <h3>{icon} {title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def create_badge(text, color_type="success"):
    """
    Create tag badge
    """
    colors = {
        "success": "#4CAF50",
        "warning": "#FF9800",
        "error": "#F44336",
        "info": "#2196F3",
        "primary": "#2E7D32"
    }

    color = colors.get(color_type, colors["primary"])

    return f'<span style="background: {color}; color: white; padding: 0.35rem 0.75rem; border-radius: 15px; font-size: 0.85rem; font-weight: 600; margin: 0.25rem; display: inline-block;">{text}</span>'


def display_tips():
    """
    Display usage tips
    """
    tips = [
        "📸 Ensure images are clear with sufficient lighting",
        "🎯 Place waste items in the center of the image",
        "📏 Avoid images that are too small or blurry",
        "🔄 Analyze only one type of waste at a time"
    ]

    st.markdown("""
    <div class="card fade-in">
        <h3>💡 Usage Tips</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
    """, unsafe_allow_html=True)

    for tip in tips:
        st.markdown(f'<div style="text-align: center; padding: 0.75rem; background: rgba(46, 125, 50, 0.1); border-radius: 8px;">{tip}</div>', unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def create_collapsible_section(title, content, expanded=False):
    """
    Create collapsible content area
    """
    section_id = f"section_{hash(title)}"

    st.markdown(f"""
    <div class="card fade-in">
        <h3 onclick="document.getElementById('{section_id}').style.display =
            document.getElementById('{section_id}').style.display === 'none' ? 'block' : 'none'"
            style="cursor: pointer; user-select: none; display: flex; justify-content: space-between; align-items: center;">
            {title}
            <span id="{section_id}_toggle" style="font-size: 1.2rem;">{'▼' if expanded else '▶'}</span>
        </h3>
        <div id="{section_id}" style="display: {'block' if expanded else 'none'}; margin-top: 1rem;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def animate_counter(target_number, duration=2000, prefix="", suffix=""):
    """
    Create animated counter effect
    """
    st.markdown(f"""
    <div class="stat-number" id="counter" data-target="{target_number}" data-duration="{duration}" data-prefix="{prefix}" data-suffix="{suffix}">
        {prefix}0{suffix}
    </div>
    <script>
        function animateCounter() {{
            const counter = document.getElementById('counter');
            if (!counter) return;

            const target = parseInt(counter.dataset.target);
            const duration = parseInt(counter.dataset.duration);
            const prefix = counter.dataset.prefix;
            const suffix = counter.dataset.suffix;

            const increment = target / (duration / 16);
            let current = 0;

            const updateCounter = () => {{
                current += increment;
                if (current < target) {{
                    counter.textContent = prefix + Math.floor(current) + suffix;
                    requestAnimationFrame(updateCounter);
                }} else {{
                    counter.textContent = prefix + target + suffix;
                }}
            }};

            updateCounter();
        }}

        // Start animation after page load
        setTimeout(animateCounter, 500);
    </script>
    """, unsafe_allow_html=True)