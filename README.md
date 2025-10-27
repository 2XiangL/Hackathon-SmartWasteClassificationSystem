# 🌱 Smart Waste Classification System

Intelligent waste recognition and analysis platform based on YOLOv8 deep learning and Qwen large language model

## ✨ Features

### 🎯 Core Functions
- **🔍 Precise Recognition**: High-precision waste detection using YOLOv8 technology
- **📊 Smart Classification**: Automatic waste type identification and statistical analysis
- **🔬 Processing Suggestions**: AI-driven eco-friendly processing recommendations
- **🌍 Carbon Footprint Analysis**: Detailed environmental impact assessment

### 🎨 Interface Features
- **Modern Design**: Eco-themed gradient color scheme
- **Responsive Layout**: Perfect adaptation for desktop and mobile devices
- **Smooth Animations**: Elegant transitions and interactive effects
- **Card-based Layout**: Clear information hierarchy and visual guidance

## 🚀 Quick Start

### Install Dependencies

#### Using uv (Recommended)
```bash
uv venv
source ./.venv/bin/activate # Windows: venv\Scripts\activate
uv pip install -r requirements.txt
```

#### Using conda
```bash
conda create -n waste-classification python=3.11
conda activate waste-classification
pip install -r requirements.txt
```

#### Using pip
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Launch Application

```bash
streamlit run app.py
```

After launch, visit http://localhost:8501

## 📋 Usage Instructions

### 1. API Configuration
- Enter your Qwen API key in the sidebar
- Please visit [Alibaba Cloud Bailian Platform](https://www.aliyun.com/product/bailian) to get your API key

### 2. Image Upload
- Click the file upload area to select an image
- Supports JPG, PNG, JPEG formats
- Recommended: clear images with sufficient lighting

### 3. View Results
- **Detection Statistics**: Real-time display of detected waste quantity and types
- **Comparison Display**: Side-by-side display of original image and recognition results
- **Smart Analysis**: Detailed classification and processing suggestions
- **Environmental Impact**: Carbon footprint calculation and eco-friendly recommendations

## 💡 Usage Tips

- 📸 **Image Quality**: Ensure images are clear with waste items centered in the frame
- 🎯 **Single Type**: It's recommended to analyze one type of waste at a time for more accurate results
- 📏 **Appropriate Size**: Avoid images that are too small or blurry
- 🔄 **Multiple Attempts**: If recognition is inaccurate, try reshooting from different angles

## 🎯 Technical Architecture

### Backend Technologies
- **YOLOv8**: Object detection model
- **Qwen Large Model**: Natural language processing and content analysis
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing

### Frontend Technologies
- **Streamlit**: Web application framework
- **HTML5/CSS3**: Modern interface design
- **JavaScript**: Interactive effects and animations

## 📁 Project Structure

```
EnvironmentalProtectionProject/
├── app.py              # Main application file
├── style.css           # Custom styles file
├── utils.py            # Utility functions library
├── model.py            # Model loading and management
├── template.py         # AI prompt templates
├── encode_image.py     # Image encoding tool
├── requirements.txt    # Dependencies list
└── README.MD          # Project documentation
```

## 📸 Demo

![](./data/demo.gif)

## 🤝 Acknowledgments

- Thanks to [teamsmcorg](https://github.com/teamsmcorg/Waste-Classification-using-YOLOv8) for the pre-trained model
- Thanks to [Qwen Team](https://qwen.aliyun.com/) for large language model support

## 👥 Development Team

- **Changyu Chen**
- **Baichuan Jiang**
- **Dongyang Yu**
- **Yunwei Long**
- **Xiang Li**
- **Yian Wang**
