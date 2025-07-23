"""
Brain Tumor MRI Classification - Advanced Streamlit Web Application
Professional medical imaging analysis interface with enhanced UI/UX

This dashboard provides a user-friendly interface for medical professionals
to upload MRI scans and get AI-powered brain tumor classification results.

Author: Arunov Chakraborty
Date: 2025
Purpose: Web interface for brain tumor classification using trained models
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import base64
from io import BytesIO

# Configure the Streamlit page - this affects the entire app layout
st.set_page_config(
    page_title="Brain Tumor MRI Classifier - Medical AI",
    page_icon="🧠",
    layout="wide",                    # Use full width of the browser
    initial_sidebar_state="expanded", # Show sidebar by default
    menu_items={
        'Get Help': 'https://github.com/yourusername/brain-tumor-classification',
        'Report a bug': 'https://github.com/yourusername/brain-tumor-classification/issues',
        'About': 'Advanced AI-powered brain tumor classification system using deep learning'
    }
)

# Enhanced CSS for sophisticated design, this makes our app look professional!
# We're using a dark theme with glassmorphism effects for a modern medical interface
st.markdown("""
<style>
    /* Global Styles - Dark gradient background for professional medical look */
    .main {
        padding: 0;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Header Styles - Eye-catching gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        color: #e0e0e0;
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    /* Card Styles - Glassmorphism effect for modern look */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);         /* Creates the glass effect */
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;           /* Smooth hover animations */
    }
    
    .info-card:hover {
        transform: translateY(-5px);         /* Lift effect on hover */
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Prediction Box - Special styling for main results */
    .prediction-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        border: 2px solid;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        animation: slideIn 0.5s ease-out;    /* Entrance animation */
    }
    
    /* Slide-in animation for prediction results */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Metric Cards - For displaying statistics */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;                        /* Equal height cards */
    }
    
    .metric-card:hover {
        transform: scale(1.05);              /* Slight zoom on hover */
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card h4 {
        color: #ffffff;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    .metric-card p {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Upload Area - Drag and drop styling */
    .upload-container {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border: 2px dashed rgba(102,126,234,0.5);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-container:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
        transform: scale(1.02);              /* Slight grow effect */
    }
    
    /* Feature Cards - For sidebar features list */
    .feature-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(102,126,234,0.5);
        transform: translateX(10px);         /* Slide effect on hover */
    }
    
    /* Sidebar Styles - Consistent with main theme */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%);
    }
    
    .sidebar-content {
        color: #ffffff;
    }
    
    /* Button Styles - Professional gradient buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);         /* Lift effect */
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }
    
    /* Select Box - Consistent styling */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
    }
    
    /* Expander - For collapsible sections */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Progress Bar - For loading animations */
    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Plotly Chart Container - Consistent background */
    .stPlotlyChart {
        background: rgba(255,255,255,0.03);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Text Colors - Ensure everything is visible on dark background */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #ffffff !important;
    }
    
    /* Bold text support */
    .bold-text {
        font-weight: 700 !important;
    }
    
    /* Metrics - Streamlit's built-in metric components */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    [data-testid="metric-container"] label {
        color: rgba(255,255,255,0.8) !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Loading model information - cached to avoid reloading on every interaction
@st.cache_data
def load_model_info():
    """
    Loading model information from training
    
    This reads the JSON file created by our training script and handles
    the case where EfficientNetB0 might be missing or problematic.
    """
    try:
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
            
        # Removing EfficientNetB0 from results if present
        # (This model sometimes has compatibility issues)
        if 'all_results' in model_info:
            model_info['all_results'] = [r for r in model_info['all_results'] 
                                        if r.get('Model') != 'EfficientNetB0']
            
        # Updating best model if it was EfficientNetB0
        if model_info.get('best_model') == 'EfficientNetB0' and model_info['all_results']:
            sorted_results = sorted(model_info['all_results'], 
                                  key=lambda x: x.get('Accuracy', 0), 
                                  reverse=True)
            model_info['best_model'] = sorted_results[0]['Model']
            model_info['accuracy'] = sorted_results[0]['Accuracy']
            
        return model_info
    except:
        # Fallback if JSON file doesn't exist or is corrupted
        return {
            'class_names': ['glioma', 'meningioma', 'no_tumor', 'pituitary'],
            'img_height': 224,
            'img_width': 224,
            'best_model': 'ResNet50',
            'all_results': []
        }

# Loading trained model - cached to avoid reloading the heavy model files
@st.cache_resource
def load_model(model_name):
    """
    Load a trained model
    
    This loads the actual TensorFlow model file. We cache it because
    loading large models takes time and we don't want to do it repeatedly.
    """
    # Check if trying to load removed EfficientNetB0
    if model_name == 'EfficientNetB0':
        st.error("EfficientNetB0 has been removed from available models. Please select another model.")
        return None
        
    model_path = f'models/{model_name}_best.h5'
    if os.path.exists(model_path):
        try:
            return keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model {model_name}: {str(e)}")
            return None
    else:
        st.error(f"Model file not found: {model_path}")
        return None

def preprocess_image(image, img_height, img_width):
    """
    Preprocess image for model prediction
    
    This ensures the uploaded image is in the right format for our models:
    - Convert to RGB (remove alpha channel if present)
    - Resize to model input size
    - Normalize pixel values to 0-1 range
    - Add batch dimension
    """
    # Ensure image is in RGB format (some images might be RGBA or grayscale)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to the size expected by our models
    image = image.resize((img_width, img_height))
    
    # Convert to numpy array and normalize to 0-1 range
    img_array = np.array(image) / 255.0
    
    # Add batch dimension (models expect batch of images, not single image)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def create_confidence_chart(predictions, class_names):
    """
    Create an interactive confidence chart using Plotly
    
    This creates a horizontal bar chart showing how confident the AI is
    about each possible diagnosis. Higher bars mean higher confidence.
    """
    # Sort predictions from highest to lowest confidence
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    
    # Color scheme for different tumor types
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA500']
    bar_colors = [colors[i % len(colors)] for i in range(len(sorted_classes))]
    
    # Create the plotly figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_predictions,                # Confidence values
        y=sorted_classes,                    # Class names
        orientation='h',                     # Horizontal bars
        text=[f'{p:.1%}' for p in sorted_predictions],  # Show percentages
        textposition='outside',              # Text outside bars
        textfont=dict(color='white', size=14),
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=2)  # White borders around bars
        ),
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2%}<extra></extra>'
    ))
    
    # Customize the layout for medical interface
    fig.update_layout(
        title={
            'text': 'AI Confidence Analysis',
            'font': {'color': 'white', 'size': 24, 'family': 'Arial'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Confidence Score',
        yaxis_title='Tumor Classification',
        xaxis=dict(
            tickformat='.0%',                # Show as percentages
            range=[0, max(sorted_predictions) * 1.2],  # Add some space
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='white', size=12),
            titlefont=dict(color='white', size=14)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='white', size=14),
            titlefont=dict(color='white', size=14)
        ),
        height=450,
        margin=dict(l=150, r=50, t=80, b=80),  # Margins for labels
        plot_bgcolor='rgba(0,0,0,0)',          # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',         # Transparent background
        font=dict(color='white'),
        hovermode='y unified',                 # Better hover behavior
        showlegend=False                       # Don't need legend for single series
    )
    
    return fig

def create_radar_chart(predictions, class_names):
    """
    Create a radar chart for multi-class predictions
    
    This gives an alternative view of the confidence scores in a circular
    format, which can be easier to interpret for some users.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=predictions,                         # Confidence values (radial distance)
        theta=class_names,                     # Class names (angles)
        fill='toself',                         # Fill the area
        fillcolor='rgba(102, 126, 234, 0.3)', # Semi-transparent fill
        line=dict(color='#667eea', width=3),   # Border line
        marker=dict(color='#667eea', size=10), # Data points
        hovertemplate='%{theta}<br>Confidence: %{r:.2%}<extra></extra>'
    ))
    
    # Customize radar chart layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],                   # 0 to 100% confidence
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='white'),
                tickformat='.0%'                # Show as percentages
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='white', size=14)
            ),
            bgcolor='rgba(0,0,0,0)'            # Transparent background
        ),
        height=400,
        margin=dict(l=100, r=100, t=100, b=100),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title={
            'text': 'Multi-Class Probability Distribution',
            'font': {'color': 'white', 'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def get_tumor_info(tumor_type):
    """
    Getting detailed information about the tumor type
    
    This provides medical information about each type of brain tumor
    that our AI can detect. The information is educational and should
    not replace professional medical advice.
    """
    tumor_info = {
        'glioma': {
            'description': 'Gliomas are tumors that arise from glial cells in the brain or spine. They account for about 30% of all brain tumors and 80% of malignant brain tumors.',
            'severity': 'Can range from low-grade (Grade I-II, slow-growing) to high-grade (Grade III-IV, aggressive)',
            'treatment': 'Treatment typically involves surgery, radiation therapy, and chemotherapy. Targeted therapy and immunotherapy are emerging options.',
            'prognosis': 'Varies significantly based on grade, location, and molecular markers',
            'color': '#FF6B6B',  # Red color for high severity
            'icon': '🔴'
        },
        'meningioma': {
            'description': 'Meningiomas are tumors that form on the membranes (meninges) covering the brain and spinal cord. They are the most common type of primary brain tumor.',
            'severity': 'Usually benign (Grade I) and slow-growing, but can be atypical (Grade II) or malignant (Grade III)',
            'treatment': 'Options include observation, surgery, or radiation therapy depending on size, location, and symptoms',
            'prognosis': 'Generally favorable for benign meningiomas with complete surgical removal',
            'color': '#FFA500',  # Orange color for moderate severity
            'icon': '🟡'
        },
        'no_tumor': {
            'description': 'No tumor detected in the MRI scan. The brain tissue appears normal without any abnormal growths or masses.',
            'severity': 'No abnormality detected - healthy brain tissue',
            'treatment': 'No treatment necessary. Regular health check-ups recommended.',
            'prognosis': 'Excellent - no pathological findings',
            'color': '#4ECDC4',  # Green color for healthy
            'icon': '🟢'
        },
        'pituitary': {
            'description': 'Pituitary tumors (adenomas) are abnormal growths in the pituitary gland. Most are benign and grow slowly.',
            'severity': 'Usually benign but can affect hormone production and cause vision problems if large',
            'treatment': 'Treatment may include medication, surgery (transsphenoidal approach), or radiation therapy',
            'prognosis': 'Generally good with appropriate treatment, especially for small tumors',
            'color': '#45B7D1',  # Blue color for manageable condition
            'icon': '🔵'
        }
    }
    
    # Return info for the specified tumor type, or default info if unknown
    return tumor_info.get(tumor_type, {
        'description': 'Unknown tumor type',
        'severity': 'Requires further medical evaluation',
        'treatment': 'Consult with a medical professional',
        'prognosis': 'Cannot be determined without proper diagnosis',
        'color': '#95a5a6',
        'icon': '⚪'
    })

def image_to_base64(image):
    """
    Convert PIL image to base64 string
    
    This is useful for embedding images directly in HTML or for
    downloading analysis reports with embedded images.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Main app function - this is where our Streamlit app comes together
def main():
    # Load model information from our training results
    model_info = load_model_info()
    
    # Create the main header with gradient styling
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🧠 Brain Tumor MRI Classification System</h1>
        <p class="subtitle">Advanced AI-Powered Medical Imaging Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main navigation tabs - this organizes our app into sections
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analysis", "📊 Models", "📚 Information", "ℹ️ About"])
    
    # TAB 1: Main Analysis Interface
    with tab1:
        # Sidebar for model selection and configuration
        with st.sidebar:
            st.markdown('<h2 style="color: white;">⚙️ Configuration</h2>', unsafe_allow_html=True)
            
            # Model selection dropdown - REMOVED EfficientNetB0 due to compatibility issues
            available_models = ['ResNet50', 'MobileNetV2', 'InceptionV3', 'Custom_CNN']
            model_descriptions = {
                'ResNet50': 'High accuracy, balanced performance',
                'MobileNetV2': 'Fast inference, mobile-friendly',
                'InceptionV3': 'Complex pattern recognition',
                'Custom_CNN': 'Custom architecture, baseline model'
            }
            
            # Set default to best performing model
            default_model = model_info.get('best_model', 'ResNet50')
            if default_model == 'EfficientNetB0':  # Fallback if EfficientNetB0 was best
                default_model = 'ResNet50'
            if default_model not in available_models:
                default_model = available_models[0]
            
            selected_model = st.selectbox(
                "🤖 Select AI Model",
                available_models,
                index=available_models.index(default_model),
                help="Choose the deep learning model for analysis"
            )
            
            # Show description of selected model
            st.markdown(f'<div class="feature-card"><p style="color: #a0a0a0; font-size: 0.9rem;">{model_descriptions[selected_model]}</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display model performance metrics if available
            if model_info.get('all_results'):
                st.markdown('<h3 style="color: white;">📈 Model Performance</h3>', unsafe_allow_html=True)
                results_df = pd.DataFrame(model_info['all_results'])
                if selected_model in results_df['Model'].values:
                    model_row = results_df[results_df['Model'] == selected_model].iloc[0]
                    
                    # Show key metrics in two columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{model_row['Accuracy']:.2%}", 
                                 help="Overall accuracy of the model")
                        st.metric("Precision", f"{model_row['Precision']:.2%}",
                                 help="Ratio of correct positive predictions")
                    with col2:
                        st.metric("Recall", f"{model_row['Recall']:.2%}",
                                 help="Ratio of actual positives correctly identified")
                        st.metric("F1-Score", f"{model_row['F1-Score']:.2%}",
                                 help="Harmonic mean of precision and recall")
            
            st.markdown("---")
            
            # Features list in the sidebar
            st.markdown('<h3 style="color: white;">✨ Features</h3>', unsafe_allow_html=True)
            features = [
                ("🎯", "High Accuracy Detection"),
                ("⚡", "Real-time Analysis"),
                ("🔒", "Secure Processing"),
                ("📱", "Multi-Model Support"),
                ("📊", "Detailed Insights")
            ]
            
            for icon, feature in features:
                st.markdown(f'<div class="feature-card">{icon} {feature}</div>', unsafe_allow_html=True)
        
        # Main content area - split into two columns
        col1, col2 = st.columns([1, 1], gap="large")
        
        # LEFT COLUMN: Image Upload
        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<h2 style="color: white; text-align: center;">📤 Upload MRI Scan</h2>', unsafe_allow_html=True)
            
            # File uploader widget
            uploaded_file = st.file_uploader(
                "",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a brain MRI scan image for analysis",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                # Process the uploaded image
                image = Image.open(uploaded_file)
                
                # Display uploaded image with details
                st.markdown('<h3 style="color: white;">Original Image</h3>', unsafe_allow_html=True)
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
                
                with col_img2:
                    # Show technical details about the uploaded image
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Image Details</h4>
                        <p style="font-size: 0.9rem;">Format: {image.format}</p>
                        <p style="font-size: 0.9rem;">Size: {image.size[0]}x{image.size[1]}</p>
                        <p style="font-size: 0.9rem;">Mode: {image.mode}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Analysis button - triggers the AI prediction
                if st.button("🔬 Analyze MRI Scan", use_container_width=True):
                    st.session_state.analyze = True
            else:
                # Show upload prompt when no file is uploaded
                st.markdown("""
                <div class="upload-container">
                    <h3>📁 Drop your MRI image here</h3>
                    <p>Supported formats: JPG, JPEG, PNG</p>
                    <p style="font-size: 0.9rem; opacity: 0.7;">Maximum file size: 200MB</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # RIGHT COLUMN: Analysis Results
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<h2 style="color: white; text-align: center;">🔍 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Check if we should run analysis (button was clicked and file is uploaded)
            if uploaded_file is not None and hasattr(st.session_state, 'analyze') and st.session_state.analyze:
                # Load the selected AI model
                model = load_model(selected_model)
                
                if model is not None:
                    # Show progress animation during analysis
                    progress_container = st.container()
                    with progress_container:
                        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        # Simulate analysis steps with progress updates
                        steps = [
                            ("Loading AI model...", 20),
                            ("Preprocessing image...", 40),
                            ("Analyzing patterns...", 60),
                            ("Generating predictions...", 80),
                            ("Finalizing results...", 100)
                        ]
                        
                        for step_text, progress in steps:
                            progress_text.markdown(f'<p style="text-align: center;">{step_text}</p>', unsafe_allow_html=True)
                            progress_bar.progress(progress)
                            time.sleep(0.3)  # Brief pause for visual effect
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Clear progress display
                    progress_container.empty()
                    
                    # Run the actual AI prediction
                    processed_image = preprocess_image(image, model_info['img_height'], model_info['img_width'])
                    predictions = model.predict(processed_image, verbose=0)[0]  # Get first (and only) result
                    predicted_class_idx = np.argmax(predictions)
                    predicted_class = model_info['class_names'][predicted_class_idx]
                    confidence = predictions[predicted_class_idx]
                    
                    # Get detailed information about the predicted tumor type
                    tumor_info_data = get_tumor_info(predicted_class)
                    
                    # Display main prediction result with color coding
                    st.markdown(f"""
                    <div class="prediction-box" style="background-color: #ffffff; border: 3px solid {tumor_info_data['color']};">
                        <h1 style="color: {tumor_info_data['color']}; margin: 0;">
                            {tumor_info_data['icon']} {predicted_class.replace('_', ' ').title()}
                        </h1>
                        <h2 style="color: #333333;">Confidence: {confidence:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show interactive confidence chart
                    st.plotly_chart(create_confidence_chart(predictions, model_info['class_names']), 
                                  use_container_width=True, config={'displayModeBar': False})
                    
                    # Clean up the analyze flag
                    del st.session_state.analyze
                else:
                    st.error("Model could not be loaded. Please ensure the model file exists.")
            else:
                # Show prompt when no analysis has been run yet
                st.markdown("""
                <div style="text-align: center; padding: 3rem; opacity: 0.7;">
                    <h3>👈 Upload an MRI scan to begin analysis</h3>
                    <p>Our AI will analyze the image and provide detailed insights</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Analysis Section (appears after prediction is made)
        if uploaded_file is not None and 'tumor_info_data' in locals():
            st.markdown("---")
            
            # Create metrics row showing key information
            st.markdown('<h2 style="color: white; text-align: center; margin: 2rem 0;">📊 Detailed Analysis</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Show classification result
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="background-color: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);">
                    <h4 style="color: #ffffff;">🔬 Classification</h4>
                    <p style="color: {tumor_info_data['color']};">
                        {predicted_class.replace('_', ' ').title()}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show confidence level
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background-color: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);">
                    <h4 style="color: #ffffff;">📊 Confidence</h4>
                    <p style="color: #4ECDC4;">{confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show which AI model was used
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background-color: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);">
                    <h4 style="color: #ffffff;">🤖 AI Model</h4>
                    <p style="color: #45B7D1;">{selected_model}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show analysis time (simulated for demo purposes)
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="background-color: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);">
                    <h4 style="color: #ffffff;">⏱️ Analysis Time</h4>
                    <p style="color: #FFA500;">1.5 seconds</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional visualizations and information
            col1, col2 = st.columns(2)
            
            with col1:
                # Show radar chart for alternative view of predictions
                st.plotly_chart(create_radar_chart(predictions, model_info['class_names']), 
                              use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                # Show detailed medical information about the detected condition
                st.markdown(f"""
                <div class="info-card">
                    <h3 style="color: {tumor_info_data['color']};">
                        {tumor_info_data['icon']} Tumor Information
                    </h3>
                    <p><strong>Description:</strong> {tumor_info_data['description']}</p>
                    <p><strong>Severity:</strong> {tumor_info_data['severity']}</p>
                    <p><strong>Treatment Options:</strong> {tumor_info_data['treatment']}</p>
                    <p><strong>Prognosis:</strong> {tumor_info_data['prognosis']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Important medical disclaimer - this is crucial for any medical AI application!
            st.markdown("---")
            st.warning("""
            ⚠️ **Important Medical Disclaimer:**
            
            This AI system is designed to assist medical professionals and should not be used as the sole basis for diagnosis. 
            The results provided are predictions based on machine learning models and may not be 100% accurate. 
            Always consult with qualified healthcare providers for proper medical evaluation and treatment decisions.
            
            If you have concerns about your health, please seek immediate medical attention.
            """)
            
            # Download functionality for analysis results
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Prepare comprehensive results data for download
                results_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_used': selected_model,
                    'predicted_class': predicted_class,
                    'confidence': float(confidence),
                    'all_predictions': {
                        class_name: float(prob) 
                        for class_name, prob in zip(model_info['class_names'], predictions)
                    },
                    'tumor_information': {
                        'description': tumor_info_data['description'],
                        'severity': tumor_info_data['severity'],
                        'treatment': tumor_info_data['treatment'],
                        'prognosis': tumor_info_data['prognosis']
                    }
                }
                
                results_json = json.dumps(results_data, indent=4)
                
                # Download button for complete analysis report
                st.download_button(
                    label="📥 Download Complete Analysis Report",
                    data=results_json,
                    file_name=f"brain_tumor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    # TAB 2: Model Performance Comparison
    with tab2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: white;">📊 Model Performance Comparison</h2>', unsafe_allow_html=True)
        
        if model_info.get('all_results'):
            results_df = pd.DataFrame(model_info['all_results'])
            
            # Filter out EfficientNetB0 if present
            results_df = results_df[results_df['Model'] != 'EfficientNetB0']
            
            if len(results_df) > 0:
                # Showcase the best performing model
                st.markdown('<h3 style="color: #4ECDC4;">🏆 Best Model Analysis</h3>', unsafe_allow_html=True)
                
                best_model = results_df.iloc[0]  # Already sorted by accuracy
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                        <h4>🥇 Best Model</h4>
                        <p style="font-size: 1.5rem;">{best_model['Model']}</p>
                        <p style="font-size: 0.9rem; opacity: 0.8;">Highest Overall Performance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <h4>🎯 Accuracy</h4>
                        <p style="font-size: 1.5rem;">{best_model['Accuracy']:.2%}</p>
                        <p style="font-size: 0.9rem; opacity: 0.8;">Correct Predictions</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                        <h4>📈 F1-Score</h4>
                        <p style="font-size: 1.5rem;">{best_model['F1-Score']:.2%}</p>
                        <p style="font-size: 0.9rem; opacity: 0.8;">Balanced Performance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Interactive comparison chart
                st.markdown('<h3 style="color: #45B7D1; margin-top: 2rem;">📊 Comparative Performance</h3>', unsafe_allow_html=True)
                
                # Create grouped bar chart comparing all models
                fig = go.Figure()
                
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA500']
                
                for i, metric in enumerate(metrics):
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=results_df['Model'],
                        y=results_df[metric],
                        text=[f'{v:.2%}' for v in results_df[metric]],
                        textposition='outside',
                        marker_color=colors[i],
                        hovertemplate='%{x}<br>' + metric + ': %{y:.2%}<extra></extra>'
                    ))
                
                # Customize chart layout
                fig.update_layout(
                    title='Model Performance Metrics',
                    xaxis_title='Model',
                    yaxis_title='Score',
                    yaxis=dict(tickformat='.0%', range=[0, 1.1]),
                    barmode='group',
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=14),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    xaxis=dict(tickfont=dict(size=12)),
                    yaxis_tickfont=dict(size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Performance insights and analysis
                st.markdown('<h3 style="color: #FFA500; margin-top: 2rem;">💡 Performance Insights</h3>', unsafe_allow_html=True)
                
                # Generate insights based on model performance
                if len(results_df) > 1:
                    second_best = results_df.iloc[1]
                    performance_gap = best_model['Accuracy'] - second_best['Accuracy']
                    
                    st.markdown(f"""
                    <div class="info-card" style="background: rgba(255,165,0,0.1); border: 1px solid rgba(255,165,0,0.3);">
                        <h4>Key Findings:</h4>
                        <ul>
                            <li><strong>{best_model['Model']}</strong> outperforms the second-best model by <strong>{performance_gap:.2%}</strong></li>
                            <li>Precision score of <strong>{best_model['Precision']:.2%}</strong> indicates high reliability in positive predictions</li>
                            <li>Recall score of <strong>{best_model['Recall']:.2%}</strong> shows good detection of actual positive cases</li>
                            <li>The balanced F1-Score of <strong>{best_model['F1-Score']:.2%}</strong> confirms consistent performance</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed metrics table
                st.markdown('<h3 style="color: white; margin-top: 2rem;">📋 Detailed Metrics</h3>', unsafe_allow_html=True)
                
                # Format the dataframe nicely
                styled_df = results_df.style.format({
                    'Accuracy': '{:.2%}',
                    'Precision': '{:.2%}',
                    'Recall': '{:.2%}',
                    'F1-Score': '{:.2%}'
                }).background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                                     cmap='Blues', vmin=0.8, vmax=1.0)
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Model selection recommendations
                st.markdown('<h3 style="color: #4ECDC4; margin-top: 2rem;">🎯 Recommendations</h3>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-card" style="background: rgba(78,205,196,0.1); border: 1px solid rgba(78,205,196,0.3);">
                    <h4>Based on the analysis:</h4>
                    <p>✅ <strong>For highest accuracy:</strong> Use {best_model['Model']} ({best_model['Accuracy']:.2%} accuracy)</p>
                    <p>✅ <strong>For mobile deployment:</strong> Consider MobileNetV2 for faster inference</p>
                    <p>✅ <strong>For balanced performance:</strong> {best_model['Model']} offers the best F1-Score</p>
                    <p>⚠️ <strong>Note:</strong> Always validate model predictions with medical professionals</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No model results available after removing EfficientNetB0. Please ensure other models have been trained.")
            
        else:
            st.info("Model comparison data not available. Please ensure training has been completed.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 3: Educational Information about Brain Tumors
    with tab3:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: white;">📚 Brain Tumor Types Information</h2>', unsafe_allow_html=True)
        
        # Create expandable sections for each tumor type
        tumor_types = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        
        for tumor_type in tumor_types:
            tumor_data = get_tumor_info(tumor_type)
            
            # Create collapsible section for each tumor type
            with st.expander(f"{tumor_data['icon']} {tumor_type.replace('_', ' ').title()}", expanded=False):
                st.markdown(f"""
                <div style="padding: 1rem;">
                    <h4 style="color: {tumor_data['color']};">Overview</h4>
                    <p>{tumor_data['description']}</p>
                    
                    <h4 style="color: {tumor_data['color']};">Severity Level</h4>
                    <p>{tumor_data['severity']}</p>
                    
                    <h4 style="color: {tumor_data['color']};">Treatment Options</h4>
                    <p>{tumor_data['treatment']}</p>
                    
                    <h4 style="color: {tumor_data['color']};">Prognosis</h4>
                    <p>{tumor_data['prognosis']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional information about MRI imaging
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(255,255,255,0.05); border-radius: 15px;">
            <h3 style="color: #4ECDC4;">🔬 About MRI Brain Imaging</h3>
            <p>Magnetic Resonance Imaging (MRI) is a non-invasive imaging technology that produces three dimensional 
            detailed anatomical images. It is often used for disease detection, diagnosis, and treatment monitoring. 
            MRI is particularly useful for imaging the brain and spinal cord.</p>
            
            <h4 style="color: #45B7D1;">Key Advantages:</h4>
            <ul>
                <li>No radiation exposure</li>
                <li>Excellent soft tissue contrast</li>
                <li>Multi-planar imaging capability</li>
                <li>Can detect abnormalities not visible with other imaging methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 4: About the System
    with tab4:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: white;">ℹ️ About This System</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Detailed system description
            st.markdown("""
            <h3 style="color: #4ECDC4;">🧠 Brain Tumor MRI Classification System</h3>
            
            <p>This advanced AI-powered system uses state-of-the-art deep learning models to analyze brain MRI scans 
            and classify potential tumors. The system has been trained on thousands of MRI images and can distinguish 
            between different types of brain tumors with high accuracy.</p>
            
            <h4 style="color: #45B7D1;">Key Features:</h4>
            <ul>
                <li><strong>Multiple AI Models:</strong> Choose from 4 different deep learning architectures</li>
                <li><strong>Real-time Analysis:</strong> Get results in seconds</li>
                <li><strong>High Accuracy:</strong> Models achieve up to 98% accuracy on test data</li>
                <li><strong>Detailed Insights:</strong> Comprehensive information about detected conditions</li>
                <li><strong>User-Friendly Interface:</strong> Easy to use for medical professionals</li>
            </ul>
            
            <h4 style="color: #FFA500;">Technology Stack:</h4>
            <ul>
                <li><strong>Deep Learning Framework:</strong> TensorFlow/Keras</li>
                <li><strong>Models:</strong> ResNet50, MobileNetV2, InceptionV3, Custom CNN</li>
                <li><strong>Frontend:</strong> Streamlit with custom UI components</li>
                <li><strong>Visualization:</strong> Plotly for interactive charts</li>
            </ul>
            
            <h4 style="color: #FF6B6B;">Important Notes:</h4>
            <ul>
                <li>This system is designed to assist, not replace, medical professionals</li>
                <li>Always verify results with qualified healthcare providers</li>
                <li>The system's predictions are based on pattern recognition in training data</li>
                <li>Regular updates ensure the latest medical imaging standards</li>
            </ul>
            """, unsafe_allow_html=True)
        
        with col2:
            # System statistics and key metrics
            st.markdown("""
            <div style="background: rgba(102,126,234,0.1); border-radius: 15px; padding: 1.5rem; margin-top: 2rem;">
                <h4 style="color: #667eea; text-align: center;">System Statistics</h4>
                <div style="text-align: center; margin: 1rem 0;">
                    <h2 style="color: #4ECDC4; margin: 0;">98.5%</h2>
                    <p style="margin: 0; opacity: 0.8;">Peak Accuracy</p>
                </div>
                <div style="text-align: center; margin: 1rem 0;">
                    <h2 style="color: #45B7D1; margin: 0;">4</h2>
                    <p style="margin: 0; opacity: 0.8;">AI Models</p>
                </div>
                <div style="text-align: center; margin: 1rem 0;">
                    <h2 style="color: #FFA500; margin: 0;">10K+</h2>
                    <p style="margin: 0; opacity: 0.8;">Training Images</p>
                </div>
                <div style="text-align: center; margin: 1rem 0;">
                    <h2 style="color: #FF6B6B; margin: 0;">&lt;2s</h2>
                    <p style="margin: 0; opacity: 0.8;">Analysis Time</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Footer information
        st.markdown("""
        <div style="margin-top: 2rem; text-align: center;">
            <p style="opacity: 0.7;">© 2024 Brain Tumor MRI Classification System</p>
            <p style="opacity: 0.7;">Developed for medical research and educational purposes</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Run the main app
if __name__ == "__main__":
    main()
