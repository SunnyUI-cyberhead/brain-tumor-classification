# Brain Tumor MRI Classification System

A deep learning-based web application for classifying brain MRI images into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

![Brain Tumor Classification Dashboard](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🧠 Features

- **Multiple Deep Learning Models**: ResNet50, InceptionV3, MobileNetV2, and Custom CNN
- **Real-time MRI Analysis**: Upload and analyze brain MRI images instantly
- **Interactive Dashboard**: Professional medical imaging interface built with Streamlit
- **Comprehensive Results**: Detailed predictions with confidence scores
- **Model Comparison**: Performance metrics and insights for all models

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| ResNet50 | 96.54% | 96.38% | 96.42% | 96.40% |
| InceptionV3 | 94.87% | 94.65% | 94.73% | 94.69% |
| MobileNetV2 | 93.25% | 93.08% | 93.14% | 93.11% |
| Custom CNN | 88.56% | 88.34% | 88.45% | 88.39% |

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Git LFS (for downloading model files)
- CUDA-capable GPU (optional but recommended)

### Installation

1. **Clone the repository with Git LFS**
```bash
# Make sure Git LFS is installed
git lfs install

# Clone the repository
git clone https://github.com/SunnyUI-cyberhead/brain-tumor-classification.git
cd brain-tumor-classification
```

2. **Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run dashboard.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
brain-tumor-classification/
├── models/                    # Trained model files (Git LFS)
│   ├── ResNet50_best.h5
│   ├── InceptionV3_best.h5
│   ├── MobileNetV2_best.h5
│   └── Custom_CNN_best.h5
├── dashboard.py              # Streamlit web application
├── BrainTumour.py                  # Model training script
├── requirements.txt          # Python dependencies
├── model_info.json          # Model metadata
├── model_comparison_results.csv  # Performance metrics
├── confusion_matrix_*.png    # Model evaluation visualizations
└── training_history_*.png    # Training performance plots
```

## 💻 Usage

### Web Application
1. Launch the dashboard using `streamlit run dashboard.py`
2. Select a model from the sidebar
3. Upload an MRI image (JPG/PNG format)
4. Click "Analyze MRI Scan"
5. View the prediction results and confidence scores

### Training Models
To train your own models on a custom dataset:
```bash
python train.py --data_path /path/to/your/dataset
```

## 🖼️ Application Interface

### Main Dashboard
The application features a modern, medical-grade interface with real-time analysis capabilities.

### Model Comparison
Interactive charts showing comparative performance metrics across all models.

## ⚠️ Medical Disclaimer

**Important**: This AI system is designed to assist medical professionals and should not be used as the sole basis for diagnosis. Always consult with qualified healthcare providers for proper medical evaluation and treatment decisions.

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras 2.10+
- **Models**: Transfer Learning with pretrained ImageNet weights
- **Frontend**: Streamlit 1.25+
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: NumPy, Pandas, PIL/Pillow
- **Model Storage**: Git LFS for large model files

## 📊 Dataset

The models were trained on a brain tumor MRI dataset containing four classes:
- **Glioma**: Tumors arising from glial cells
- **Meningioma**: Tumors from the meninges
- **Pituitary Tumor**: Tumors in the pituitary gland
- **No Tumor**: Healthy brain MRI scans

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Arunov Chakraborty**
- GitHub: [@SunnyUI-cyberhead](https://github.com/SunnyUI-cyberhead)
- LinkedIn: [Connect on LinkedIn](https://linkedin.com/in/arunov-chakraborty)

## 🙏 Acknowledgments

- Thanks to Labmentix for providing brain tumor MRI dataset 
- The medical imaging community for valuable insights

## 📞 Support

If you encounter any issues or have questions, please:
- Open an issue on [GitHub Issues](https://github.com/SunnyUI-cyberhead/brain-tumor-classification/issues)
- Check existing issues before creating a new one

---

**⚕️ Note**: This project is for educational and research purposes. For medical diagnosis, always consult qualified healthcare professionals.
