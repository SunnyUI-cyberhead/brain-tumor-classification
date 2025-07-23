## Brain Tumor MRI Classification System

A deep learning-based web application for classifying brain MRI images into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

![Brain Tumor Classification Dashboard](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)

## Features

- **Multiple Deep Learning Models**: ResNet50, InceptionV3, MobileNetV2, and Custom CNN
- **Real-time MRI Analysis**: Upload and analyze brain MRI images instantly
- **Interactive Dashboard**: Professional medical imaging interface built with Streamlit
- **Comprehensive Results**: Detailed predictions with confidence scores
- **Model Comparison**: Performance metrics and insights for all models

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| ResNet50 | 96.54% | 96.38% | 96.42% | 96.40% |
| InceptionV3 | 94.87% | 94.65% | 94.73% | 94.69% |
| MobileNetV2 | 93.25% | 93.08% | 93.14% | 93.11% |
| Custom CNN | 88.56% | 88.34% | 88.45% | 88.39% |

## Quick Start

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (optional but recommended)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run dashboard.py
```

4. Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
brain-tumor-classification/
├── models/                    # Trained model files
├── dashboard.py              # Streamlit web application
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
├── model_info.json          # Model metadata
├── model_comparison_results.csv  # Performance metrics
├── confusion_matrix_*.png    # Model evaluation visualizations
└── training_history_*.png    # Training performance plots
```

## Usage

### Web Application
1. Launch the dashboard using `streamlit run dashboard.py`
2. Select a model from the sidebar
3. Upload an MRI image (JPG/PNG)
4. Click "Analyze MRI Scan"
5. View the prediction results and confidence scores

### Training Models
To train your own models:
```bash
python train.py
```

## Screenshots

### Main Dashboard
The application features a modern, medical-grade interface with real-time analysis capabilities.

### Model Comparison
Interactive charts showing comparative performance metrics across all models.

## Medical Disclaimer

**Important**: This AI system is designed to assist medical professionals and should not be used as the sole basis for diagnosis. Always consult with qualified healthcare providers for proper medical evaluation and treatment decisions.

## Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Models**: Transfer Learning with ResNet50, InceptionV3, MobileNetV2
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: NumPy, Pandas, PIL

## Dataset

The models were trained on a brain tumor MRI dataset containing four classes:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

[Arunov Chakraborty]
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourusername)

## Acknowledgments

- Thanks to the dataset providers
- TensorFlow/Keras team for the excellent deep learning framework
- Streamlit team for the amazing web app framework

---

**Note**: This project is for educational and research purposes. For medical diagnosis, always consult qualified healthcare professionals.
