"""
Brain Tumor MRI Classification - Complete Training Pipeline
This script handles data preprocessing, model building (custom CNN & transfer learning),
training, evaluation, and model comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BrainTumorClassifier:
    def __init__(self, data_dir, img_height=224, img_width=224, batch_size=32):
        """
        Initialize the Brain Tumor Classifier
        
        Args:
            data_dir: Path to dataset directory
            img_height: Height of input images
            img_width: Width of input images
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.num_classes = None
        self.class_names = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.models = {}
        self.histories = {}
        
    def explore_dataset(self):
        """Explore and analyze the dataset structure"""
        print("🔍 Exploring Dataset Structure...")
        
        # Count images per class
        class_counts = {}
        for split in ['train', 'test']:
            split_path = os.path.join(self.data_dir, split)
            if os.path.exists(split_path):
                print(f"\n{split.upper()} Set:")
                for class_name in os.listdir(split_path):
                    class_path = os.path.join(split_path, class_name)
                    if os.path.isdir(class_path):
                        count = len([f for f in os.listdir(class_path) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
                        print(f"  {class_name}: {count} images")
                        if split == 'train':
                            class_counts[class_name] = count
        
        # Visualize class distribution
        if class_counts:
            plt.figure(figsize=(10, 6))
            plt.bar(class_counts.keys(), class_counts.values())
            plt.title('Training Set Class Distribution')
            plt.xlabel('Tumor Type')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('class_distribution.png')
            plt.close()
            print("\n✅ Class distribution plot saved as 'class_distribution.png'")
            
    def prepare_data(self):
        """Prepare data generators with augmentation"""
        print("\n📊 Preparing Data Generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation and test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        self.val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.num_classes = len(self.train_generator.class_indices)
        self.class_names = list(self.train_generator.class_indices.keys())
        
        print(f"✅ Data prepared successfully!")
        print(f"   Classes: {self.class_names}")
        print(f"   Training samples: {self.train_generator.samples}")
        print(f"   Validation samples: {self.val_generator.samples}")
        print(f"   Test samples: {self.test_generator.samples}")
        
    def visualize_augmented_images(self):
        """Visualize augmented images"""
        print("\n🖼️ Visualizing Augmented Images...")
        
        # Get a batch of images
        images, labels = next(self.train_generator)
        
        plt.figure(figsize=(15, 10))
        for i in range(min(9, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            class_idx = np.argmax(labels[i])
            plt.title(f'Class: {self.class_names[class_idx]}')
            plt.axis('off')
        plt.suptitle('Augmented Training Images')
        plt.tight_layout()
        plt.savefig('augmented_samples.png')
        plt.close()
        print("✅ Augmented samples saved as 'augmented_samples.png'")
        
    def build_custom_cnn(self):
        """Build custom CNN architecture"""
        print("\n🏗️ Building Custom CNN Model...")
        
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Custom CNN model built successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def build_transfer_learning_model(self, base_model_name='ResNet50'):
        """Build transfer learning model"""
        print(f"\n🏗️ Building {base_model_name} Transfer Learning Model...")
        
        # Select base model
        if base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False,
                                 input_shape=(self.img_height, self.img_width, 3))
        elif base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(weights='imagenet', include_top=False,
                                    input_shape=(self.img_height, self.img_width, 3))
        elif base_model_name == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet', include_top=False,
                                    input_shape=(self.img_height, self.img_width, 3))
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                       input_shape=(self.img_height, self.img_width, 3))
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ {base_model_name} model built successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Trainable parameters: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
        
        return model, base_model
    
    def train_model(self, model, model_name, epochs=50):
        """Train a model with callbacks"""
        print(f"\n🚀 Training {model_name}...")
        
        # Create callbacks
        checkpoint = ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Train model
        history = model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
        
        self.models[model_name] = model
        self.histories[model_name] = history
        
        print(f"✅ {model_name} training completed!")
        
        return history
    
    def fine_tune_model(self, model, base_model, model_name, epochs=20):
        """Fine-tune a transfer learning model"""
        print(f"\n🎯 Fine-tuning {model_name}...")
        
        # Unfreeze top layers
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 20
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training
        history_fine = self.train_model(model, f"{model_name}_finetuned", epochs=epochs)
        
        return history_fine
    
    def evaluate_model(self, model, model_name):
        """Evaluate model performance"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Get predictions
        predictions = model.predict(self.test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     output_dict=True)
        
        # Print classification report
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png')
        plt.close()
        
        print(f"✅ Confusion matrix saved as 'confusion_matrix_{model_name}.png'")
        
        return report
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_history_{model_name}.png')
        plt.close()
        
        print(f"✅ Training history plot saved as 'training_history_{model_name}.png'")
    
    def compare_models(self, evaluation_results):
        """Compare all trained models"""
        print("\n📊 Model Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, report in evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': report['accuracy'],
                'Precision': report['weighted avg']['precision'],
                'Recall': report['weighted avg']['recall'],
                'F1-Score': report['weighted avg']['f1-score']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Accuracy', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(df_comparison.to_string(index=False))
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_comparison))
        width = 0.2
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, df_comparison[metric], width, 
                  label=metric, color=colors[i])
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        
        print("\n✅ Model comparison plot saved as 'model_comparison.png'")
        
        # Save comparison results
        df_comparison.to_csv('model_comparison_results.csv', index=False)
        print("✅ Model comparison results saved as 'model_comparison_results.csv'")
        
        return df_comparison

def main():
    """Main training pipeline"""
    print("🧠 Brain Tumor MRI Classification Training Pipeline")
    print("=" * 50)
    
    # Initialize classifier
    # Update this path to your dataset location
    DATA_DIR = r"D:\Machine Learning\New folder\Tumour"  # Change this to your actual dataset path
    
    classifier = BrainTumorClassifier(
        data_dir=DATA_DIR,
        img_height=224,
        img_width=224,
        batch_size=32
    )
    
    # Step 1: Explore dataset
    classifier.explore_dataset()
    
    # Step 2: Prepare data
    classifier.prepare_data()
    
    # Step 3: Visualize augmented images
    classifier.visualize_augmented_images()
    
    # Step 4: Build and train models
    evaluation_results = {}
    
    # Train Custom CNN
    custom_cnn = classifier.build_custom_cnn()
    history_cnn = classifier.train_model(custom_cnn, 'Custom_CNN', epochs=50)
    classifier.plot_training_history(history_cnn, 'Custom_CNN')
    eval_cnn = classifier.evaluate_model(custom_cnn, 'Custom_CNN')
    evaluation_results['Custom_CNN'] = eval_cnn
    
    # Train Transfer Learning Models
    transfer_models = ['ResNet50', 'MobileNetV2', 'InceptionV3', 'EfficientNetB0']
    
    for model_name in transfer_models:
        # Build and train
        model, base_model = classifier.build_transfer_learning_model(model_name)
        history = classifier.train_model(model, model_name, epochs=30)
        classifier.plot_training_history(history, model_name)
        
        # Fine-tune
        history_fine = classifier.fine_tune_model(model, base_model, model_name, epochs=20)
        classifier.plot_training_history(history_fine, f"{model_name}_finetuned")
        
        # Evaluate
        eval_result = classifier.evaluate_model(model, model_name)
        evaluation_results[model_name] = eval_result
    
    # Step 5: Compare models
    comparison_results = classifier.compare_models(evaluation_results)
    
    # Save best model info
    best_model_name = comparison_results.iloc[0]['Model']
    best_model_accuracy = comparison_results.iloc[0]['Accuracy']
    
    # Convert all numpy/tensor values to Python native types for JSON serialization
    results_for_json = []
    for record in comparison_results.to_dict('records'):
        clean_record = {}
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                clean_record[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                clean_record[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                clean_record[key] = int(value)
            else:
                clean_record[key] = value
        results_for_json.append(clean_record)
    
    model_info = {
        'best_model': best_model_name,
        'accuracy': float(best_model_accuracy),
        'all_results': results_for_json,
        'class_names': classifier.class_names,
        'num_classes': classifier.num_classes,
        'img_height': classifier.img_height,
        'img_width': classifier.img_width,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print("\n✅ Training pipeline completed successfully!")
    print(f"🏆 Best Model: {best_model_name} with {best_model_accuracy:.4f} accuracy")
    print("\n📁 Generated Files:")
    print("   - models/[model_name]_best.h5 (trained models)")
    print("   - class_distribution.png")
    print("   - augmented_samples.png")
    print("   - confusion_matrix_[model_name].png")
    print("   - training_history_[model_name].png")
    print("   - model_comparison.png")
    print("   - model_comparison_results.csv")
    print("   - model_info.json")

if __name__ == "__main__":
    main()