"""
Brain Tumor MRI Classification - Complete Training Pipeline
This script handles data preprocessing, model building (custom CNN & transfer learning),
training, evaluation, and model comparison.

Author: Arunov Chakraborty
Date: 2025
Purpose: Train multiple deep learning models to classify brain tumors from MRI scans
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

# Set random seeds for reproducibility - this ensures we get consistent results across runs
# Very important for comparing different models fairly!
np.random.seed(42)
tf.random.set_seed(42)

class BrainTumorClassifier:
    def __init__(self, data_dir, img_height=224, img_width=224, batch_size=32):
        """
        Initialize the Brain Tumor Classifier
        
        Why these defaults?
        - 224x224: Standard input size for most pre-trained models
        - batch_size=32: Good balance between memory usage and training stability
        
        Args:
            data_dir: Path to dataset directory (should have train/test folders)
            img_height: Height of input images
            img_width: Width of input images
            batch_size: Batch size for training (adjust based on your GPU memory)
        """
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        
        # These will be populated later during data preparation
        self.num_classes = None
        self.class_names = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        
        # Store our trained models and their training histories
        self.models = {}
        self.histories = {}
        
    def explore_dataset(self):
        """
        Explore and analyze the dataset structure
        
        This is super important! Always understand your data before training.
        We'll check class distribution to see if we have balanced data.
        """
        print("🔍 Exploring Dataset Structure...")
        
        # Count images per class - helps us understand if we have class imbalance
        class_counts = {}
        for split in ['train', 'test']:
            split_path = os.path.join(self.data_dir, split)
            if os.path.exists(split_path):
                print(f"\n{split.upper()} Set:")
                for class_name in os.listdir(split_path):
                    class_path = os.path.join(split_path, class_name)
                    if os.path.isdir(class_path):
                        # Count only actual image files, not other random files
                        count = len([f for f in os.listdir(class_path) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
                        print(f"  {class_name}: {count} images")
                        if split == 'train':
                            class_counts[class_name] = count
        
        # Visualize class distribution - this helps spot class imbalance issues
        if class_counts:
            plt.figure(figsize=(10, 6))
            plt.bar(class_counts.keys(), class_counts.values())
            plt.title('Training Set Class Distribution')
            plt.xlabel('Tumor Type')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('class_distribution.png')
            plt.close()  # Close to free memory
            print("\n✅ Class distribution plot saved as 'class_distribution.png'")
            
    def prepare_data(self):
        """
        Prepare data generators with augmentation
        
        Data augmentation is crucial for medical imaging! We have limited data,
        so we artificially increase variety to prevent overfitting.
        """
        print("\n📊 Preparing Data Generators...")
        
        # Data augmentation for training - these transforms help the model generalize better
        # Each parameter is chosen specifically for medical images
        train_datagen = ImageDataGenerator(
            rescale=1./255,              # Normalize pixel values to 0-1 range
            rotation_range=20,           # Rotate images up to 20 degrees
            width_shift_range=0.2,       # Shift images horizontally
            height_shift_range=0.2,      # Shift images vertically  
            horizontal_flip=True,        # Flip images horizontally
            vertical_flip=True,          # Flip images vertically (okay for brain scans)
            zoom_range=0.2,              # Zoom in/out slightly
            brightness_range=[0.8, 1.2], # Adjust brightness (simulates different scan conditions)
            fill_mode='nearest',         # How to fill pixels after transforms
            validation_split=0.2         # Use 20% of training data for validation
        )
        
        # Only rescaling for validation and test - no augmentation here!
        # We want to evaluate on clean, unmodified images
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Set up directory paths
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        
        # Create data generators - these will feed batches of images to our models
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',    # One-hot encoded labels
            subset='training',           # Use training split
            shuffle=True                 # Shuffle for better training
        )
        
        # Validation generator uses same augmentation setup but different subset
        self.val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',         # Use validation split
            shuffle=False                # Don't shuffle validation data
        )
        
        # Test generator - completely separate data, no augmentation
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False                # Never shuffle test data
        )
        
        # Store important info about our dataset
        self.num_classes = len(self.train_generator.class_indices)
        self.class_names = list(self.train_generator.class_indices.keys())
        
        print(f"✅ Data prepared successfully!")
        print(f"   Classes: {self.class_names}")
        print(f"   Training samples: {self.train_generator.samples}")
        print(f"   Validation samples: {self.val_generator.samples}")
        print(f"   Test samples: {self.test_generator.samples}")
        
    def visualize_augmented_images(self):
        """
        Visualize augmented images
        
        This is really helpful to see what our augmentation is doing.
        Sometimes augmentation can be too aggressive and distort important features!
        """
        print("\n🖼️ Visualizing Augmented Images...")
        
        # Get a batch of augmented images from the training generator
        images, labels = next(self.train_generator)
        
        # Create a nice grid to show the augmented samples
        plt.figure(figsize=(15, 10))
        for i in range(min(9, len(images))):  # Show up to 9 images
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            # Show which class this image belongs to
            class_idx = np.argmax(labels[i])
            plt.title(f'Class: {self.class_names[class_idx]}')
            plt.axis('off')  # Remove axes for cleaner look
        plt.suptitle('Augmented Training Images')
        plt.tight_layout()
        plt.savefig('augmented_samples.png')
        plt.close()
        print("✅ Augmented samples saved as 'augmented_samples.png'")
        
    def build_custom_cnn(self):
        """
        Build custom CNN architecture
        
        This is our baseline model built from scratch. It's designed specifically
        for brain tumor classification with progressively larger filters and 
        regularization to prevent overfitting.
        """
        print("\n🏗️ Building Custom CNN Model...")
        
        model = keras.Sequential([
            # First Convolutional Block - Start with smaller filters to capture fine details
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),  # Helps with training stability
            layers.MaxPooling2D((2, 2)),  # Reduce spatial dimensions
            layers.Dropout(0.25),         # Prevent overfitting early
            
            # Second Convolutional Block - More filters to capture complex patterns
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),  # BatchNorm after each conv layer
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # Double conv for deeper features
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block - Even more filters for abstract features
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block - Highest level features
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers - Classification head
            layers.Flatten(),              # Convert 2D feature maps to 1D
            layers.Dense(512, activation='relu'),  # First dense layer
            layers.BatchNormalization(),
            layers.Dropout(0.5),          # Higher dropout in dense layers
            layers.Dense(256, activation='relu'),  # Second dense layer
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')  # Output layer
        ])
        
        # Compile the model with appropriate loss and optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001),  # Adam is usually good for most tasks
            loss='categorical_crossentropy',      # Standard for multi-class classification
            metrics=['accuracy']                  # Track accuracy during training
        )
        
        print("✅ Custom CNN model built successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def build_transfer_learning_model(self, base_model_name='ResNet50'):
        """
        Build transfer learning model
        
        Transfer learning is often better than training from scratch, especially
        with limited medical data. We use pre-trained weights and adapt them.
        """
        print(f"\n🏗️ Building {base_model_name} Transfer Learning Model...")
        
        # Select the appropriate base model - each has different strengths
        if base_model_name == 'ResNet50':
            # ResNet50: Great for general image classification, good accuracy
            base_model = ResNet50(weights='imagenet', include_top=False,
                                 input_shape=(self.img_height, self.img_width, 3))
        elif base_model_name == 'MobileNetV2':
            # MobileNetV2: Lightweight, good for deployment, fast inference
            base_model = MobileNetV2(weights='imagenet', include_top=False,
                                    input_shape=(self.img_height, self.img_width, 3))
        elif base_model_name == 'InceptionV3':
            # InceptionV3: Good at handling multiple scales, complex patterns
            base_model = InceptionV3(weights='imagenet', include_top=False,
                                    input_shape=(self.img_height, self.img_width, 3))
        elif base_model_name == 'EfficientNetB0':
            # EfficientNetB0: State-of-art efficiency, good accuracy-to-size ratio
            base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                       input_shape=(self.img_height, self.img_width, 3))
        
        # Freeze base model layers initially - we'll fine-tune later
        # This prevents destroying the pre-trained features early in training
        base_model.trainable = False
        
        # Build our custom head on top of the base model
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        x = base_model(inputs, training=False)  # training=False keeps BatchNorm frozen
        x = layers.GlobalAveragePooling2D()(x)  # Better than Flatten for transfer learning
        x = layers.Dense(256, activation='relu')(x)  # First custom layer
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)              # Regularization
        x = layers.Dense(128, activation='relu')(x)  # Second custom layer
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)  # Final classification
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),  # Start with higher learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ {base_model_name} model built successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Trainable parameters: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
        
        return model, base_model
    
    def train_model(self, model, model_name, epochs=50):
        """
        Train a model with callbacks
        
        We use several callbacks to make training more robust:
        - ModelCheckpoint: Save the best model automatically
        - EarlyStopping: Stop if we're not improving (prevent overfitting)
        - ReduceLROnPlateau: Lower learning rate when stuck
        """
        print(f"\n🚀 Training {model_name}...")
        
        # Create callbacks for better training
        checkpoint = ModelCheckpoint(
            f'models/{model_name}_best.h5',  # Save path
            monitor='val_loss',              # Watch validation loss
            save_best_only=True,             # Only save when we improve
            verbose=1                        # Print when saving
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',              # Watch validation loss
            patience=10,                     # Wait 10 epochs without improvement
            restore_best_weights=True,       # Go back to best weights
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',              # Watch validation loss
            factor=0.5,                      # Reduce LR by half
            patience=5,                      # Wait 5 epochs before reducing
            min_lr=1e-7,                     # Don't go below this
            verbose=1
        )
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Train the model - this is where the magic happens!
        history = model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1  # Show progress
        )
        
        # Store the trained model and its history
        self.models[model_name] = model
        self.histories[model_name] = history
        
        print(f"✅ {model_name} training completed!")
        
        return history
    
    def fine_tune_model(self, model, base_model, model_name, epochs=20):
        """
        Fine-tune a transfer learning model
        
        After initial training, we unfreeze some layers of the pre-trained model
        and train with a lower learning rate. This often gives better results!
        """
        print(f"\n🎯 Fine-tuning {model_name}...")
        
        # Unfreeze the base model for fine-tuning
        base_model.trainable = True
        
        # Only fine-tune the top layers - bottom layers have general features
        # that we usually want to keep
        fine_tune_at = len(base_model.layers) - 20  # Fine-tune last 20 layers
        
        # Keep the earlier layers frozen
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Recompile with a much lower learning rate
        # High learning rate would destroy the pre-trained features!
        model.compile(
            optimizer=Adam(learning_rate=0.0001),  # 10x lower learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training with fine-tuning
        history_fine = self.train_model(model, f"{model_name}_finetuned", epochs=epochs)
        
        return history_fine
    
    def evaluate_model(self, model, model_name):
        """
        Evaluate model performance
        
        We'll get detailed metrics and create visualizations to understand
        how well our model is performing on each class.
        """
        print(f"\n📊 Evaluating {model_name}...")
        
        # Get predictions on test set
        predictions = model.predict(self.test_generator)
        y_pred = np.argmax(predictions, axis=1)  # Convert probabilities to class predictions
        y_true = self.test_generator.classes        # True labels
        
        # Calculate detailed metrics using sklearn
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     output_dict=True)
        
        # Print human-readable classification report
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Create confusion matrix - shows where our model gets confused
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix with nice styling
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
        """
        Plot training history
        
        These plots help us understand if the model trained well:
        - Accuracy should increase over time
        - Loss should decrease over time
        - Training and validation curves should be close (not too much gap)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot - higher is better
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot - lower is better
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
        """
        Compare all trained models
        
        This gives us a nice overview of which model performed best
        and helps us make informed decisions about which one to use.
        """
        print("\n📊 Model Comparison")
        
        # Create comparison dataframe with key metrics
        comparison_data = []
        for model_name, report in evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': report['accuracy'],
                'Precision': report['weighted avg']['precision'],
                'Recall': report['weighted avg']['recall'],
                'F1-Score': report['weighted avg']['f1-score']
            })
        
        # Sort by accuracy to see which model is best
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Accuracy', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(df_comparison.to_string(index=False))
        
        # Create a nice visualization comparing all models
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_comparison))
        width = 0.2  # Width of bars
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Create grouped bar chart
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
        
        # Save results to CSV for future reference
        df_comparison.to_csv('model_comparison_results.csv', index=False)
        print("✅ Model comparison results saved as 'model_comparison_results.csv'")
        
        return df_comparison

def main():
    """
    Main training pipeline
    
    This is where everything comes together! We'll train multiple models,
    compare them, and save the best one for deployment.
    """
    print("🧠 Brain Tumor MRI Classification Training Pipeline")
    print("=" * 50)
    
    # Initialize classifier
    # TODO: Update this path to your actual dataset location!
    DATA_DIR = r"D:\Machine Learning\New folder\Tumour"  # Change this to your actual dataset path
    
    classifier = BrainTumorClassifier(
        data_dir=DATA_DIR,
        img_height=224,      # Standard size for most pre-trained models
        img_width=224,       # Square images work best
        batch_size=32        # Adjust based on your GPU memory
    )
    
    # Step 1: Explore dataset - Always understand your data first!
    classifier.explore_dataset()
    
    # Step 2: Prepare data - Set up augmentation and data generators
    classifier.prepare_data()
    
    # Step 3: Visualize augmented images - Make sure augmentation looks good
    classifier.visualize_augmented_images()
    
    # Step 4: Build and train models - The main event!
    evaluation_results = {}
    
    # Train Custom CNN first - our baseline model
    print("\n" + "="*50)
    print("Training Custom CNN (Baseline Model)")
    print("="*50)
    custom_cnn = classifier.build_custom_cnn()
    history_cnn = classifier.train_model(custom_cnn, 'Custom_CNN', epochs=50)
    classifier.plot_training_history(history_cnn, 'Custom_CNN')
    eval_cnn = classifier.evaluate_model(custom_cnn, 'Custom_CNN')
    evaluation_results['Custom_CNN'] = eval_cnn
    
    # Train Transfer Learning Models - Usually perform better than custom CNN
    transfer_models = ['ResNet50', 'MobileNetV2', 'InceptionV3', 'EfficientNetB0']
    
    for model_name in transfer_models:
        print("\n" + "="*50)
        print(f"Training {model_name} (Transfer Learning)")
        print("="*50)
        
        # Build and train with frozen base
        model, base_model = classifier.build_transfer_learning_model(model_name)
        history = classifier.train_model(model, model_name, epochs=30)
        classifier.plot_training_history(history, model_name)
        
        # Fine-tune with unfrozen layers - often gives better results
        print(f"\n🎯 Fine-tuning {model_name}...")
        history_fine = classifier.fine_tune_model(model, base_model, model_name, epochs=20)
        classifier.plot_training_history(history_fine, f"{model_name}_finetuned")
        
        # Evaluate final model
        eval_result = classifier.evaluate_model(model, model_name)
        evaluation_results[model_name] = eval_result
    
    # Step 5: Compare all models and find the winner!
    print("\n" + "="*50)
    print("Final Model Comparison")
    print("="*50)
    comparison_results = classifier.compare_models(evaluation_results)
    
    # Save information about the best model for the dashboard
    best_model_name = comparison_results.iloc[0]['Model']
    best_model_accuracy = comparison_results.iloc[0]['Accuracy']
    
    # Clean up data for JSON serialization (remove numpy types that cause issues)
    results_for_json = []
    for record in comparison_results.to_dict('records'):
        clean_record = {}
        for key, value in record.items():
            # Convert numpy types to Python native types
            if isinstance(value, np.ndarray):
                clean_record[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                clean_record[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                clean_record[key] = int(value)
            else:
                clean_record[key] = value
        results_for_json.append(clean_record)
    
    # Save model information for the dashboard to use
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
    
    # Save to JSON file
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Final summary
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
    print("\n🚀 Ready to run the dashboard with: streamlit run dashboard.py")

if __name__ == "__main__":
    main()
