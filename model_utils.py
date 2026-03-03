# model_utils.py
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

class DogCatClassifier:
    def __init__(self, model_path='model/model.tflite', labels_path='model/labels.txt'):
        """
        Khởi tạo classifier với model từ Teachable Machine
        """
        self.model_path = model_path
        self.labels_path = labels_path
        
        # Load model và labels
        self.interpreter = self.load_model()
        self.labels = self.load_labels()
        
        # Lấy thông tin input/output của model
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Kích thước input model (thường là 224x224 hoặc 96x96)
        self.input_shape = self.input_details[0]['shape'][1:3]  # [height, width]
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Labels: {self.labels}")
    
    def load_model(self):
        """Load TensorFlow Lite model"""
        try:
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_labels(self):
        """Load labels từ file"""
        try:
            with open(self.labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            return labels
        except Exception as e:
            print(f"Error loading labels: {e}")
            return ['dog', 'cat']  # Default labels
    
    def preprocess_image(self, image):
        """
        Tiền xử lý ảnh cho model
        Supports: PIL Image, numpy array, file path
        """
        # Nếu là đường dẫn file
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert BGR to RGB nếu cần (OpenCV uses BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Kiểm tra nếu là BGR (OpenCV)
            if image[0, 0, 0] > image[0, 0, 2]:  # Blue > Red
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize về kích thước model yêu cầu
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize: Teachable Machine models thường cần uint8 [0, 255]
        # Hoặc float32 [0, 1] tùy model
        
        # Kiểm tra input type của model
        input_type = self.input_details[0]['dtype']
        
        if input_type == np.uint8:
            # Model quantized uint8
            # Scale về [0, 255] nếu cần
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
        else:
            # Model float32
            image = image.astype(np.float32)
            image = image / 255.0  # Normalize về [0, 1]
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        """
        Dự đoán ảnh input
        Returns: (predicted_class, confidence, all_probabilities)
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                processed_image
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # Xử lý kết quả
            probabilities = output_data[0]

            # Nếu model là uint8 (trả về 0-255), chia cho 255.0 để đưa về 0.0 - 1.0
            if self.output_details[0]['dtype'] == np.uint8:
                probabilities = probabilities.astype(np.float32) / 255.0
            
            # Lấy class có probability cao nhất
            predicted_class_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class_idx])
            predicted_class = self.labels[predicted_class_idx]
            
            # Format probabilities với tên class
            class_probabilities = {}
            for i, (label, prob) in enumerate(zip(self.labels, probabilities)):
                class_probabilities[label] = float(prob)
            
            return predicted_class, confidence, class_probabilities
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, 0.0, {}
    
    def predict_from_file(self, file_path):
        """Dự đoán từ file path"""
        try:
            image = Image.open(file_path)
            return self.predict(image)
        except Exception as e:
            print(f"Error loading image file: {e}")
            return None, 0.0, {}
    
    def get_model_info(self):
        """Lấy thông tin về model"""
        info = {
            'input_shape': self.input_shape,
            'input_type': str(self.input_details[0]['dtype']),
            'labels': self.labels,
            'model_size': f"{os.path.getsize(self.model_path) / 1024:.2f} KB"
        }
        return info


# Test model
if __name__ == "__main__":
    classifier = DogCatClassifier()
    
    # Test với ảnh mẫu
    test_image_path = "test_images/dog1.jpg"
    
    if os.path.exists(test_image_path):
        predicted_class, confidence, probs = classifier.predict_from_file(test_image_path)
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"All probabilities: {probs}")
    else:
        print("Test image not found")
