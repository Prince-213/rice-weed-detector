"""
Enhanced YOLOv11 Model Integration with Better Error Handling
"""

from ultralytics import YOLO
from PIL import Image
import supervision as sv
import numpy as np
import cv2
import os
import sys

class YOLOv11WeedDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        """
        Initialize YOLOv11 model for weed detection using supervision
        
        Args:
            model_path (str): Path to your trained YOLOv11 model
            confidence_threshold (float): Minimum confidence for detections
        """
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        try:
            print(f"Attempting to load model: {model_path}")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                print("Attempting to download default model...")
                
                # Try to download a default model
                try:
                    self.model = YOLO('yolov8n.pt')  # This will auto-download
                    print("Successfully downloaded and loaded YOLOv8n model")
                except Exception as download_error:
                    print(f"Failed to download default model: {download_error}")
                    return
            else:
                # Load the specified model
                self.model = YOLO(model_path)
                print(f"Successfully loaded model: {model_path}")
            
            # Initialize supervision annotators
            self.box_annotator = sv.BoxAnnotator(
                color=sv.Color.RED,
                thickness=2,
            )
            self.label_annotator = sv.LabelAnnotator(
                color=sv.Color.RED,
                text_color=sv.Color.WHITE,
                text_thickness=1,
                text_scale=0.6
            )
            
            # Print model information
            if self.model and hasattr(self.model, 'names'):
                print(f"Model classes: {self.model.names}")
                print(f"Total classes: {len(self.model.names)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Error type: {type(e).__name__}")
            print("Please check:")
            print("1. Model file exists and is valid")
            print("2. All dependencies are installed (ultralytics, supervision)")
            print("3. Internet connection for auto-download")
            self.model = None
    
    def detect_weeds(self, image_path):
        """
        Perform weed detection on the input image using supervision
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            tuple: (annotated_image_pil, weed_detected, detection_count, max_confidence)
        """
        if self.model is None:
            print("Model is not loaded")
            return None, False, 0, 0.0
        
        try:
            print(f"Processing image: {image_path}")
            
            # Validate image path
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None, False, 0, 0.0
            
            # Load and validate image
            try:
                image = Image.open(image_path)
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                print(f"Image loaded successfully: {image.size}")
            except Exception as img_error:
                print(f"Error loading image: {img_error}")
                return None, False, 0, 0.0
            
            # Run YOLO prediction with error handling
            try:
                print("Running YOLO prediction...")
                results = self.model.predict(
                    source=image, 
                    conf=self.confidence_threshold,
                    verbose=False  # Reduce output noise
                )
                
                if not results:
                    print("No results from YOLO prediction")
                    return image, False, 0, 0.0
                
                result = results[0]
                print(f"YOLO prediction completed. Found {len(result.boxes) if result.boxes else 0} detections")
                
            except Exception as pred_error:
                print(f"Error during YOLO prediction: {pred_error}")
                return None, False, 0, 0.0
            
            # Convert to supervision detections
            try:
                detections = sv.Detections.from_ultralytics(result)
                print(f"Converted to supervision detections: {len(detections)}")
            except Exception as conv_error:
                print(f"Error converting to supervision format: {conv_error}")
                return image, False, 0, 0.0
            
            # Check for weed detections
            weed_detected = False
            weed_count = 0
            max_confidence = 0.0
            labels = []
            
            if len(detections) > 0:
                print("Processing detections...")
                for i, class_id in enumerate(detections.class_id):
                    try:
                        class_name = self.model.names[class_id].lower()
                        confidence = detections.confidence[i]
                        
                        print(f"Detection {i}: {class_name} ({confidence:.2f})")
                        
                        # Check if the detected class is a weed
                        if self.is_weed_class(class_name):
                            weed_detected = True
                            weed_count += 1
                            max_confidence = max(max_confidence, confidence)
                            print(f"Weed detected: {class_name}")
                        
                        # Create label for annotation
                        labels.append(f"{self.model.names[class_id]}: {confidence:.2f}")
                        
                    except Exception as detection_error:
                        print(f"Error processing detection {i}: {detection_error}")
                        continue
            
            # Annotate image
            try:
                annotated_image = image.copy()
                
                if len(detections) > 0:
                    # Convert PIL to numpy array for supervision
                    annotated_array = np.array(annotated_image)
                    
                    # Apply annotations
                    annotated_array = self.box_annotator.annotate(
                        scene=annotated_array, 
                        detections=detections
                    )
                    annotated_array = self.label_annotator.annotate(
                        scene=annotated_array, 
                        detections=detections, 
                        labels=labels
                    )
                    
                    # Convert back to PIL
                    annotated_image = Image.fromarray(annotated_array)
                    print("Image annotation completed")
                
            except Exception as annot_error:
                print(f"Error during annotation: {annot_error}")
                # Return original image if annotation fails
                annotated_image = image
            
            print(f"Detection summary: {weed_count} weeds found, max confidence: {max_confidence:.2f}")
            return annotated_image, weed_detected, weed_count, max_confidence
            
        except Exception as e:
            print(f"Unexpected error in detect_weeds: {e}")
            print(f"Error type: {type(e).__name__}")
            return None, False, 0, 0.0
    
    def is_weed_class(self, class_name):
        """Check if a class name represents a weed"""
        weed_keywords = ['weed', 'unwanted', 'invasive', 'pest', 'grass']
        class_name_lower = class_name.lower()
        is_weed = any(keyword in class_name_lower for keyword in weed_keywords)
        print(f"Checking if '{class_name}' is weed: {is_weed}")
        return is_weed
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return {
                "status": "Model not loaded",
                "error": "Model initialization failed"
            }
        
        try:
            return {
                "model_type": "YOLO",
                "model_path": self.model_path,
                "classes": self.model.names if hasattr(self.model, 'names') else "Unknown",
                "confidence_threshold": self.confidence_threshold,
                "total_classes": len(self.model.names) if hasattr(self.model, 'names') and self.model.names else 0,
                "status": "Loaded successfully"
            }
        except Exception as e:
            return {
                "status": "Error getting model info",
                "error": str(e)
            }
    
    def update_confidence_threshold(self, new_threshold):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.1, min(1.0, new_threshold))
        print(f"Confidence threshold updated to: {self.confidence_threshold}")
    
    def get_class_names(self):
        """Get all class names from the model"""
        if self.model is None or not hasattr(self.model, 'names'):
            return []
        return list(self.model.names.values()) if self.model.names else []

# Test function for debugging
def test_detector(model_path="best.pt", image_path=None):
    """
    Test the detector with a sample image
    """
    print("="*50)
    print("Testing YOLOv11 Weed Detector")
    print("="*50)
    
    detector = YOLOv11WeedDetector(model_path)
    
    if detector.model is None:
        print("❌ Failed to load model")
        return False
    
    print("✅ Model loaded successfully")
    print("Model Info:", detector.get_model_info())
    print("Available classes:", detector.get_class_names())
    
    if image_path and os.path.exists(image_path):
        try:
            print(f"\nTesting with image: {image_path}")
            annotated_image, weed_detected, count, confidence = detector.detect_weeds(image_path)
            
            if annotated_image:
                print(f"✅ Detection completed:")
                print(f"   - Weeds detected: {weed_detected}")
                print(f"   - Weed count: {count}")
                print(f"   - Max confidence: {confidence:.2f}")
                
                # Save result
                output_path = "detection_result.jpg"
                annotated_image.save(output_path)
                print(f"   - Result saved as: {output_path}")
                return True
            else:
                print("❌ Detection failed")
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    else:
        print("⚠️ No test image provided or image not found")
        return True

if __name__ == "__main__":
    # Test the detector
    success = test_detector()
    if success:
        print("\n✅ Detector test completed successfully!")
    else:
        print("\n❌ Detector test failed!")
