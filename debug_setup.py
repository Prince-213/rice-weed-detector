"""
Debug and Setup Helper Script
Run this to diagnose issues with your setup
"""

import os
import sys
import subprocess
import importlib

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking Dependencies...")
    print("="*50)
    
    required_packages = [
        'customtkinter',
        'PIL',
        'cv2',
        'numpy',
        'ultralytics',
        'supervision',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'cv2':
                importlib.import_module('cv2')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def check_model_files():
    """Check for available model files"""
    print("\n🔍 Checking Model Files...")
    print("="*50)
    
    model_files = ['best.pt', 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']
    found_models = []
    
    for model in model_files:
        if os.path.exists(model):
            size = os.path.getsize(model) / (1024*1024)  # Size in MB
            print(f"✅ {model} ({size:.1f} MB)")
            found_models.append(model)
        else:
            print(f"❌ {model} - NOT FOUND")
    
    if not found_models:
        print("\n⚠️ No model files found!")
        print("Solutions:")
        print("1. Place your trained 'best.pt' file in this directory")
        print("2. Or run the app - it will auto-download yolov8n.pt")
        return False
    else:
        print(f"\n✅ Found {len(found_models)} model file(s)")
        return True

def test_yolo_basic():
    """Test basic YOLO functionality"""
    print("\n🔍 Testing YOLO Basic Functionality...")
    print("="*50)
    
    try:
        from ultralytics import YOLO
        print("✅ ultralytics import successful")
        
        # Try to load a model
        try:
            model = YOLO('yolov8n.pt')  # This will download if not present
            print("✅ YOLO model loading successful")
            print(f"   Model classes: {len(model.names) if model.names else 0}")
            return True
        except Exception as model_error:
            print(f"❌ YOLO model loading failed: {model_error}")
            return False
            
    except ImportError as e:
        print(f"❌ ultralytics import failed: {e}")
        return False

def test_supervision():
    """Test supervision functionality"""
    print("\n🔍 Testing Supervision...")
    print("="*50)
    
    try:
        import supervision as sv
        print("✅ supervision import successful")
        
        # Test basic supervision components
        try:
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            print("✅ supervision annotators created successfully")
            return True
        except Exception as sv_error:
            print(f"❌ supervision components failed: {sv_error}")
            return False
            
    except ImportError as e:
        print(f"❌ supervision import failed: {e}")
        return False

def create_test_image():
    """Create a test image for testing"""
    print("\n🔍 Creating Test Image...")
    print("="*50)
    
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a simple test image
        img = Image.new('RGB', (640, 480), color='green')
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes to simulate a crop field
        draw.rectangle([100, 100, 200, 200], fill='brown', outline='black')
        draw.rectangle([300, 200, 400, 300], fill='yellow', outline='black')
        
        test_path = 'test_image.jpg'
        img.save(test_path)
        print(f"✅ Test image created: {test_path}")
        return test_path
        
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None

def run_full_test():
    """Run a complete test of the detection system"""
    print("\n🔍 Running Full System Test...")
    print("="*50)
    
    try:
        from model_integration import test_detector
        
        # Create test image
        test_image = create_test_image()
        if not test_image:
            print("❌ Cannot create test image")
            return False
        
        # Run detector test
        success = test_detector(image_path=test_image)
        
        if success:
            print("✅ Full system test passed!")
            return True
        else:
            print("❌ Full system test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Full test error: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("🚀 Rice Weed Detection System - Diagnostic Tool")
    print("="*60)
    
    all_good = True
    
    # Check dependencies
    if not check_dependencies():
        all_good = False
    
    # Check model files
    if not check_model_files():
        all_good = False
    
    # Test YOLO
    if not test_yolo_basic():
        all_good = False
    
    # Test supervision
    if not test_supervision():
        all_good = False
    
    # Run full test if everything looks good
    if all_good:
        run_full_test()
    
    print("\n" + "="*60)
    if all_good:
        print("🎉 System appears to be working correctly!")
        print("You can now run: python main.py")
    else:
        print("⚠️ Issues detected. Please fix the above problems.")
        print("\nCommon solutions:")
        print("1. pip install -r requirements.txt")
        print("2. Place your model file (best.pt) in this directory")
        print("3. Check internet connection for auto-downloads")

if __name__ == "__main__":
    main()
