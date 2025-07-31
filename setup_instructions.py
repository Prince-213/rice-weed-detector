"""
Setup Instructions for Rice Weed Detection App

1. Install Required Packages:
   pip install -r requirements.txt
   pip install ultralytics  # For YOLOv11

2. Model Setup:
   - Place your trained YOLOv11 model file (e.g., 'best.pt') in the project directory
   - Update the model path in model_integration.py

3. Email Configuration:
   - Update email_config.py with your SMTP settings
   - For Gmail, use app passwords instead of regular passwords
   - Enable 2-factor authentication and generate an app password

4. Directory Structure:
   rice_weed_detector/
   ├── main.py                 # Main application file
   ├── model_integration.py    # YOLOv11 integration
   ├── email_config.py        # Email notification setup
   ├── requirements.txt       # Python dependencies
   ├── best.pt               # Your trained YOLOv11 model
   └── users.json            # User data (auto-generated)

5. Running the Application:
   python main.py

6. Features:
   - Beautiful CustomTkinter GUI
   - User registration and authentication
   - Farmer dashboard with profile information
   - Image upload and YOLOv11 weed detection
   - Email alerts for detected weeds
   - Detection history tracking

7. Customization:
   - Modify colors and themes in main.py
   - Adjust detection confidence threshold in model_integration.py
   - Customize email templates in email_config.py
   - Add more farmer profile fields as needed

8. Security Notes:
   - Store passwords securely (consider hashing)
   - Use environment variables for sensitive data
   - Implement proper input validation
   - Consider database storage for production use
"""

print("Rice Weed Detection App Setup Instructions")
print("==========================================")
print("Please read the setup_instructions.py file for detailed setup steps.")
