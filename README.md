# Rice Crop Weed Detection System

A Python application for farmers that uses YOLOv8 to detect weeds in rice crop images, with user authentication and email notification features.

![Application Screenshot](Screenshot%202025-08-22%20035401.png)

## Overview

This application helps farmers identify weeds in their rice crops through computer vision technology. Farmers can create accounts, upload images of their crops, and receive email notifications with detection results.

## Features

- 👨‍🌾 User authentication system (signup/login)
- 📷 Image upload functionality for rice crop analysis
- 🤖 YOLOv8 model integration for weed detection
- 📧 Automated email notifications with detection results
- 💾 User data persistence with JSON storage
- 🎨 Simple and intuitive Tkinter GUI

## Project Structure

```
project/
├── main.py                 # Main application entry point
├── model_integration.py    # YOLOv8 model integration
├── email_config.py         # Email configuration and sending
├── users.json              # User database (created automatically)
├── requirements.txt        # Python dependencies
├── setup_instructions.py   # Setup instructions
├── debug_setup.py          # Debugging utilities
├── seperate.py             # Utility functions
├── test.py                 # Testing module
├── webcam_capture.jpg      # Example webcam capture
├── vido.mp4                # Example video file
├── extracted_frames/       # Directory for extracted video frames
├── extracted_frames.zip    # Compressed frames
├── extracted_frames.rar    # Alternative compressed frames
└── .gitignore             # Git ignore rules
```

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up email configuration in `email_config.py`:
   - Update SMTP server details
   - Configure sender email credentials

## Usage

1. Run the application:
```bash
python main.py
```

2. Create an account or login with existing credentials
3. Upload an image of rice crops for analysis
4. The system will process the image using YOLOv8 model
5. Detection results will be emailed to you with annotated image

## Model Information

The application uses a YOLOv8 model trained specifically for rice crop weed detection. The model can identify:
- Various types of weeds commonly found in rice fields
- Different growth stages of rice plants
- Potential crop health issues

## Configuration

### Email Settings
Edit `email_config.py` to set up your email service:
- SMTP server address and port
- Email account credentials
- Email message templates

### User Management
User accounts are stored in `users.json` with secure password hashing.

### Email Report Sample

<img width="1589" height="715" alt="Screenshot 2025-08-22 035633" src="https://github.com/user-attachments/assets/525cddca-d156-4839-a857-70d8e17b3616" />


## Support

For issues related to:
- Model performance: Check `model_integration.py`
- Email functionality: Review `email_config.py`
- User authentication: Check `users.json` format
- General application: Run `debug_setup.py` for diagnostics

## License

This project is intended for agricultural research and educational purposes. Please ensure proper attribution if used in other projects.

---

**Note**: This application requires proper configuration of email services to function completely. Farmers should use this tool as a supplementary aid rather than sole decision-making tool for crop management.
