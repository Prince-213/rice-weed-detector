"""
Enhanced Email Configuration for Warning Alerts
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import os

class EmailNotifier:
    def __init__(self):
        # Configure your SMTP settings here
        self.smtp_server = "smtp.gmail.com"  # For Gmail
        self.smtp_port = 587
        self.sender_email = "your_email@gmail.com"  # Your email
        self.sender_password = "your_app_password"  # App password for Gmail
        
    def send_weed_alert(self, farmer_email, farmer_name, farm_location, detection_count=1, confidence=0.0, image_path=None):
        """
        Send enhanced weed detection alert email
        
        Args:
            farmer_email (str): Farmer's email address
            farmer_name (str): Farmer's name
            farm_location (str): Farm location
            detection_count (int): Number of weeds detected
            confidence (float): Detection confidence
            image_path (str): Path to annotated image (optional)
        """
        try:
            # Create message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = farmer_email
            message["Subject"] = f"🚨 URGENT: {detection_count} Weed(s) Detected - {farmer_name}'s Farm"
            
            # Email body with detection details
            body = f"""
            Dear {farmer_name},

            🚨 WEED DETECTION ALERT 🚨

            Our AI-powered Rice Weed Detection System has identified weeds in your rice crop.

            📊 DETECTION DETAILS:
            📍 Farm Location: {farm_location}
            🕐 Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            🔍 Weeds Detected: {detection_count}
            📈 Confidence Level: {confidence:.1%}
            ⚠️  Status: IMMEDIATE ATTENTION REQUIRED

            🎯 RECOMMENDED ACTIONS:
            • Inspect the affected areas immediately
            • Apply appropriate weed control measures
            • Monitor crop health closely over the next few days
            • Consider consulting with agricultural experts
            • Document the affected areas for future reference

            ⏰ URGENCY LEVEL: HIGH
            Early intervention is crucial to prevent crop damage and ensure optimal yield.
            Weeds can spread rapidly and compete with rice plants for nutrients and water.

            📞 SUPPORT:
            If you need assistance with weed identification or treatment recommendations,
            please contact our agricultural support team.

            Best regards,
            Rice Weed Detection System Team
            🌾 Protecting Your Crops with AI Technology

            ---
            This is an automated alert from your Rice Weed Detection System.
            Detection powered by YOLOv11 AI technology.
            """
            
            message.attach(MIMEText(body, "plain"))
            
            # Attach annotated image if provided
            if image_path and os.path.exists(image_path):
                try:
                    with open(image_path, 'rb') as f:
                        img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', 'attachment', filename='weed_detection_result.jpg')
                    message.attach(image)
                except Exception as e:
                    print(f"Failed to attach image: {e}")
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
                
            print(f"Alert email sent successfully to {farmer_email}")
            return True
            
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False
    
    def send_test_email(self, test_email):
        """Send a test email to verify configuration"""
        try:
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = test_email
            message["Subject"] = "🧪 Rice Weed Detection System - Test Email"
            
            body = """
            This is a test email from your Rice Weed Detection System.
            
            If you received this email, your email configuration is working correctly!
            
            Best regards,
            Rice Weed Detection System
            """
            
            message.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
                
            print("Test email sent successfully!")
            return True
            
        except Exception as e:
            print(f"Test email failed: {e}")
            return False

# Usage example
if __name__ == "__main__":
    notifier = EmailNotifier()
    # Test the email configuration
    # notifier.send_test_email("test@example.com")
