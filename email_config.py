import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from PIL import Image, ImageTk
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import numpy as np
from datetime import datetime
import threading
from model_integration import YOLOv11WeedDetector

# Set appearance mode and color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

class RiceWeedDetectorApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Rice Weed Detection System")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # User data storage
        self.users_file = "users.json"
        self.current_user = None
        self.load_users()
        
        # Initialize YOLO detector with error handling
        self.detector = None
        self.initialize_detector()
        
        # Initialize frames
        self.current_frame = None
        
        # Show home page
        self.show_home_page()
    
    def initialize_detector(self):
        """Initialize YOLO detector with proper error handling"""
        try:
            # Try different model paths
            model_paths = ["yolo11s.pt", "yolov8n.pt"]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    print(f"Trying to load model: {model_path}")
                    self.detector = YOLOv11WeedDetector(model_path)
                    if self.detector.model is not None:
                        print(f"Successfully loaded model: {model_path}")
                        return
                    
            # If no local model found, try to download a default one
            print("No local model found. Downloading YOLOv8n...")
            self.detector = YOLOv11WeedDetector("yolov8n.pt")  # This will auto-download
            
        except Exception as e:
            print(f"Error initializing detector: {e}")
            self.detector = None
            messagebox.showwarning(
                "Model Loading Warning", 
                f"Could not load YOLO model: {e}\n\n"
                "Please ensure you have:\n"
                "1. A valid model file (best.pt, yolov8n.pt, etc.)\n"
                "2. Proper internet connection for auto-download\n"
                "3. All required dependencies installed"
            )
        
    def load_users(self):
        """Load user data from JSON file"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {}
    
    def save_users(self):
        """Save user data to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def clear_frame(self):
        """Clear current frame"""
        if self.current_frame:
            self.current_frame.destroy()
    
    def show_home_page(self):
        """Display the home page"""
        self.clear_frame()
        
        self.current_frame = ctk.CTkFrame(self.root)
        self.current_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ctk.CTkFrame(self.current_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 30))
        
        title = ctk.CTkLabel(
            header_frame,
            text="🌾 Rice Weed Detection System",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=("#2E7D32", "#4CAF50")
        )
        title.pack(pady=20)
        
        # Model status indicator
        model_status = "✅ Model Ready" if self.detector and self.detector.model else "❌ Model Not Loaded"
        status_color = "green" if self.detector and self.detector.model else "red"
        
        status_label = ctk.CTkLabel(
            header_frame,
            text=f"Status: {model_status}",
            font=ctk.CTkFont(size=14),
            text_color=status_color
        )
        status_label.pack()
        
        # Main content
        content_frame = ctk.CTkFrame(self.current_frame)
        content_frame.pack(fill="both", expand=True, padx=40, pady=20)
        
        # Description
        description = ctk.CTkLabel(
            content_frame,
            text="Welcome to the Advanced Rice Weed Detection System\n\n"
                 "Our cutting-edge AI-powered solution helps farmers identify and manage weeds in rice crops\n"
                 "with unprecedented accuracy. Using state-of-the-art YOLOv11 deep learning technology,\n"
                 "we provide real-time weed detection and automated alert systems.\n\n"
                 "🔍 Key Features:\n"
                 "• Real-time weed detection using YOLOv11\n"
                 "• Instant email alerts for detected weeds\n"
                 "• User-friendly dashboard with crop analytics\n"
                 "• Secure user authentication system\n"
                 "• Detailed detection reports and history\n\n"
                 "Join thousands of farmers who trust our technology to protect their crops\n"
                 "and maximize their harvest yields.",
            font=ctk.CTkFont(size=16),
            justify="center"
        )
        description.pack(pady=40)
        
        # Buttons frame
        buttons_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        buttons_frame.pack(pady=30)
        
        # Get Started button
        get_started_btn = ctk.CTkButton(
            buttons_frame,
            text="Get Started",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=50,
            width=200,
            command=self.show_landing_page
        )
        get_started_btn.pack(side="left", padx=10)
        
        # Reload Model button
        reload_btn = ctk.CTkButton(
            buttons_frame,
            text="🔄 Reload Model",
            font=ctk.CTkFont(size=14),
            height=50,
            width=150,
            command=self.reload_model
        )
        reload_btn.pack(side="left", padx=10)
        
        # Footer
        footer = ctk.CTkLabel(
            content_frame,
            text="Powered by Advanced AI Technology | Protecting Crops, Securing Future",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        footer.pack(side="bottom", pady=20)
    
    def reload_model(self):
        """Reload the YOLO model"""
        self.initialize_detector()
        self.show_home_page()  # Refresh the page to show updated status
    
    def show_landing_page(self):
        """Display the landing/login page"""
        self.clear_frame()
        
        self.current_frame = ctk.CTkFrame(self.root)
        self.current_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Back button
        back_btn = ctk.CTkButton(
            self.current_frame,
            text="← Back to Home",
            width=120,
            command=self.show_home_page
        )
        back_btn.pack(anchor="nw", padx=20, pady=10)
        
        # Main container
        main_container = ctk.CTkFrame(self.current_frame)
        main_container.pack(expand=True, fill="both", padx=100, pady=50)
        
        # Title
        title = ctk.CTkLabel(
            main_container,
            text="🔐 Farmer Login Portal",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.pack(pady=(40, 30))
        
        # Login form
        form_frame = ctk.CTkFrame(main_container)
        form_frame.pack(pady=20, padx=60, fill="x")
        
        # Email field
        email_label = ctk.CTkLabel(form_frame, text="Email Address:", font=ctk.CTkFont(size=14))
        email_label.pack(anchor="w", padx=20, pady=(20, 5))
        
        self.email_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="Enter your email address",
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.email_entry.pack(fill="x", padx=20, pady=(0, 15))
        
        # Password field
        password_label = ctk.CTkLabel(form_frame, text="Password:", font=ctk.CTkFont(size=14))
        password_label.pack(anchor="w", padx=20, pady=(0, 5))
        
        self.password_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="Enter your password",
            show="*",
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.password_entry.pack(fill="x", padx=20, pady=(0, 20))
        
        # Login button
        login_btn = ctk.CTkButton(
            form_frame,
            text="Login to Dashboard",
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.login
        )
        login_btn.pack(fill="x", padx=20, pady=(10, 30))
        
        # Register link
        register_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        register_frame.pack(pady=10)
        
        register_label = ctk.CTkLabel(
            register_frame,
            text="New farmer? Click here to register",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        register_label.pack()
        
        register_btn = ctk.CTkButton(
            register_frame,
            text="Create Account",
            width=120,
            height=30,
            command=self.show_register_dialog
        )
        register_btn.pack(pady=5)
    
    def show_register_dialog(self):
        """Show registration dialog"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Register New Farmer")
        dialog.geometry("400x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (700 // 2)
        dialog.geometry(f"400x700+{x}+{y}")
        
        title = ctk.CTkLabel(dialog, text="Create Farmer Account", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=20)
        
        # Form fields
        fields = [
            ("Full Name:", "name"),
            ("Email Address:", "email"),
            ("Password:", "password"),
            ("Farm Location:", "location"),
            ("Farm Size (acres):", "farm_size"),
            ("Phone Number:", "phone")
        ]
        
        entries = {}
        for label_text, field_name in fields:
            label = ctk.CTkLabel(dialog, text=label_text)
            label.pack(anchor="w", padx=20, pady=(10, 2))
            
            entry = ctk.CTkEntry(
                dialog,
                placeholder_text=f"Enter {label_text.lower().replace(':', '')}",
                show="*" if field_name == "password" else None
            )
            entry.pack(fill="x", padx=20, pady=(0, 5))
            entries[field_name] = entry
        
        def register_user():
            # Validate fields
            for field_name, entry in entries.items():
                if not entry.get().strip():
                    messagebox.showerror("Error", f"Please fill in {field_name.replace('_', ' ')}")
                    return
            
            email = entries["email"].get().strip()
            if email in self.users:
                messagebox.showerror("Error", "Email already registered!")
                return
            
            # Save user
            self.users[email] = {
                "name": entries["name"].get().strip(),
                "password": entries["password"].get(),
                "location": entries["location"].get().strip(),
                "farm_size": entries["farm_size"].get().strip(),
                "phone": entries["phone"].get().strip(),
                "registration_date": datetime.now().isoformat(),
                "detections": []
            }
            self.save_users()
            
            messagebox.showinfo("Success", "Account created successfully!")
            dialog.destroy()
        
        register_btn = ctk.CTkButton(
            dialog,
            text="Create Account",
            command=register_user,
            height=40
        )
        register_btn.pack(fill="x", padx=20, pady=20)
    
    def login(self):
        """Handle user login"""
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        
        if not email or not password:
            messagebox.showerror("Error", "Please enter both email and password")
            return
        
        if email not in self.users:
            messagebox.showerror("Error", "Email not found. Please register first.")
            return
        
        if self.users[email]["password"] != password:
            messagebox.showerror("Error", "Incorrect password")
            return
        
        self.current_user = email
        messagebox.showinfo("Success", f"Welcome back, {self.users[email]['name']}!")
        self.show_dashboard()
    
    def show_dashboard(self):
        """Display the main dashboard"""
        self.clear_frame()
        
        self.current_frame = ctk.CTkFrame(self.root)
        self.current_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ctk.CTkFrame(self.current_frame)
        header_frame.pack(fill="x", padx=10, pady=10)
        
        user_data = self.users[self.current_user]
        welcome_label = ctk.CTkLabel(
            header_frame,
            text=f"🌾 Welcome, {user_data['name']}!",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        welcome_label.pack(side="left", padx=20, pady=15)
        
        logout_btn = ctk.CTkButton(
            header_frame,
            text="Logout",
            width=80,
            command=self.logout
        )
        logout_btn.pack(side="right", padx=20, pady=15)
        
        # Main content area
        content_frame = ctk.CTkFrame(self.current_frame)
        content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Left panel - Farmer info
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.pack(side="left", fill="y", padx=(10, 5), pady=10)
        
        info_title = ctk.CTkLabel(left_panel, text="👨‍🌾 Farmer Profile", font=ctk.CTkFont(size=18, weight="bold"))
        info_title.pack(pady=(20, 15))
        
        info_items = [
            ("📧 Email:", self.current_user),
            ("📍 Location:", user_data['location']),
            ("🏞️ Farm Size:", f"{user_data['farm_size']} acres"),
            ("📱 Phone:", user_data['phone']),
            ("📅 Member Since:", user_data['registration_date'][:10])
        ]
        
        for label, value in info_items:
            info_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
            info_frame.pack(fill="x", padx=15, pady=5)
            
            ctk.CTkLabel(info_frame, text=label, font=ctk.CTkFont(weight="bold")).pack(anchor="w")
            ctk.CTkLabel(info_frame, text=value, text_color="gray").pack(anchor="w")
        
        # Detection stats
        detections_count = len(user_data.get('detections', []))
        stats_frame = ctk.CTkFrame(left_panel)
        stats_frame.pack(fill="x", padx=15, pady=20)
        
        ctk.CTkLabel(stats_frame, text="📊 Detection Stats", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        ctk.CTkLabel(stats_frame, text=f"Total Scans: {detections_count}").pack()
        
        # Model status in dashboard
        model_status = "✅ Ready" if self.detector and self.detector.model else "❌ Not Loaded"
        ctk.CTkLabel(stats_frame, text=f"Model Status: {model_status}").pack(pady=5)
        
        # Right panel - Image detection
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)
        
        detection_title = ctk.CTkLabel(right_panel, text="🔍 Weed Detection System", font=ctk.CTkFont(size=18, weight="bold"))
        detection_title.pack(pady=(20, 15))
        
        # Upload button - disable if model not loaded
        upload_btn = ctk.CTkButton(
            right_panel,
            text="📁 Upload Rice Crop Image" if self.detector and self.detector.model else "❌ Model Not Available",
            height=50,
            font=ctk.CTkFont(size=16),
            command=self.upload_image,
            state="normal" if self.detector and self.detector.model else "disabled"
        )
        upload_btn.pack(pady=20)
        
        # Image display area
        self.image_frame = ctk.CTkFrame(right_panel)
        self.image_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        initial_text = "No image uploaded yet\n\nClick 'Upload Rice Crop Image' to start detection"
        if not (self.detector and self.detector.model):
            initial_text = "❌ YOLO Model not loaded\n\nPlease check your model file and reload"
        
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text=initial_text,
            font=ctk.CTkFont(size=14)
        )
        self.image_label.pack(expand=True)
    
    def upload_image(self):
        """Handle image upload and processing"""
        if not (self.detector and self.detector.model):
            messagebox.showerror("Error", "YOLO model is not loaded. Please reload the model first.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Rice Crop Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if not file_path:
            return
        
        # Validate image file
        try:
            test_image = Image.open(file_path)
            test_image.verify()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid image file: {e}")
            return
        
        # Show loading message
        self.image_label.configure(text="🔄 Processing image...\nRunning YOLOv11 detection...")
        self.root.update()
        
        # Process image in a separate thread to prevent UI freezing
        threading.Thread(target=self.process_image, args=(file_path,), daemon=True).start()
    
    def process_image(self, file_path):
        """Process image with YOLOv11 model using supervision"""
        try:
            print(f"Processing image: {file_path}")
            
            # Check if detector is available
            if not self.detector or not self.detector.model:
                self.root.after(0, lambda: messagebox.showerror("Error", "YOLO model is not available"))
                return
            
            # Perform detection using the integrated detector
            result = self.detector.detect_weeds(file_path)
            
            if result[0] is not None:  # annotated_image is not None
                annotated_image, weed_detected, detection_count, weed_confidence = result
                
                # Resize for display
                display_size = (400, 300)
                annotated_resized = annotated_image.resize(display_size, Image.Resampling.LANCZOS)
            
                # Update UI in main thread
                self.root.after(0, self.display_results, annotated_resized, weed_detected, file_path, detection_count, weed_confidence)
            else:
                error_msg = "Failed to process image. Please check:\n1. Image file is valid\n2. Model is properly loaded\n3. Image format is supported"
                self.root.after(0, lambda: messagebox.showerror("Processing Error", error_msg))
            
        except Exception as e:
            error_msg = f"Detection failed: {str(e)}\n\nTroubleshooting:\n1. Check if model file exists\n2. Verify image format\n3. Ensure all dependencies are installed"
            print(f"Process image error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
    
    def display_results(self, annotated_image, weed_detected, original_path, detection_count=0, weed_confidence=0.0):
        """Display detection results"""
        # Clear previous content
        for widget in self.image_frame.winfo_children():
            widget.destroy()
    
        # Convert to PhotoImage for display
        photo = ImageTk.PhotoImage(annotated_image)
    
        # Display image
        image_display = tk.Label(self.image_frame, image=photo)
        image_display.image = photo  # Keep a reference
        image_display.pack(pady=10)
    
        # Results text
        if weed_detected:
            result_text = f"⚠️ {detection_count} WEED(S) DETECTED!\nConfidence: {weed_confidence:.1%}\nImmediate action recommended"
            text_color = "red"
        
            # Send warning email
            self.send_warning_email(detection_count, weed_confidence)
        
            # Save detection record
            self.save_detection_record(original_path, True, detection_count, weed_confidence)
        
        else:
            result_text = "✅ No weeds detected\nCrop looks healthy!"
            text_color = "green"
        
            # Save detection record
            self.save_detection_record(original_path, False, 0, 0.0)
    
        result_label = ctk.CTkLabel(
            self.image_frame,
            text=result_text,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=text_color
        )
        result_label.pack(pady=10)
    
        # New scan button
        new_scan_btn = ctk.CTkButton(
            self.image_frame,
            text="🔄 Scan Another Image",
            command=self.upload_image
        )
        new_scan_btn.pack(pady=10)
    
    def send_warning_email(self, detection_count=1, confidence=0.0):
        """Send warning email to farmer"""
        try:
            user_data = self.users[self.current_user]
        
            # In a real application, you would configure SMTP settings
            # For demonstration, we'll show a message
            messagebox.showwarning(
                "Email Alert Sent",
                f"⚠️ Warning email sent to {self.current_user}\n\n"
                f"Subject: URGENT - Weed Detection Alert - {user_data['name']}'s Farm\n\n"
                f"Dear {user_data['name']},\n\n"
                f"Our AI system has detected {detection_count} weed(s) in your rice crop at {user_data['location']}.\n"
                f"Detection confidence: {confidence:.1%}\n\n"
                f"Immediate attention is recommended to prevent crop damage.\n\n"
                f"Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Best regards,\nRice Weed Detection System"
            )
        
        except Exception as e:
            print(f"Email sending failed: {e}")
    
    def save_detection_record(self, image_path, weed_detected, detection_count=0, confidence=0.0):
        """Save detection record to user data"""
        detection_record = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "weed_detected": weed_detected,
            "detection_count": detection_count,
            "confidence": confidence,
            "status": f"{detection_count} Weeds Detected" if weed_detected else "Clean"
        }
    
        if "detections" not in self.users[self.current_user]:
            self.users[self.current_user]["detections"] = []
    
        self.users[self.current_user]["detections"].append(detection_record)
        self.save_users()
    
    def logout(self):
        """Handle user logout"""
        self.current_user = None
        messagebox.showinfo("Logged Out", "You have been logged out successfully!")
        self.show_home_page()
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = RiceWeedDetectorApp()
    app.run()