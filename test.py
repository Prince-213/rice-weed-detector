import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime
import threading
import base64
from model_integration import YOLOv8WeedDetector
import resend

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
        
        # Webcam variables
        self.cap = None
        self.webcam_running = False
        self.current_webcam_frame = None
        self.snapshot_mode = False
        self.realtime_mode = False
        self.last_alert_time = None
        
        # Initialize frames
        self.current_frame = None
        
        # Show home page
        self.show_home_page()
        
        resend.api_key = "re_Lw6fGjzZ_3WtN5JNbALdC92Jigy686CkA"
    
    def initialize_detector(self):
        """Initialize YOLO detector with proper error handling"""
        try:
            model_paths = ["best.pt"]
            for model_path in model_paths:
                if os.path.exists(model_path):
                    print(f"Trying to load model: {model_path}")
                    self.detector = YOLOv8WeedDetector(model_path)
                    if self.detector.model is not None:
                        print(f"Successfully loaded model: {model_path}")
                        return
                    
            print("No local model found. Downloading best.pt...")
            self.detector = YOLOv8WeedDetector("best.pt")

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
                 "with unprecedented accuracy. Using state-of-the-art YOLOv8n deep learning technology,\n"
                 "we provide real-time weed detection and automated alert systems.\n\n"
                 "🔍 Key Features:\n"
                 "• Real-time weed detection using YOLOv8n\n"
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
    
    def send_resend_email(self, frame, confidence):
        """Send email alert using Resend API"""
        user_data = self.users[self.current_user]
        
        # Convert PIL Image to numpy array if needed
        if hasattr(frame, 'mode') and hasattr(frame, 'size'):
            frame = np.array(frame)
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame.dtype == np.float32:
            frame = frame.astype('uint8')
        
        try:
            # Encode the image to JPEG in memory
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            params = {
                "from": "Rice Weed Alert <no-reply@kargoxlogistics.com>",
                "to": [self.current_user],
                "subject": f"RICE WEED ALERT - {self.current_user}",
                "html": f"""
                <h2>Dear {user_data['name']}</h2>
                <p>Our AI system has detected weed(s) in your rice crop at {user_data['location']}.</p>
                <p>Weed Severity: {confidence:.1%}</p>
                <p>Immediate attention is recommended to prevent crop damage.</p>
                <p>Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Please check the attached image and take appropriate action.</p>
                """,
                "attachments": [{
                    "filename": "weed_detection.jpg",
                    "content": jpg_as_text
                }]
            }

            response = resend.Emails.send(params)
            print("Email sent:", response)
            
        except Exception as e:
            error_msg = f"Failed to send email: {e}"
            print(error_msg)
    
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
        
        # Detection mode selection
        mode_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        mode_frame.pack(pady=10)
        
        # Upload button - disable if model not loaded
        upload_btn = ctk.CTkButton(
            mode_frame,
            text="📁 Upload Image",
            height=40,
            font=ctk.CTkFont(size=14),
            command=self.upload_image,
            state="normal" if self.detector and self.detector.model else "disabled"
        )
        upload_btn.pack(side="left", padx=5)
        
        # Webcam snapshot button
        webcam_btn = ctk.CTkButton(
            mode_frame,
            text="📷 Take Snapshot",
            height=40,
            font=ctk.CTkFont(size=14),
            command=self.start_webcam_snapshot,
            state="normal" if self.detector and self.detector.model else "disabled"
        )
        webcam_btn.pack(side="left", padx=5)
        
        # Realtime detection button
        realtime_btn = ctk.CTkButton(
            mode_frame,
            text="🎥 Realtime Detection",
            height=40,
            font=ctk.CTkFont(size=14),
            command=self.toggle_realtime_detection,
            state="normal" if self.detector and self.detector.model else "disabled"
        )
        realtime_btn.pack(side="left", padx=5)
        
        # Image display area
        self.image_frame = ctk.CTkFrame(right_panel)
        self.image_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        initial_text = "No image uploaded yet\n\nSelect a detection mode to begin"
        if not (self.detector and self.detector.model):
            initial_text = "❌ YOLO Model not loaded\n\nPlease check your model file and reload"
        
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text=initial_text,
            font=ctk.CTkFont(size=14)
        )
        self.image_label.pack(expand=True)
        
        # Webcam control frame (initially hidden)
        self.webcam_controls = ctk.CTkFrame(right_panel, fg_color="transparent")
        self.webcam_controls.pack_forget()
        
        self.capture_btn = ctk.CTkButton(
            self.webcam_controls,
            text="Capture Image",
            command=self.capture_webcam_image,
            state="disabled"
        )
        self.capture_btn.pack(side="left", padx=5)
        
        self.stop_webcam_btn = ctk.CTkButton(
            self.webcam_controls,
            text="Stop Webcam",
            command=self.stop_webcam,
            fg_color="red",
            hover_color="darkred"
        )
        self.stop_webcam_btn.pack(side="left", padx=5)
    
    def start_webcam_snapshot(self):
        """Start webcam in snapshot mode"""
        self.snapshot_mode = True
        self.realtime_mode = False
        self.start_webcam()
        
    def toggle_realtime_detection(self):
        """Toggle realtime detection mode"""
        if self.realtime_mode:
            self.stop_webcam()
            self.realtime_mode = False
        else:
            self.snapshot_mode = False
            self.realtime_mode = True
            self.start_webcam()
    
    def start_webcam(self):
        """Initialize and start webcam capture"""
        if self.webcam_running:
            return
            
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
                
            self.webcam_running = True
            self.webcam_controls.pack(pady=10)
            
            if self.snapshot_mode:
                self.capture_btn.configure(state="normal")
                self.image_label.configure(text="Webcam live view - Click 'Capture Image' when ready")
            else:
                self.image_label.configure(text="Realtime detection running...")
            
            # Start webcam update thread
            threading.Thread(target=self.update_webcam, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Webcam Error", f"Failed to initialize webcam: {str(e)}")
    
    def stop_webcam(self):
        """Stop webcam capture"""
        self.webcam_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.webcam_controls.pack_forget()
        self.snapshot_mode = False
        self.realtime_mode = False
        
        # Reset image label if no image was captured
        if not hasattr(self, 'current_webcam_frame'):
            self.image_label.configure(text="Webcam stopped\n\nSelect a detection mode to begin")
    
    def update_webcam(self):
        """Update webcam feed in a separate thread"""
        while self.webcam_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Store the current frame
            self.current_webcam_frame = frame.copy()
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Process frame in realtime mode
            if self.realtime_mode and self.detector and self.detector.model:
                try:
                    # Convert to numpy array for processing
                    img_np = np.array(img)
                    
                    # Perform detection using the standard detect_weeds method
                    temp_path = "temp_realtime.jpg"
                    cv2.imwrite(temp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                    result = self.detector.detect_weeds(temp_path)
                    os.remove(temp_path)
                    
                    if result[0] is not None:
                        annotated_img, weed_detected, detection_count, weed_confidence = result
                        img = annotated_img
                        
                        # If weeds detected, send alert (throttled)
                        if weed_detected:
                            current_time = datetime.now()
                            if (self.last_alert_time is None or 
                                (current_time - self.last_alert_time).seconds > 60):
                                self.last_alert_time = current_time
                                threading.Thread(
                                    target=self.send_resend_email,
                                    args=(annotated_img, weed_confidence),
                                    daemon=True
                                ).start()
                
                except Exception as e:
                    print(f"Realtime detection error: {e}")
            
            # Resize for display
            img = img.resize((500, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Update UI in main thread
            self.root.after(0, self.update_webcam_display, photo)
            
        # Clean up when loop ends
        self.root.after(0, self.stop_webcam)

    def update_webcam_display(self, photo):
        """Update the webcam display in the main thread"""
        # Clear previous content safely
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        # Create a new label for the image
        self.image_display = tk.Label(self.image_frame)
        self.image_display.pack()
        
        # Update the image
        self.image_display.configure(image=photo)
        self.image_display.image = photo  # Keep a reference

    def capture_webcam_image(self):
        """Capture and process the current webcam frame"""
        if self.current_webcam_frame is None:
            return
            
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(self.current_webcam_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Clear the current display and show processing message
        for widget in self.image_frame.winfo_children():
            widget.destroy()
            
        processing_label = ctk.CTkLabel(
            self.image_frame,
            text="🔄 Processing captured image...",
            font=ctk.CTkFont(size=14)
        )
        processing_label.pack(expand=True)
        self.root.update()
        
        # Process in a separate thread
        threading.Thread(
            target=self.process_captured_image,
            args=(pil_img,),
            daemon=True
        ).start()

    def process_captured_image(self, pil_img):
        """Process a captured webcam image"""
        try:
            # Save to temporary file
            temp_path = "temp_webcam_capture.jpg"
            pil_img.save(temp_path)
            
            # Process the image
            result = self.detector.detect_weeds(temp_path)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
                
            if result[0] is not None:
                annotated_image, weed_detected, detection_count, weed_confidence = result
                
                # Resize for display
                display_size = (500, 500)
                annotated_resized = annotated_image.resize(display_size, Image.Resampling.LANCZOS)
                
                # Update UI in main thread
                self.root.after(0, self.display_results, annotated_resized, weed_detected, "webcam_capture", detection_count, weed_confidence)
                
        except Exception as e:
            error_msg = f"Failed to process captured image: {str(e)}"
            print(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Processing Error", error_msg))

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
        self.image_label.configure(text="🔄 Processing image...\nRunning YOLOv8n detection...")
        self.root.update()
        
        # Process image in a separate thread to prevent UI freezing
        threading.Thread(target=self.process_image, args=(file_path,), daemon=True).start()
    
    def process_image(self, file_path):
        """Process image with YOLOv8 model using supervision"""
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
                display_size = (500, 500)
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

            email_thread = threading.Thread(target=self.send_resend_email, args=(annotated_image, weed_confidence), daemon=True)
            email_thread.start()
        
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
            command=lambda: self.upload_image() if not self.webcam_running else self.start_webcam_snapshot()
        )
        new_scan_btn.pack(pady=10)
    
    def send_warning_email(self, detection_count=1, confidence=0.0):
        """Send warning email to farmer"""
        try:
            user_data = self.users[self.current_user]
        
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
            "image_path": image_path if isinstance(image_path, str) else "webcam_capture",
            "weed_detected": bool(weed_detected),
            "detection_count": int(detection_count),
            "confidence": float(confidence),  # Convert numpy float32 to Python float
            "status": f"{int(detection_count)} Weeds Detected" if weed_detected else "Clean"
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
        
        # Ensure webcam is released when app closes
        if self.cap is not None:
            self.cap.release()

if __name__ == "__main__":
    app = RiceWeedDetectorApp()
    app.run()