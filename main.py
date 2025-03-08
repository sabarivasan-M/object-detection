import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import urllib.request
import time

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.camera = None
        self.is_camera_on = False
        
        # Load the pre-trained YOLO model
        self.load_yolo_model()
        
        # Create the UI
        self.create_ui()
        
        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_yolo_model(self):
        # Load YOLO model configuration and weights
        model_dir = "model_files"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        self.config_path = os.path.join(model_dir, "yolov3.cfg")
        self.weights_path = os.path.join(model_dir, "yolov3.weights")
        self.classes_path = os.path.join(model_dir, "coco.names")
        
        # Check if model files exist, if not, download them
        self.check_and_download_model_files(model_dir)
        
        try:
            # Load the COCO class labels
            with open(self.classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Load YOLO network
            print("Loading YOLO model...")
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            
            # Use GPU if available
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get the output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
            
            # Generate random colors for class labels
            np.random.seed(42)
            self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
            print("YOLO model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Error loading YOLO model: {str(e)}")
            raise e
    
    def check_and_download_model_files(self, model_dir):
        """Check if model files exist, if not download them"""
        files_to_check = {
            'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
            'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        }
        
        for file_name, url in files_to_check.items():
            file_path = os.path.join(model_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Downloading {file_name}...")
                try:
                    urllib.request.urlretrieve(url, file_path)
                    print(f"Downloaded {file_name} successfully!")
                except Exception as e:
                    messagebox.showerror("Download Error", 
                                        f"Failed to download {file_name}. Please download it manually from {url}")
                    print(f"Error downloading {file_name}: {e}")
                    raise e
    
    def create_ui(self):
        # Create frames
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.image_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control buttons
        tk.Button(control_frame, text="Upload Image", command=self.upload_image, 
                 bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        self.camera_button = tk.Button(control_frame, text="Start Camera", command=self.toggle_camera,
                 bg="#9C27B0", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Detect Objects", command=self.detect_objects,
                 bg="#2196F3", fg="white", font=("Arial", 12), padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Save Result", command=self.save_result,
                 bg="#FF9800", fg="white", font=("Arial", 12), padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        # Confidence threshold slider
        tk.Label(control_frame, text="Confidence:", bg="#f0f0f0", font=("Arial", 12)).pack(side=tk.LEFT, padx=(20, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        tk.Scale(control_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, 
                variable=self.confidence_var, bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        
        # Create image display labels
        self.original_img_label = tk.Label(self.image_frame, text="Upload an image to begin", 
                                          bg="#e0e0e0", font=("Arial", 14), height=20)
        self.original_img_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.processed_img_label = tk.Label(self.image_frame, text="Detection results will appear here", 
                                           bg="#e0e0e0", font=("Arial", 14), height=20)
        self.processed_img_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results text area
        self.results_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.results_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(self.results_frame, text="Detection Results:", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.results_text = tk.Text(self.results_frame, height=6, width=80, font=("Courier", 11))
        self.results_text.pack(fill=tk.X, pady=5)
        self.results_text.config(state=tk.DISABLED)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        
        if file_path:
            self.image_path = file_path
            # Load and display the original image
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.original_image, self.original_img_label)
            
            # Clear the processed image and results
            self.processed_img_label.config(image=None)
            self.processed_img_label.config(text="Click 'Detect Objects' to process")
            self.update_results_text([])
    
    def toggle_camera(self):
        if not self.is_camera_on:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
        
        self.is_camera_on = True
        self.camera_button.config(text="Stop Camera")
        self.update_camera()

    def stop_camera(self):
        self.is_camera_on = False
        self.camera_button.config(text="Start Camera")
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.original_img_label.config(image=None, text="Camera stopped")

    def update_camera(self):
        if self.is_camera_on and self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                self.original_image = frame
                self.display_image(frame, self.original_img_label)
            self.root.after(10, self.update_camera)

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

    def detect_objects(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image or start the camera first")
            return
        
        # Create a copy of the original image for drawing
        image = self.original_image.copy()
        height, width = image.shape[:2]
        
        # Preprocess the image for YOLO
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        try:
            # Run forward pass and get detections
            outputs = self.net.forward(self.output_layers)
            
            # Process the outputs
            class_ids = []
            confidences = []
            boxes = []
            
            confidence_threshold = self.confidence_var.get()
            nms_threshold = 0.4
            
            # Process detections
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > confidence_threshold:
                        # Scale the bounding box coordinates to the original image size
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            detection_results = []
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = self.classes[class_ids[i]]
                    confidence = confidences[i]
                    color = [int(c) for c in self.colors[class_ids[i]]]
                    
                    detection_results.append({
                        "class": label,
                        "confidence": confidence,
                        "box": (x, y, w, h)
                    })
                    
                    # Draw bounding box and label
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = f"{label}: {confidence:.2f}"
                    cv2.rectangle(image, (x, y - 30), (x + len(text) * 10, y), color, -1)
                    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Store and display the processed image
            self.processed_image = image
            self.display_image(image, self.processed_img_label)
            
            # Update results text
            self.update_results_text(detection_results)
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Error during object detection: {str(e)}")
            print(f"Detection error: {str(e)}")
    
    def display_image(self, cv_image, label_widget):
        if cv_image is None:
            return
        
        # Get the current width of the label for proper scaling
        max_width = label_widget.winfo_width() - 20
        max_height = label_widget.winfo_height() - 20
        
        if max_width <= 1:  # If widget not fully initialized, use default size
            max_width = 400
            max_height = 400
        
        # Resize the image to fit the label while maintaining aspect ratio
        h, w = cv_image.shape[:2]
        ratio = min(max_width/w, max_height/h)
        new_size = (int(w * ratio), int(h * ratio))
        
        resized_image = cv2.resize(cv_image, new_size)
        
        # Convert from BGR (OpenCV format) to RGB (for tkinter)
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage format
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        
        # Update the label
        label_widget.config(image=tk_image)
        label_widget.image = tk_image  # Keep a reference
        label_widget.config(text="")
    
    def update_results_text(self, detection_results):
        # Enable text widget for editing
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        if not detection_results:
            self.results_text.insert(tk.END, "No objects detected.")
        else:
            self.results_text.insert(tk.END, f"Found {len(detection_results)} objects:\n\n")
            for i, result in enumerate(detection_results, 1):
                self.results_text.insert(
                    tk.END, 
                    f"{i}. {result['class']} (Confidence: {result['confidence']:.2f})\n"
                )
        
        # Disable editing
        self.results_text.config(state=tk.DISABLED)
    
    def save_result(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("Success", "Image saved successfully")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()