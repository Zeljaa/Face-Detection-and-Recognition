import tkinter as tk
import cv2
import numpy as np

from PIL import Image, ImageTk, ImageDraw

from moduls import face_detection, face_recognition

_recognition = True #True if you want GUI to do face recognition
_faceSaving = True #True if you want detected faces to be saved
saving_path = ''


def update_frame():
    ret, frame = cap.read()
    if ret:
        # Get the size of the left_frame
        width = left_frame.winfo_width()
        height = left_frame.winfo_height()

        # Resize the video frame to match the left_frame size
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        lbl_video.imgtk = imgtk  # Keep a reference to the image
        lbl_video.configure(image=imgtk)

    lbl_video.after(10, update_frame)
    
def adjust_widgets():
    # Get the current size of the canvas
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    
    # Recalculate the circle position and size
    center_x = canvas_width / 2
    center_y = canvas_height / 2
    radius = min(canvas_width, canvas_height) / 3
    
    # Clear existing canvas items
    canvas.delete("circle")
    canvas.delete("camera")
    
    # Draw the circle
    canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, fill="white", width=2, tags="circle")
    
    # Update camera icon size
    camera_img = Image.open("camera_icon.png").resize((int(radius), int(radius)), Image.Resampling.LANCZOS)
    camera_photo = ImageTk.PhotoImage(camera_img)
    canvas.camera_photo = camera_photo
    canvas.create_image(center_x, center_y, image=camera_photo, anchor=tk.CENTER, tags="camera")

def capture_image():
    ret, frame = cap.read()
    starting_number = 0
    if ret:
        # Convert the captured frame to an image
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        
        # Detect faces in the image
        bboxes, points = face_detection(np.array(img))
        
        # Draw the bounding boxes
        draw = ImageDraw.Draw(img)
        for bbox in bboxes[0]:
            x_min, y_min, x_max, y_max = bbox
            face_img = img.crop((x_min, y_min, x_max, y_max))
            if _recognition:
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                #call face_rec to recognize face
                draw.text((x_min, y_min - 10), face_recognition(face_img), fill="red")
        # Display the processed image in a new window
            
            if _faceSaving:
                face_img.save(saving_path + '/image_{starting_number}.jpg')
                starting_number += 1
                
        img.show()
        
def on_closing():
    # Release the camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

window = tk.Tk()
window.title("Face Detection App")
# Set an initial size for the window
window.geometry("800x600")  # Adjust width and height as needed

# Initialize the camera
cap = cv2.VideoCapture(0)

# Configure window layout with weight for responsiveness
window.grid_columnconfigure(0, weight=1)  # Left frame column
window.grid_columnconfigure(1, weight=0)  # Right frame column
window.grid_rowconfigure(0, weight=1)

# Create a frame for the left side (video feed)
left_frame = tk.Frame(window)
left_frame.grid(row=0, column=0, sticky="nsew")

# Create a frame for the right side (canvas)
right_frame = tk.Frame(window, width=120)
right_frame.grid(row=0, column=1, sticky="ns")

# Create a label in the left frame to hold the video stream
lbl_video = tk.Label(left_frame)
lbl_video.pack(fill=tk.BOTH, expand=True)

# Create a Canvas widget in the right frame
canvas = tk.Canvas(right_frame, width=120,bg="black")
canvas.pack(fill=tk.BOTH, expand=True)

# Update widget positions periodically
def update_positions():
    adjust_widgets()
    window.after(100, update_positions)  # Call this function every 100ms

window.after(500, update_positions)  # Start updating positions

# Bind the circle to the capture_image function
canvas.tag_bind("circle", "<Button-1>", lambda event: capture_image())
canvas.tag_bind("camera", "<Button-1>", lambda event: capture_image())


# Bind the window close event to the on_closing function
window.protocol("WM_DELETE_WINDOW", on_closing)

# Start updating the camera feed
window.after(500, update_frame)

# Start the Tkinter loop
window.mainloop()
