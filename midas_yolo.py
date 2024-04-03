import cv2
import streamlit as st
import tempfile
import os
import numpy as np
from ultralytics import YOLO
import torch
import urllib.request
import matplotlib.pyplot as plt


def calculate_object_depth(depth_map, x_min, y_min, x_max, y_max, confidence,confidence_threshold=0.7):
    # Convert bounding box coordinates to integers
    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    # Check if confidence is above the threshold
    if confidence >= confidence_threshold:
        # Extract the depth values corresponding to the bounding box coordinates
        depth_values = depth_map[y_min:y_max, x_min:x_max]  # Assuming depth_map is a numpy array

        # Calculate the average depth value within the bounding box
        object_depth = np.mean(depth_values) if depth_values.size > 0 else 0  # Handling case where depth_values is empty

        return object_depth
    else:
        # If confidence is below the threshold, return None
        return 0

# Build a YOLOv9c model from pretrained weight
model = YOLO('yolov9c.pt')

#Build Midas DPT_Large model
model_type = "DPT_Large" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Set Streamlit app title
st.title("Image Path Example")

# Upload an image file
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_image:
    # Read the image
    image_bytes = uploaded_image.read()
    image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Save the image to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_image_path = os.path.join(temp_dir, "uploaded_image.jpg")
    print(temp_image_path)
    cv2.imwrite(temp_image_path, image_array)
    

    # Display the image
    st.image(image_array, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Print the temporary image path
    st.write(f"Temporary image path: {temp_image_path}")

    # Read the image from the temp path
    image = cv2.imread(temp_image_path)
    

    

    results = model(image_array,imgsz=640, save=True)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    
    for box, cls in zip(boxes, classes):
      x1, y1, x2, y2 = map(int, box)
      color = (0, 255, 0)  # Green color
      thickness = 1
      cv2.rectangle(image_array, (x1, y1), (x2, y2), color, thickness)
      class_name = names.get(cls, f"Class {cls}")
      cv2.putText(image_array, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    st.image(image_array, channels="BGR", caption="Uploaded Image", use_column_width=True)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)
    with torch.no_grad():
      prediction = midas(input_batch)

      prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
      ).squeeze()

    
    depth_map = prediction.cpu().numpy()
    # print(depth_map)
    # st.pyplot(depth_map)

    # Assuming depth_map is your depth map data and objects_info is obtained from the previous code snippet
    depth_threshold = 24  # Define a threshold distance in meters

    # Iterate through each detected object
    for box, cls, confidence in zip(boxes, classes, confidences):
      # Extract bounding box coordinates
      x_min, y_min, x_max, y_max = box

      # Calculate the depth information for the object (you may need to adjust this based on your depth map format)
      object_depth = calculate_object_depth(depth_map, x_min, y_min, x_max, y_max,confidence)

      # Check if the object is too close based on the threshold distance
      if object_depth > depth_threshold:
        # Print a message indicating that the object is too close
        st.text(f"Warning! {names[cls]} is too close to you")












