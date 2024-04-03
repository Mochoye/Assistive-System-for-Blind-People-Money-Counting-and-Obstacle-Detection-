# Project Title: Assistive System for Blind People (Money Counting and Obstacle Detection)

## Overview
The **Assistive System for Blind People** is designed to address two critical challenges faced by visually impaired individuals: **money counting** and **obstacle detection**. The system combines computer vision and machine learning techniques to provide real-time assistance to blind users.

## Features
1. **Obstacle Detection:**
   - Utilizes **YOLOv9** (You Only Look Once) for real-time object detection.
   - **MIDAS** (Monocular Depth Estimation in Real-time) estimates the depth of detected objects.
   - Warns the user if an object is too close, helping them avoid collisions.

2. **Currency Reader:**
   - Custom **ResNet50** model trained to recognize **Indian currency notes**.
   - Predicts the denomination of currency notes held by the user.
   - Provides an audio output of the detected currency value.

3. **Money Counter:**
   - Given a target amount (e.g., "I need 150 rupees"), the system identifies the positions of currency notes.
   - Guides the user to take out the specific notes needed to meet the target amount.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/assistive-system-for-blind.git
   cd assistive-system-for-blind
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
## Usage
1. Run the obstacle detection module:
   ```bash
    python obstacle_detection.py
   ```
2. Run the currency reader module:
   ```bash
   python currency_reader.py
   ```

## Future Enhancements
- **Implement Voice Alerts:** Enhance the obstacle detection module by incorporating voice alerts. For instance, when an object is detected ahead, the system can audibly announce, "Object detected! Please change your path."
- **Explore Additional Currency Recognition Models:** Investigate other pre-trained models or custom architectures to improve the accuracy of currency recognition. Consider ensemble methods or fine-tuning on a larger dataset for better performance.
- **Video-Based Obstacle Detection:** Extend the obstacle detection capabilities from static images to real-time video streams. Utilize video frames to identify obstacles and provide timely warnings to the user.
- **Dynamic Currency Prediction:** Currently, currency recognition is based on static images. Enhance the system to predict currency denominations from live video feeds, allowing blind users to identify money in real time.

## Results and Visuals


### Obstacle Detection
#### Image Provided
![gettyimages-928538518-640x640](https://github.com/Mochoye/assistive-system-for-blind/assets/95351969/6e1a7fa5-2443-458b-a8a0-674f5c471330)
#### HeatMap for Depth Estimation (Midas)
![heatmap](https://github.com/Mochoye/assistive-system-for-blind/assets/95351969/6cfcc9cc-44dd-4ba2-b783-4832a9dd9bd2)
#### Object Detection(Yolov9)
![BoundingBox](https://github.com/Mochoye/assistive-system-for-blind/assets/95351969/533061d6-3bb2-4383-b8a9-41321a8d556e)
#### Obstacle Detection
![Screenshot](https://github.com/Mochoye/assistive-system-for-blind/assets/95351969/9d212527-c879-43c1-8cf3-1ad924af55da)


## Conclusion

This Assistive System for Blind People represents a significant step toward improving the quality of life for visually impaired individuals. By combining obstacle detection, currency recognition, and money counting, we aim to empower blind users with greater independence and confidence.

Feel free to contribute to this open-source project and make a positive impact in the lives of those who need it most! 🌟


## License
This project is licensed under the **MIT License**. You are free to use, modify, and distribute the code as long as you comply with the terms specified in the LICENSE file.

##





