import streamlit as st
import subprocess
import re
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Load the pre-trained model
model = load_model('/content/currency_detector_2.4GB_earlyStopping_model.h5')
class_labels = ['10', '100', '20', '200', '2000', '50', '500', 'Background']

def predict_currency(image_file):
    img = image.load_img(image_file, target_size=(256, 256))
    image_to_test = image.img_to_array(img)
    list_of_images = np.expand_dims(image_to_test, axis=0)
    results = model.predict(list_of_images)
    most_likely_class_index = int(np.argmax(results[0]))
    class_label = class_labels[most_likely_class_index]
    st.write(f"Predicted Amount for note {image_file.name}: {class_label} rs")
    return int(class_label)  # Return integer value of denomination
def main():
    st.markdown("<h1 style='text-align: center; color: #ff5733;'>Indian Currency Denomination Recognition and Calculation</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: #006400;'>By Soumedhik Bharati and Archisman Ray.</h2>", unsafe_allow_html=True)
    
    # Your existing code goes here

    # Step 1: Upload images of Indian currency notes
    original_array = []
    num_notes = st.number_input("Enter the number of notes:", min_value=1, step=1)

    uploaded_files = []
    for i in range(num_notes):
        uploaded_file = st.file_uploader(f"Upload an image of Indian currency note {i+1}", key=f"file_uploader_{i}", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            uploaded_files.append((i, uploaded_file))  # Store position and file tuple

    # Step 2: Predict denominations from uploaded images and populate the original array
    for position, uploaded_file in uploaded_files:
        denomination = predict_currency(uploaded_file)
        original_array.append((position, denomination))  # Store position and denomination

    # Step 3: Transcribe the uploaded audio file and extract the total amount mentioned
    audio_file = st.file_uploader("Upload the audio file", type=["m4a"])
    if audio_file:
        # Save the uploaded audio file temporarily
        temp_audio_path = "/tmp/uploaded_audio.m4a"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.read())

        # Transcribe the saved audio file
        try:
            output = subprocess.check_output(["whisper", temp_audio_path, "--model", "base", "--word_timestamps", "True"]).decode("utf-8")
            # Split the output at the last occurrence of ']'
            parts = output.split(']', 1)
            if len(parts) > 1:
                # If the split was successful, keep only the second part
                transcription = parts[1].strip()  # strip() is used to remove leading and trailing whitespace
            else:
                # If the split was not successful, keep the whole output
                transcription = output
            st.write(f"Transcription: {transcription}")
        except subprocess.CalledProcessError:
            st.error("Error occurred while transcribing the audio. Please try again.")
            return

        # Remove the temporary audio file
        os.remove(temp_audio_path)

        # Step 4: Extract the total amount mentioned in the transcription
        match = re.search(r'\b(\d+(,\d{3})*)\b', transcription)
        if match:
            total_amount = int(match.group(1).replace(',', ''))  # Remove commas
            st.write(f"Total Amount: {total_amount}")
        else:
            st.write("No number found in the transcription.")
            return

        # Step 5: Sort the denominations based on their original positions
        sorted_array = sorted(original_array, key=lambda x: x[0])

        # Step 6: Initialize variables to track the remaining amount and the included notes
        remaining_amount = total_amount
        included_notes = []

        # Step 7: Calculate the best denominations to meet the total amount
        for position, denomination in sorted_array:
            count = remaining_amount // denomination
            if count > 0:
                included_notes.extend([(position, denomination)] * count)
                remaining_amount %= denomination

        # Step 8: Display the included notes with their original positions
        st.subheader("Denominations:")
        for position, denomination in included_notes:
            st.write(f"Note of {denomination} rs from position {position + 1}")

        # Step 9: Display the remaining change (if any)
        if remaining_amount > 0:
            st.write(f"Remaining change: {remaining_amount} rs")

if __name__ == "__main__":
    main()
