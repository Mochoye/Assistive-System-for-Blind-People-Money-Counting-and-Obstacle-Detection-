import streamlit as st
import subprocess
import re
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('../input/indian-currency-note-resnet-weights/currency_detector_2.4GB_earlyStopping_model.h5')
class_labels = ['10', '100', '20', '200', '2000', '50', '500', 'Background']

def predict_currency(image_file):
    img = image.load_img(image_file, target_size=(256, 256))
    image_to_test = image.img_to_array(img)
    list_of_images = np.expand_dims(image_to_test, axis=0)
    results = model.predict(list_of_images)
    most_likely_class_index = int(np.argmax(results[0]))
    class_likelihood = results[0][most_likely_class_index]
    class_label = class_labels[most_likely_class_index]
    return class_label, class_likelihood
import tempfile
import os

def main():
    st.title("Indian Currency Note Denomination Calculator")

    # Step 1: Create the array of tuples with duplicate denominations
    original_array = []
    num_notes = st.number_input("Enter the number of notes:", min_value=1, step=1)
    
    uploaded_files = []
    for i in range(num_notes):
        uploaded_file = st.file_uploader(f"Upload image {i+1} of an Indian currency note", type=["jpg", "jpeg", "png"], key=f"file_uploader_{i}")
        if uploaded_file:
            uploaded_files.append(uploaded_file)

    for uploaded_file in uploaded_files:
        class_label, _ = predict_currency(uploaded_file)
        original_array.append(int(class_label))

    # Step 2: Sort the array based on the first element of each tuple (the value) in descending order
    sorted_array = sorted([(value, index + 1) for index, value in enumerate(original_array)], key=lambda x: x[0], reverse=True)

    # Transcribe the audio using Whisper
    audio_file = st.file_uploader("Upload the audio file", type=["m4a"], key="audio_file")
    if audio_file:
        # Save the uploaded audio file locally
        with tempfile.NamedTemporaryFile(delete=False) as tmp_audio:
            tmp_audio.write(audio_file.read())
            tmp_audio_path = tmp_audio.name

        # Transcribe the audio file
        try:
            transcription = subprocess.check_output(["whisper", tmp_audio_path, "--model", "base", "--word_timestamps", "True"]).decode("utf-8")
            os.unlink(tmp_audio_path)  # Remove the temporary audio file after transcription
        except subprocess.CalledProcessError as e:
            st.error("Error occurred during audio transcription.")
            os.unlink(tmp_audio_path)  # Remove the temporary audio file in case of error
            return

        # Extract the number mentioned after "find me"
        match = re.search(r'find me (\d+)', transcription)
        if match:
            total_amount = int(match.group(1))

            # Step 3: Initialize variables to track the total and the notes included
            remaining_amount = total_amount
            included_notes = []

            # Step 4: Iterate through the sorted array and calculate the best denominations
            for denomination, index in sorted_array:
                count = remaining_amount // denomination
                if count > 0:
                    included_notes.extend([(denomination, index)] * count)
                    remaining_amount %= denomination

            # Step 5: Display the included notes with their original positions
            st.subheader("Denominations:")
            for denomination, index in included_notes:
                st.write(f"Note of {denomination} rs from position {index}")

            # Step 6: Display the remaining change (if any)
            if remaining_amount > 0:
                st.write(f"Remaining change: {remaining_amount} rs")
        else:
            st.write("No number found in the transcription.")

if __name__ == "__main__":
    main()

	
