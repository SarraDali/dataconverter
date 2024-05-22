import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import pytesseract
import pandas as pd

# Configure the path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Initialize YOLO model path
model_path = 'Modele/TrainingModel.pt'

# Load the YOLO model
@st.cache_data
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_path)



# Function to detect objects in the image using the YOLO model
def detect_objects(image, model):
    try:
        results = model.predict(image)
        return results
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return None

# Function to extract text from a specific region in the image
def extract_text_from_region(image, box):
    try:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_image = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cropped_image)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from region: {e}")
        return ""

# Function to determine the cable type based on the text in the detected table
def determine_cable_type_from_table(image, box):
    try:
        x1, y1, x2, y2 = map(int, box[:4])
        table_region = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(table_region)
        lines = text.strip().split('\n')
        num_lines = len(lines)
        if num_lines == 5:
            cable_type = 'Ethernet'
        elif num_lines > 5:
            cable_type = 'Hsd'
        else:
            cable_type = 'Antenna'
        return cable_type
    except Exception as e:
        st.error(f"Error determining cable type from table: {e}")
        return "Unknown"

# Main function to run the Streamlit app
def main():
    st.title('Data Converter LEONI \n Convert Technical Drawings with Accuracy and Ease')

    uploaded_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "png", "jpeg", "pdf"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        results_list = detect_objects(image_cv2, model)

        # Mapping of old class names to new class names
        class_name_mapping = {
            "0- Side1": "Side1",
            "1- Side2": "Side2",
            "2- LEONIPartNumber": "LEONIPartNumber",
            "3- SupplierPartNumber": "SupplierPartNumber",
            "4- Wiretype": "Wiretype",
            "5- Length": "Length",
            "6- TypeOfCableAssembly": "TypeOfCableAssembly"
        }

        # Dictionary to store the extracted data
        class_data = {new_name: [] for new_name in class_name_mapping.values()}

        if results_list:
            for results in results_list:
                if hasattr(results, 'boxes') and results.boxes is not None:
                    for i, box in enumerate(results.boxes.xyxy):
                        if len(box) >= 4:
                            class_id = int(results.boxes.cls[i]) if len(results.boxes.cls) > i else -1
                            label = results.names[class_id] if class_id in results.names else "Unknown"
                            new_label = class_name_mapping.get(label, label)
                            if label == '6- TypeOfCableAssembly':
                                cable_type = determine_cable_type_from_table(image_cv2, box)
                                text = cable_type
                            else:
                                text = extract_text_from_region(image_cv2, box)
                            if new_label in class_data:
                                class_data[new_label].append(text)
                            else:
                                st.warning(f"Detected label '{label}' is not in the specified columns.")
                            cv2.rectangle(image_cv2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            annotated_image = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
            st.image(annotated_image, caption='Annotated Image', use_column_width=True)

            # Create a DataFrame for the CSV export
            df = pd.DataFrame.from_dict(class_data, orient='index').transpose()
            column_order = ['Side1', 'Side2', 'LEONIPartNumber', 'SupplierPartNumber', 'Wiretype', 'Length', 'TypeOfCableAssembly']
            df = df[column_order]  # Reorder the columns

            # Display data in a table
            st.write("Extracted Data:")
            st.dataframe(df)

            # Provide a download button for the CSV file
            csv = df.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(label="Download data as CSV",
                               data=csv,
                               file_name='extracted_data.csv',
                               mime='text/csv')
        else:
            st.write("No detections or incorrect result format.")

if __name__ == '__main__':
    main()