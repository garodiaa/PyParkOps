import cv2
import easyocr
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import streamlit as st
from PIL import Image
import streamlit_shadcn_ui as ui

# Initialize EasyOCR reader
reader = easyocr.Reader(['bn'])

# Initialize YOLOv8 model
model = YOLO("../Yolo_weights/yolov8n.pt")  # Use a pre-trained YOLOv8 model
lp_model = YOLO("../Yolo_weights/license_plate_detector.pt")

# Main Function
def main():
    # Initialize session state variables
    if "df1" not in st.session_state:
        st.session_state["df1"] = pd.DataFrame(
            columns=['Serial', 'License Number', 'Date', 'In Time', 'Out Time', 'Duration', 'Taka'])
    if "image_index" not in st.session_state:
        st.session_state["image_index"] = 0
    if "serial_number" not in st.session_state:
        st.session_state["serial_number"] = 1

    # List of images
    images = [
        'H:/MLAI/pythonProject/pypark/car1.jpg',
        'H:/MLAI/pythonProject/pypark/car2.jpeg',
        'H:/MLAI/pythonProject/pypark/car3.jpg',
        'H:/MLAI/pythonProject/pypark/car4.jpg',
        'H:/MLAI/pythonProject/pypark/car5.jpg',
        'H:/MLAI/pythonProject/pypark/car6.jpg',
        'H:/MLAI/pythonProject/pypark/car7.png',
        'H:/MLAI/pythonProject/pypark/car1.jpg',
        'H:/MLAI/pythonProject/pypark/car2.jpeg',
        'H:/MLAI/pythonProject/pypark/car3.jpg',
        'H:/MLAI/pythonProject/pypark/car4.jpg',
        'H:/MLAI/pythonProject/pypark/car5.jpg',
        'H:/MLAI/pythonProject/pypark/car6.jpg',
        'H:/MLAI/pythonProject/pypark/car7.png'
    ]

    # Sidebar Section
    with st.sidebar:
        st.image("H:\MLAI\pythonProject\pypark\logo.png", width=50)
        st.subheader('PyPark:red[Ops] Menu')
        if st.session_state["image_index"] < len(images) - 1:
            if st.button("Next"):
                st.session_state["image_index"] += 1
                st.session_state["serial_number"] += 1

        ui.link_button(text="Github", url="https://github.com/garodiaa/pypark",
                       key="link_btn")

    # Home Page Title
    st.image('H:\MLAI\pythonProject\pypark\logo.png', width=80)
    st.title("PyPark:red[Ops]")
    st.subheader("A Car Parking Operations System")

    # Load and display the selected image
    image_path = images[st.session_state["image_index"]]
    selected_image, updated_df = load_image(image_path, 780, 640, st.session_state["serial_number"],
                                            st.session_state["df1"])
    st.session_state["df1"] = updated_df

    # Showing the Image
    st.image(selected_image, caption=f"Image {st.session_state['image_index'] + 1}", use_column_width=True)

    # Display DataFrame
    st.subheader("Update Table 📈")
    st.table(st.session_state["df1"])

    # Button -> Calculate Earning by Date
    st.markdown("<h3 style='text-align: center; color: white;'>💵 Calculate Earning by Date 💵</h3>", unsafe_allow_html=True)
    # Button to trigger the calculation
    if st.button("Calculate Sum"):
        st.session_state["df1"]['Date'] = pd.to_datetime(st.session_state["df1"]['Date'])
        selected_date = st.date_input("Select a date", value=pd.to_datetime('2024-05-21'))

        # Filter DataFrame based on the selected date
        filtered_df = st.session_state["df1"][st.session_state["df1"]['Date'] == pd.to_datetime(selected_date)]

        # Calculate the sum of the 'Taka' column for the filtered data
        total_taka = filtered_df['Taka'].sum()

        # Display the result
        st.write(f"The total Billed Amount for {selected_date} is: ৳{total_taka:.4f}")
        st.write("Earned Dataframe")
        filtered_df['Date'] = filtered_df['Date'].dt.date
        st.write(filtered_df)

    # Button -> Download Database as CSV
    st.markdown("<h3 style='text-align: center; color: white;'>⬇️ Download Your Database! ⬇️</h3>", unsafe_allow_html=True)
    csv = st.session_state["df1"].to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='data.csv',
        mime='text/csv',
    )

    # Add some CSS to center the button
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            margin: 0 auto;
            display: block;
        }

        div.stDownloadButton > button:first-child {
            margin: 0 auto;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to Load image and Detect
def load_image(image_path, width, height, serial_number, df1):
    # Load image using OpenCV
    image = cv2.imread(image_path)

    # Detect cars in the image using YOLOv8
    results = model(image)
    # Bounding Box Code
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get the bounding box coordinates and class ID
            box_data = box.xyxy[0].cpu().numpy().astype(int)
            if len(box_data) == 4:
                x1, y1, x2, y2 = box_data
                conf = box.conf[0]
                cls = int(box.cls[0])
            else:
                x1, y1, x2, y2, conf, cls = box_data

            class_id = cls

            # Check if the detected object is a car
            if class_id == 2:  # The car class ID is 2
                # Crop car region from the image
                car_roi = image[y1:y2, x1:x2]

                # Detect license plate in the car region
                lp_results = lp_model(car_roi)

                for lp in lp_results:
                    # Get license plate bounding box coordinates
                    lp_boxes = lp.boxes
                    for lp_box in lp_boxes:
                        lp_x1, lp_y1, lp_x2, lp_y2 = lp_box.xyxy[0]
                        lp_x1, lp_y1, lp_x2, lp_y2 = int(lp_x1), int(lp_y1), int(lp_x2), int(lp_y2)
                        lp_conf = box.conf[0]

                        # Crop license plate region from the car region
                        lp_roi = car_roi[lp_y1:lp_y2, lp_x1:lp_x2]

                        # Extracting the text from licence plate
                        lp_text = reader.readtext(lp_roi)

                        if len(lp_text) > 0:
                            lp_text_1 = lp_text[0][-2]
                        else:
                            lp_text_1 = ""
                        # Bounding box of License Plate
                        cv2.rectangle(image, (x1 + lp_x1, y1 + lp_y1), (x1 + lp_x2, y1 + lp_y2), (0, 0, 255), 2)
                        if lp_text_1 != "":
                            combined_text = ' '.join([text for bbox, text, prob in lp_text])
                            # st.write(combined_text)

                            current_date = datetime.now().date()
                            current_time = datetime.now().strftime("%H:%M:%S")
                            cost = 0
                            payable = 0
                            duration = 0
                            out_time = '00:00:00'
                            in_time = 0
                            # Checking if already exists in Dataframe
                            if combined_text in df1['License Number'].values:
                                df1.loc[df1['License Number'] == combined_text, 'Out Time'] = current_time
                                time_diff = (pd.to_datetime(df1.loc[df1['License Number'] == combined_text, 'Out Time'],
                                                            format='%H:%M:%S') -
                                             pd.to_datetime(df1.loc[df1['License Number'] == combined_text, 'In Time'],
                                                            format='%H:%M:%S'))
                                df1.loc[df1[
                                            'License Number'] == combined_text, 'Duration'] = time_diff.dt.total_seconds() / 3600  # convert seconds to hours

                                time_diff_hours = df1.loc[df1['License Number'] == combined_text, 'Duration']

                                df1.loc[df1['License Number'] == combined_text, 'Taka'] = (time_diff_hours * 36000)

                                dur = df1.loc[df1['License Number'] == combined_text, 'Duration']
                                duration = (dur.iloc[0])

                                cost = df1.loc[df1['License Number'] == combined_text, 'Taka']
                                payable = (cost.iloc[0])

                                out_time = df1.loc[df1['License Number'] == combined_text, 'Out Time']
                                out_time = (out_time.iloc[0])

                            else:
                                # Add data to the DataFrame
                                df1 = df1._append(
                                    {'Serial': serial_number,
                                     'License Number': combined_text,
                                     'Date': current_date,
                                     'In Time': current_time,
                                     'Out Time': '',
                                     'Duration': 0,
                                     'Taka': 0},
                                    ignore_index=True)
                                serial_number += 1

                            in_time = df1.loc[df1['License Number'] == combined_text, 'In Time']
                            in_time = (in_time.iloc[0])

                            # DashBoard Card Printing
                            cols = st.columns([3, 1])
                            with cols[0]:
                                ui.metric_card(title="License No. 🚗", content=combined_text, key="card1")
                            with cols[1]:
                                ui.metric_card(title="Payable 💸", content=f'৳{int(payable)}', key="card2")

                            cols1 = st.columns([1, 1, 1])
                            with cols1[0]:
                                ui.metric_card(title="In Time 🕜", content=f'{in_time}', key="card3")
                            with cols1[1]:
                                ui.metric_card(title="Out Time 🕞", content=f'{out_time}', key="card4")
                            with cols1[2]:
                                ui.metric_card(title="Duration ⌛", content=f'{duration:.4f}hrs', key="card5")

                # Draw a rectangle around the car
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image
    image = cv2.resize(image, (width, height))
    # Convert image to PIL format
    image = Image.fromarray(image)
    return image, df1


if __name__ == "__main__":
    main()
