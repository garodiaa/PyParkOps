# bangla number plate in dataframe of car with everything on last edit 9 may
import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize EasyOCR reader
reader = easyocr.Reader(['bn'])

# Initialize YOLOv8 model
model = YOLO("../Yolo_weights/yolov8n.pt")  # Use a pre-trained YOLOv8 model
lp_model = YOLO("../Yolo_weights/license_plate_detector.pt")

df1 = pd.DataFrame(columns=['Serial', 'License Number', 'Date', 'In Time', 'Out Time', 'Time Dif','Teka'])
serial_number = 1  # Initialize serial number

images = ['car1.jpg', 'car2.jpeg', 'car3.jpg', 'car4.jpg', 'car5.jpg', 'car1.jpg', 'car2.jpeg', 'car3.jpg', 'car4.jpg', 'car5.jpg']
for img in images:

    # Read the image
    image = cv2.imread(img)  # Replace 'image.jpg' with your image file name

    # Detect cars in the image using YOLOv8
    results = model(image)

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
            if class_id == 2:  # Assuming the car class ID is 2
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

                        lp_text = reader.readtext(lp_roi)

                        if len(lp_text) > 0:
                            lp_text_1 = lp_text[0][-2]
                        else:
                            lp_text_1 = ""

                        cv2.rectangle(image, (x1 + lp_x1, y1 + lp_y1), (x1 + lp_x2, y1 + lp_y2), (0, 0, 255), 2)
                        # cv2.putText(image, f"{lp_text_1}", (x1 + lp_x1, y1 + lp_y1 - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                        if lp_text_1 != "":
                            combined_text = ' '.join([text for bbox, text, prob in lp_text])
                            print(combined_text)
                            current_date = datetime.now().date()
                            current_time = datetime.now().strftime("%H:%M:%S")
                            # milaaa gelee
                            if combined_text in df1['License Number'].values:
                                df1.loc[df1['License Number'] == combined_text, 'Out Time'] = current_time
                                df1.loc[df1['License Number'] == combined_text, 'Time Dif'] = pd.to_datetime(
                                    df1['Out Time'], format='%H:%M:%S') - pd.to_datetime(df1['In Time'],
                                                                                         format='%H:%M:%S')

                                time_diff_hours = (pd.to_datetime(df1.loc[df1['License Number'] == combined_text, 'Out Time'], format='%H:%M:%S') -pd.to_datetime(df1.loc[df1['License Number'] == combined_text, 'In Time'],
                                                       format='%H:%M:%S')).dt.total_seconds() / 3600

                                df1.loc[df1['License Number'] == combined_text, 'Teka'] = (time_diff_hours*36000)
                            # na milleeee
                            else:
                                # Add data to the DataFrame
                                df1 = df1._append(
                                    {'Serial': serial_number,
                                     'License Number': combined_text,
                                     'Date': current_date,
                                     'In Time': current_time},
                                    ignore_index=True)
                                serial_number += 1

                # Draw a rectangle around the car and display the license plate
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)


print(df1.to_string(index=False))
df1.to_csv('test.csv', encoding='utf-8', index=False)

# ct1 = df1.iloc[0]['License Number']
# ct3 = df1.iloc[3]['License Number']
#
# print(ct1)
# print(ct3)

# if ct1 == ct3:
#     print("The combined texts are exactly the same.")
# else:
#     print("The combined texts are different.")
# Display the processed image
cv2.destroyAllWindows()
