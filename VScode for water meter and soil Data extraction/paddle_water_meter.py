import cv2
import subprocess
import os
import easyocr
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
#from util import read_water_meter, write_csv  # Assuming you have utility functions for reading water meters and writing to CSV
#from water_meter_error_handeling import correct_text_errors, value_handling_error
import time
import re
from multible_detection_water import coordinate_tackle

#from ultralytics import YOLO
# Define YOLO detection arguments
#image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/1.png'
#image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/Screenshot 2024-03-04 154752.png'
#image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/Screenshot 2024-03-04 155645.png'
#image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_520_value_102_667.jpg'
#image= '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_519_value_43_335.jpg'
#image ='/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/17.png'
#image ='/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_518_value_14_377.jpg'
#image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_524_value_198_818.jpg'
#image = "/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_522_value_505_029.jpg"
#image = "/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_26_value_252_131.jpg"
#image = "/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_114_value_406_672.jpg"
#image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_521_value_423_037.jpg'
#image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_524_value_198_818.jpg'
#image ='/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_519_value_43_335.jpg'
#image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_72_value_132_107.jpg'
#image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_1151_value_22_418.jpg'
#image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_941_value_37_11.jpg'
#image ='/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_104_value_322_032.jpg'
image = '/home/youssefokab/catkin_ws/src/yolov7/water_meter_tests/id_123_value_337_294.jpg'

ocr_model = PaddleOCR(lang='en', use_gpu=True) 

detect_args = [
    'detect.py',
    '--weights', 'water_meter_best.pt',
    '--conf', '0.5',
    '--img-size', '640',
    '--source', image,
    '--view-img',
    '--no-trace'
]

# Run detection by calling the script as a subprocess
process = subprocess.Popen(['python3'] + detect_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

# Decode stdout to string
stdout_str = stdout.decode()
print("STDOUT:", stdout.decode())
print("STDERR:", stderr.decode())


# Path to the file you want to delete
del_path = "/home/youssefokab/catkin_ws/src/Image_text_detection_code/meter_scan_folder/cropped_image.png"

# Check if the file exists
if os.path.exists(del_path):
    # Delete the file
    os.remove(del_path)
    print("File deleted successfully.")
else:
    print("The file does not exist.")
# Process image
frame = cv2.imread(image)
# Read water meter plate number
def extract_info(stdout_str, cls):
    extracted_info = [] 
    #current_cls = None  
    for line in stdout_str.split('\n'):
        if "class:" in line:
            current_cls = line.split(": ")[1].strip()
            # print(current_cls)
        elif "box location:" in line:
            box_location = line.split(":")[1].strip()
            #print(box_location)
            # If the current class matches the specified class, extract and store the coordinates
            if current_cls == cls:
                #print(current_cls)
                # Extract coordinates for the Analysis number result class
                x1, y1, x2, y2 = map(int, box_location.strip('[]').split(', '))
                extracted_info.append((x1))
                extracted_info.append((y1))
                extracted_info.append((x2))
                extracted_info.append((y2))
    return extracted_info
placeholder ='N/A'
meter_cor = extract_info(stdout_str, 'meter')
if meter_cor:
    num_sets = len(meter_cor) // 4
    if num_sets > 1:
            coordinate_tackle(meter_cor, frame,extract_info,stdout_str, num_sets, ocr_model)
    elif num_sets == 1:
            print("meter result")
            print(meter_cor)
            x1, y1, x2, y2 = meter_cor
            tl_corner = (int(x1), int(y1))
            br_corner = (int(x2), int(y2))
            # Extract the region of interest (ROI) from the frame
            meter_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
            save_path = '/home/youssefokab/catkin_ws/src/Image_text_detection_code/meter_scan_folder/cropped_image.png'
            
            # Saving the cropped image
            cv2.imwrite(save_path, meter_roi)
            image_to_scan = save_path
            result = ocr_model.ocr(image_to_scan)
            result_str = str(result)
            meter_text = re.search(r"\('(.*?)',", result_str).group(1)
            #print("can yo hear me")
            print(result)
            print(meter_text)
            cv2.rectangle(frame, tl_corner, br_corner, (0, 255, 0), 5)
            #cv2.putText(frame, meter_text, tl_corner, cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
            cv2.imshow('Meter ROI', meter_roi)  # Show the meter ROI
            cv2.imshow('Original Image', frame)
            # Wait for a key press indefinitely
            cv2.waitKey(0)

            # Close all OpenCV windows
            cv2.destroyAllWindows()
            #temp = []
            #meter_text = correct_text_errors(meter_text,value_handling_error)
            
else:
    meter_text = placeholder
  # Show the original image frame