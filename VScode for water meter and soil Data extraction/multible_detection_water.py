import cv2
import subprocess
import os
import easyocr
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt

#from util import read_water_meter, write_csv  # Assuming you have utility functions for reading water meters and writing to CSV
#from water_meter_error_handeling import correct_text_errors, value_handling_error
import time
# Define a function to sort coordinates based on y-coordinate
def sort_coordinates(coords):
    # Sort coordinates based on y-coordinate
    sorted_coords = sorted(zip(coords[::2], coords[1::2]), key=lambda x: x[1])
    # Flatten the sorted coordinates list
    flattened_sorted_coords = [val for sublist in sorted_coords for val in sublist]
    return flattened_sorted_coords
def coordinate_tackle(meter_cor, frame,extract_info,stdout_str, num_sets, ocr_model):
    
    meter_cor = sort_coordinates(meter_cor)

    for i in range(num_sets):

        # Extract coordinates for each variable
        meter_cor = meter_cor[i*4 : (i+1)*4]
        for coordinates in [meter_cor]:
            if meter_cor:
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
                #print("can yo hear me")
                print(result)
                cv2.rectangle(frame, tl_corner, br_corner, (0, 255, 0), 5)
                #cv2.putText(frame, meter_text, tl_corner, cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
                cv2.imshow('Meter ROI', meter_roi)  # Show the meter ROI
                cv2.imshow('Original Image', frame)
                # Wait for a key press indefinitely
                cv2.waitKey(0)

                # Close all OpenCV windows
                cv2.destroyAllWindows()