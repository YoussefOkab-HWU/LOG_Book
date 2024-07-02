import cv2
import subprocess
import os
import easyocr
import time
import psutil
import resource
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import pandas as pd
from  error_handeling_dictionary import correct_text_errors, value_handling_error , num_date, value_handling_error2, spelling_errors 
from pdf_to_png import pdf_to_png
import json
import time


page = 0
# Define YOLO detection arguments

pdf_directory ='/home/youssefokab/catkin_ws/src/yolov7/pdf_file_output/'

output_directory = "/home/youssefokab/catkin_ws/src/yolov7/pdf_images"
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_file = os.path.join(pdf_directory, filename)
        pdf_to_png(pdf_file, output_directory)
        #id = os.path.splitext(filename)[0]
image_dir = '/home/youssefokab/catkin_ws/src/yolov7/pdf_images'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
# Sort the image files based on the numeric part extracted from the filename
# Function to extract the numeric part from the filename
def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])
image_files.sort(key=extract_number)
detect_args = [
    'detect.py',
    '--weights', 'brand_new_happy.pt',
    '--conf', '0.5',
    '--img-size', '640',
    '--view-img',
    '--no-trace'
] 
for image_file in image_files:
     # Extracted data for each image
    # Get memory usage
    mem = psutil.virtual_memory()
    print(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {mem.used / (1024 ** 3):.2f} GB")
    # Get memory usage
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Memory usage: {mem_usage / (1024 ** 2):.2f} MB")
    #(i want if there is no analysis number it will come back to here straight away)
    image_path = os.path.join(image_dir, image_file)
    
    # Run detection by calling the script as a subprocess
    process = subprocess.Popen(['python3'] + detect_args + ['--source', image_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decode stdout to string
    stdout_str = stdout.decode()
    print(stdout_str)