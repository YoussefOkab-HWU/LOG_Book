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
from  error_handeling_dictionary import correct_text_errors, value_handling_error , num_date, value_handling_error2,value_handling_error3, spelling_errors 
from multible_detection import coordinate_tackle
from pdf_to_png import pdf_to_png
import json
import time
from datetime import datetime
#from nvitop import train_model

#import GPUtil


page = 0
# Define YOLO detection arguments
y=0
pdf_directory ='/home/youssefokab/catkin_ws/src/yolov7/pdf_file_output/'

output_directory = "/home/youssefokab/catkin_ws/src/yolov7/pdf_images"
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_file = os.path.join(pdf_directory, filename)
        pdf_to_png(pdf_file, output_directory)
        id =  os.path.splitext(filename)[0]
image_dir = '/home/youssefokab/catkin_ws/src/yolov7/pdf_images'

# Function to extract the numeric part from the filename
def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])

image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
# Sort the image files based on the numeric part extracted from the filename
image_files.sort(key=extract_number)
detect_args = [
    'detect.py',
    '--weights', 'brand_new_happy.pt',
    '--conf', '0.5',
    '--img-size', '640',
    '--view-img',
    '--no-trace'
] 

api_data = {
    "Date": time.strftime("%Y-%m-%d %H:%M:%S"),  # Current date and time
    "iddocument": id,  # Generate a unique ID for the document
    "data": []
}

    # Process images in the directory
for image_file in image_files:
     # Extracted data for each image
        # Get memory usage
    mem = psutil.virtual_memory()
    print(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {mem.used / (1024 ** 3):.2f} GB")

    # # Get memory usage
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Memory usage: {mem_usage / (1024 ** 2):.2f} MB")
    image_path = os.path.join(image_dir, image_file)
    
    # Run detection by calling the script as a subprocess
    process = subprocess.Popen(['python3'] + detect_args + ['--source', image_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decode stdout to string
    stdout_str = stdout.decode()
    print(stdout_str)
    if process.returncode != 0:
        print(f"Error occurred while processing {image_path}:")
        print("Standard Output:")
        print(stdout.decode())
        print("Standard Error:")
        print(stderr.decode())
    
    

    # Process image
    frame = cv2.imread(image_path)
# Read water meter plate number
    reader = easyocr.Reader(['en'], gpu=True)  
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
    #additional_info_results = []
    placeholder = "N/A"
    page = page+1
    # Extract information
    analysis_number_results_cor = extract_info(stdout_str, 'Analysis number result')
    laboratory_sample_result_cor = extract_info(stdout_str,'laboratory sample result')
    analysis_chemical_results_cor = extract_info(stdout_str,'analysis chemical result')
    zone_area_cor = extract_info(stdout_str,'zone area')
        #print(analysis_number_results_cor)
    print("hello?")
    if analysis_number_results_cor or laboratory_sample_result_cor or analysis_chemical_results_cor or zone_area_cor:
        
        json_flag = False 
        print("am I working?") 
        depth_text = "N/A"
        phos_text = "N/A"
        pota_text = "N/A"
        mag_text = "N/A"
        sod_text = "N/A"
        Aluminium_text = "N/A"
        cal_text = "N/A"
        ph_text = "N/A"
        carbone_organic_text = "N/A"
        humus_text = "N/A"
        man_text = "N/A"
        iron_text = "N/A"
        nitro_text = "N/A"
        boron_text = "N/A"
        sulfer_text = "N/A"
        carb_text = "N/A"
        azote_text = "N/A"
        car_azo_text = "N/A"
        chl_de_sod_text = "N/A"
        conduct_text = "N/A"
        cap_cat_text = "N/A"
        zone_text = "N/A"
        nt_text = "N/A"
        ph_acetate_text = "N/A"
        taux_argile_text = "N/A"
        CEC_text = "N/A"
        reportCN_text = "N/A"
        reportKMG_text = "N/A"
        reportCAMG_text = "N/A"
        PH_only_text = "N/A"
        hardness_total_text = "N/A"
        PO4_text = "N/A"
        SO4_text = "N/A"
        chlo_text = "N/A"
        sables_50_100_text = "N/A"
        sables_100_200_text = "N/A"
        sables_200_500_text = "N/A"
        sables_500_1000_text = "N/A"
        sables_1000_2000_text = "N/A"
        sables_larger_2000_text = "N/A"
        salt_text = "N/A"
        phossoul_text = "N/A"
        copp_text = "N/A"
        zinc_text = "N/A"
        sample_marked_text = "N/A"
        if analysis_number_results_cor:
            #cv2.imshow('frame', frame)
        
            # Wait for a key press and then close the window
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            print("analysis number:")
            print(analysis_number_results_cor)
            x1, y1, x2, y2 = analysis_number_results_cor 

            tl_corner = (int(x1), int(y1))
            br_corner = (int(x2), int(y2))
            print(tl_corner)
            print(br_corner)
            # Extract the region of interest (ROI) from the frame
            ana_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
            #cv2.imshow('ana ROI', ana_roi)
            Analysis_number_result = reader.readtext(ana_roi)
            for t_, t in enumerate(Analysis_number_result):
                bbox, text, score = t
                t = bbox, Analysis_number_result, score
                print(t)
            analysis_number_text = ''.join([text for bbox, text, score in Analysis_number_result])
            analysis_number_text = correct_text_errors(analysis_number_text,value_handling_error2)
        else:
            analysis_number_text = placeholder
        if analysis_number_text == placeholder:
            print("No analysis number detected.")
            
        else:

                ##additional_info_results.append(('Analysis number', text))
            date_cor = extract_info(stdout_str, 'date')
            print(date_cor)

            if date_cor:
                print("date")
                print(date_cor)
                x1, y1, x2, y2 = date_cor
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                print(tl_corner)
                print(br_corner)
            # Extract the region of interest (ROI) from the frame
                date_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                #cv2.imshow('Date ROI', date_roi)
                date = reader.readtext(date_roi)
                for t_, t in enumerate(date):
                    bbox, text, score = t
                    t = bbox, date, score
                    print(t)
                date.sort(key=lambda x: x[0][0][0])
                date_text = ' '.join([text for bbox, text, score in date])
                temp = []
                for word in date_text.split():
                    temp.append(num_date.get(word, word))
                    date_text = ''.join(temp)
                date_text = datetime.strptime(date_text, "%d-%m-%Y")
                date_text = date_text.strftime("%Y-%m-%d")
            else:
                date_text = placeholder
                ##additional_info_results.append(('date', text))


            Name_of_the_plot_cor = extract_info(stdout_str, 'Name of the plot')
            if Name_of_the_plot_cor:
                print("name of plot:")
                print(Name_of_the_plot_cor)
                try:
                    x1, y1, x2, y2 = Name_of_the_plot_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    NOP_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    Name_of_the_plot = reader.readtext(NOP_roi)
                    for t_, t in enumerate(Name_of_the_plot):
                        bbox, text, score = t
                        t = bbox, Name_of_the_plot, score
                        print(t)
                    NOP_text = ' '.join([text for bbox, text, score in Name_of_the_plot])
                    temp = []
                    for word in NOP_text.split():
                        temp.append(spelling_errors.get(word, word))
                        #temp.append(num_date.get(word, word))
                        NOP_text = ' '.join(temp)
                except ValueError:
                    NOP_text = placeholder          
            else:
                NOP_text = placeholder
                ##additional_info_results.append(('name of the plot', text))

            cultural_precedent_cor = extract_info(stdout_str, 'cultural precedent')
            if cultural_precedent_cor:
                print("cultural_precedent")
                print(cultural_precedent_cor)
                x1, y1, x2, y2 = cultural_precedent_cor
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                prec_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                cultural_precedent = reader.readtext(prec_roi)
                for t_, t in enumerate(cultural_precedent):
                    bbox, text, score = t
                    t = bbox, cultural_precedent, score
                    print(t)
                prec_text = ' '.join([text for bbox, text, score in cultural_precedent])
            else:
                prec_text = placeholder
                ##additional_info_results.append(('cultural precedent', text))

            cultural_project_cor = extract_info(stdout_str, 'cultural project')
            if cultural_project_cor:
                print("cultural project")
                print(cultural_project_cor)
                x1, y1, x2, y2 = cultural_project_cor
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                proj_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                cultural_project = reader.readtext(proj_roi)
                for t_, t in enumerate(cultural_project):
                    bbox, text, score = t
                    t = bbox, cultural_project, score
                    print(t)
                proj_text = ' '.join([text for bbox, text, score in cultural_project])
            else:
                proj_text = placeholder
            # #additional_info_results.append(('Cultural project', text))
            if len(proj_text) > len(prec_text):
                prec_text = proj_text
            elif len(proj_text) < len(prec_text):
                proj_text = prec_text
            elif len(proj_text) == len(prec_text):
                proj_text = proj_text
                prec_text = prec_text
            phosphore_result_cor = extract_info(stdout_str, 'phosphore result')
            if phosphore_result_cor:
                print("phosphore_result")
                print(phosphore_result_cor)
                x1, y1, x2, y2 = phosphore_result_cor
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                phos_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                phosphore_result = reader.readtext(phos_roi)
                for t_, t in enumerate(phosphore_result):
                    bbox, text, score = t
                    t = bbox, phosphore_result, score
                    print(t)
                phosphore_result.sort(key=lambda x: x[0][0][0])
                phos_text = ' '.join([text for bbox, text, score in phosphore_result])
                if ',' not in phos_text and '_' not in phos_text:
                    if '  '  in phos_text:
                            phos_text = phos_text.replace('  ', '.') 
                    elif ' ' in phos_text:
                            phos_text = phos_text.replace(' ', '.') 
                temp = []
                for word in phos_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    phos_text = ''.join(temp) 
                    phos_text = correct_text_errors(phos_text,value_handling_error2)
                if phos_text.endswith('.'):
                    phos_text += '0'
            else:
                phos_text = placeholder
            # #additional_info_results.append(('phosphore', text))

            potassium_result_cor = extract_info(stdout_str, 'potassium result')
            if potassium_result_cor:
                print("potassium result")
                print(potassium_result_cor)
                try:
                    x1, y1, x2, y2 = potassium_result_cor
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    pota_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    potassium_result = reader.readtext(pota_roi)
                    for t_, t in enumerate(potassium_result):
                        bbox, text, score = t
                        t = bbox, potassium_result, score
                        print(t)
                    potassium_result.sort(key=lambda x: x[0][0][0])
                    pota_text = ' '.join([text for bbox, text, score in potassium_result])
                    if ',' not in pota_text and '_' not in pota_text:
                        if '  '  in pota_text:
                                pota_text = pota_text.replace('  ', '.') 
                        elif ' ' in pota_text:
                                pota_text = pota_text.replace(' ', '.') 
                    temp = []
                    for word in pota_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        pota_text = ''.join(temp)
                        pota_text = correct_text_errors(pota_text,value_handling_error2)
                    if pota_text.endswith('.'):
                        pota_text += '0'
                except ValueError:
                        pota_text = placeholder
            else:
                pota_text = placeholder
                #additional_info_results.append(('potassium', text))

            magnesium_result_cor = extract_info(stdout_str, 'magnesium result')
            if magnesium_result_cor:
                print("magnesium_result")
                print(magnesium_result_cor)
                try:
                    x1,y1,x2,y2 = magnesium_result_cor
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    mag_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    magnesium_result = reader.readtext(mag_roi)
                    for t_, t in enumerate(magnesium_result):
                        bbox, text, score = t
                        t = bbox, magnesium_result, score
                        print(t)
                    magnesium_result.sort(key=lambda x: x[0][0][0])
                    mag_text = ' '.join([text for bbox, text, score in magnesium_result])
                    if ',' not in mag_text and '_' not in mag_text:
                        if '  '  in mag_text:
                                mag_text = mag_text.replace('  ', '.') 
                        elif ' ' in mag_text:
                                mag_text = mag_text.replace(' ', '.') 
                    temp = []
                    for word in mag_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        mag_text = ''.join(temp)
                    
                        mag_text = correct_text_errors(mag_text,value_handling_error2) 
                    if mag_text.endswith('.'):
                        mag_text += '0'
                
                except ValueError:
                    mag_text = placeholder
            else:
                mag_text = placeholder
                #additional_info_results.append(('magnesium', text))

            sodium_result_cor = extract_info(stdout_str, 'sodium result')
            if sodium_result_cor:
                print("sodium_result")
                x1,y1,x2,y2 = sodium_result_cor
                print(sodium_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
            # Extract the region of interest (ROI) from the frame
                sod_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                sodium_result = reader.readtext(sod_roi)
                for t_, t in enumerate(sodium_result):
                    bbox, text, score = t
                    t = bbox, sodium_result, score
                    print(t)
                sodium_result.sort(key=lambda x: x[0][0][0])
                sod_text = ' '.join([text for bbox, text, score in sodium_result])
                if ',' not in sod_text and '_' not in sod_text:
                    if '  '  in sod_text:
                            sod_text = sod_text.replace('  ', '.') 
                    elif ' ' in sod_text:
                            sod_text = sod_text.replace(' ', '.') 
                temp = []
                for word in sod_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    sod_text = ''.join(temp)
                    sod_text = correct_text_errors(sod_text,value_handling_error2)
                if sod_text.endswith('.'):
                    sod_text += '0'
            else:
                sod_text = placeholder
                #additional_info_results.append(('sodium', text))

            calcium_result_cor = extract_info(stdout_str, 'calcium result')
            if calcium_result_cor:
                print("calcium result")
                x1,y1,x2,y2 = calcium_result_cor
                print(calcium_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                cal_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                calcium_result = reader.readtext(cal_roi)
                for t_, t in enumerate(calcium_result):
                    bbox, text, score = t
                    t = bbox, calcium_result, score
                    print(t)
                calcium_result.sort(key=lambda x: x[0][0][0])
                cal_text = ' '.join([text for bbox, text, score in calcium_result])
                if ',' not in cal_text and '_' not in cal_text:
                    if '  '  in cal_text:
                            cal_text = cal_text.replace('  ', '.') 
                    elif ' ' in cal_text:
                            cal_text = cal_text.replace(' ', '.') 
                temp = []
                for word in cal_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    cal_text = ''.join(temp)
                    cal_text = correct_text_errors(cal_text,value_handling_error2)
                if cal_text.endswith('.'):
                    cal_text += '0'
            else:
                cal_text = placeholder
                #additional_info_results.append(('calcium', text))

            PH_at_KCL_results_cor = extract_info(stdout_str, 'PH at KCL results')
            if PH_at_KCL_results_cor:
                print("ph at kcl")
                x1,y1,x2,y2 = PH_at_KCL_results_cor
                print(PH_at_KCL_results_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                ph_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                PH_at_KCL_results = reader.readtext(ph_roi)
                for t_, t in enumerate(PH_at_KCL_results):
                    bbox, text, score = t
                    t = bbox, PH_at_KCL_results, score
                    print(t)
                PH_at_KCL_results.sort(key=lambda x: x[0][0][0])
                ph_text = ' '.join([text for bbox, text, score in PH_at_KCL_results])
                if ',' not in ph_text and '_' not in ph_text:
                    if '  '  in ph_text:
                            ph_text = ph_text.replace('  ', '.') 
                    elif ' ' in ph_text:
                            ph_text = ph_text.replace(' ', '.') 
                temp = []
                for word in ph_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    ph_text = ''.join(temp)
                    ph_text = correct_text_errors(ph_text,value_handling_error2)
                if ph_text.endswith('.'):
                    ph_text += '0'
            else:
                ph_text = placeholder
                #additional_info_results.append(('PH at KCL', text))

            HUMUS_percent_result_cor = extract_info(stdout_str, 'HUMUS % result')
            if HUMUS_percent_result_cor:
                print("humus percent")
                x1,y1,x2,y2 = HUMUS_percent_result_cor

                print(HUMUS_percent_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                hum_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                HUMUS_result = reader.readtext(hum_roi)
                for t_, t in enumerate(HUMUS_result):
                    bbox, text, score = t
                    t = bbox, HUMUS_result, score
                    print(t)
                HUMUS_result.sort(key=lambda x: x[0][0][0])
                humus_text = ' '.join([text for bbox, text, score in HUMUS_result])
                if ',' not in humus_text and '_' not in humus_text:
                    if '  '  in humus_text:
                            humus_text = humus_text.replace('  ', '.') 
                    elif ' ' in humus_text:
                            humus_text = humus_text.replace(' ', '.') 
                temp = []
                for word in humus_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    humus_text = ''.join(temp)
                    humus_text = correct_text_errors(humus_text,value_handling_error2)
                if humus_text.endswith('.'):
                    humus_text += '0'
            else:
                humus_text = placeholder
                ##additional_info_results.append(('humus', text))
            manganese_result_cor = extract_info(stdout_str, 'manganese result')
            if manganese_result_cor:
                print("manganese result")
                print(manganese_result_cor)
                try:
                    x1,y1,x2,y2 = manganese_result_cor

                    
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    man_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    manganese_result = reader.readtext(man_roi)
                    for t_, t in enumerate(manganese_result):
                        bbox, text, score = t
                        t = bbox, manganese_result, score
                        print(t)
                    manganese_result.sort(key=lambda x: x[0][0][0])
                    man_text = ' '.join([text for bbox, text, score in manganese_result])
                    print(man_text)
                    if ',' not in man_text and '_' not in man_text:
                        if '  '  in man_text:
                                man_text = man_text.replace('  ', '.') 
                        elif ' ' in man_text:
                                man_text = man_text.replace(' ', '.') 
                    print(man_text)

                    temp = []
                    for word in man_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        man_text = ''.join(temp)
                        man_text = correct_text_errors(man_text,value_handling_error2)
                    if man_text.endswith('.'):
                        man_text += '0'
                except ValueError:
                    man_text = placeholder
            else:
                man_text = placeholder
                #additional_info_results.append(('manganese', text))

            Iron_result_cor = extract_info(stdout_str, 'Iron result')
            if Iron_result_cor:
                print("iron")
                x1,y1,x2,y2 = Iron_result_cor

                print(Iron_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                iron_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                Iron_result = reader.readtext(iron_roi)
                for t_, t in enumerate(Iron_result):
                    bbox, text, score = t
                    t = bbox, Iron_result, score
                    print(t)
                Iron_result.sort(key=lambda x: x[0][0][0])
                iron_text = ' '.join([text for bbox, text, score in Iron_result])
                if ',' not in iron_text and '_' not in iron_text:
                    if '  '  in iron_text:
                            iron_text = iron_text.replace('  ', '.') 
                    elif ' ' in iron_text:
                            iron_text = iron_text.replace(' ', '.') 
                temp = []
                for word in iron_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    iron_text = ''.join(temp)
                    iron_text = correct_text_errors(iron_text,value_handling_error2)
                if iron_text.endswith('.'):
                    iron_text += '0'
            else:
                iron_text = placeholder
                #additional_info_results.append(('iron', text))

            AZOTE_TOTAL_percent_result_cor = extract_info(stdout_str, 'AZOTE TOTAL % result')
            if AZOTE_TOTAL_percent_result_cor:
                print("azote total ")
                x1,y1,x2,y2 = AZOTE_TOTAL_percent_result_cor

                print(AZOTE_TOTAL_percent_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                azote_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                AZOTE_TOTAL_result = reader.readtext(azote_roi)
                for t_, t in enumerate(AZOTE_TOTAL_result):
                    bbox, text, score = t
                    t = bbox, AZOTE_TOTAL_result, score
                    print(t)
                AZOTE_TOTAL_result.sort(key=lambda x: x[0][0][0])
                azote_text = ' '.join([text for bbox, text, score in AZOTE_TOTAL_result])
                if ',' not in azote_text and '_' not in azote_text:
                    if '  '  in azote_text:
                            azote_text = azote_text.replace('  ', '.') 
                    elif ' ' in azote_text:
                            azote_text = azote_text.replace(' ', '.') 
                temp = []
                for word in azote_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    azote_text = ''.join(temp)
                    azote_text = correct_text_errors(azote_text,value_handling_error2)
                if azote_text.endswith('.'):
                    azote_text += '0'
                #additional_info_results.append(('azote', text))
            else:
                azote_text = placeholder
            carbone_result_cor = extract_info(stdout_str, 'carbone result')
            if carbone_result_cor:
                print("carbone")
                x1,y1,x2,y2 = carbone_result_cor

                print(carbone_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                car_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                carbone_result = reader.readtext(car_roi)
                for t_, t in enumerate(carbone_result):
                    bbox, text, score = t
                    t = bbox, carbone_result, score
                    print(t)
                carbone_result.sort(key=lambda x: x[0][0][0])
                carb_text = ' '.join([text for bbox, text, score in carbone_result])
                if ',' not in carb_text and '_' not in carb_text:
                    if '  '  in carb_text:
                            carb_text = carb_text.replace('  ', '.') 
                    elif ' ' in carb_text:
                            carb_text = carb_text.replace(' ', '.') 
                temp = []
                for word in carb_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    carb_text = ''.join(temp)
                    carb_text = correct_text_errors(carb_text,value_handling_error2)
                if carb_text.endswith('.'):
                    carb_text += '0'
            else:
                carb_text = placeholder
                #additional_info_results.append(('carbone', text))

            chlorine_in_mgr_result_cor = extract_info(stdout_str, 'chlorine in mgr result')
            if chlorine_in_mgr_result_cor:
                print("chlorine")
                x1,y1,x2,y2 = chlorine_in_mgr_result_cor

                print(chlorine_in_mgr_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                chlo_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                chlorine_in_mgr_result = reader.readtext(chlo_roi)
                for t_, t in enumerate(chlorine_in_mgr_result):
                    bbox, text, score = t
                    t = bbox, chlorine_in_mgr_result, score
                    print(t)
                chlorine_in_mgr_result.sort(key=lambda x: x[0][0][0])
                chlo_text = ' '.join([text for bbox, text, score in chlorine_in_mgr_result])
                if ',' not in chlo_text:
                    if '  '  in chlo_text:
                            chlo_text = chlo_text.replace('  ', '.') 
                    elif ' ' in chlo_text:
                            chlo_text = chlo_text.replace(' ', '.') 
                temp = []
                for word in chlo_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    chlo_text = ''.join(temp)
                    chlo_text = correct_text_errors(chlo_text,value_handling_error2)
                if chlo_text.endswith('.'):
                    chlo_text += '0'
            else:
                chlo_text = placeholder
                #additional_info_results.append(('chlorine', text))

            salt_concentration_in_microsieme_results_cor = extract_info(stdout_str, 'salt concentration in microsieme resultns')
            if salt_concentration_in_microsieme_results_cor:
                print("salt concentration")
                x1,y1,x2,y2 = salt_concentration_in_microsieme_results_cor 

                print(salt_concentration_in_microsieme_results_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                salt_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                salt_concentration_in_microsieme_results = reader.readtext(salt_roi)
                for t_, t in enumerate(salt_concentration_in_microsieme_results):
                    bbox, text, score = t
                    t = bbox, salt_concentration_in_microsieme_results, score
                    print(t)
                salt_concentration_in_microsieme_results.sort(key=lambda x: x[0][0][0])
                salt_text = ' '.join([text for bbox, text, score in salt_concentration_in_microsieme_results])
                if ',' not in salt_text and '_' not in salt_text:
                    if '  '  in salt_text:
                            salt_text = salt_text.replace('  ', '.') 
                    elif ' ' in salt_text:
                            salt_text = salt_text.replace(' ', '.') 
                temp = []
                for word in salt_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    salt_text = ''.join(temp)
                    salt_text = correct_text_errors(salt_text,value_handling_error2)
                if salt_text.endswith('.'):
                    salt_text += '0'
            else:
                salt_text = placeholder
                #additional_info_results.append(('salt concentration', text))

            phossoul_result_cor = extract_info(stdout_str, 'phossoul result')
            if phossoul_result_cor:
                print("phossoul")
                x1,y1,x2,y2 = phossoul_result_cor

                print(phossoul_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
            # Extract the region of interest (ROI) from the frame
                pho_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                phossoul_result = reader.readtext(pho_roi)
                for t_, t in enumerate(phossoul_result):
                    bbox, text, score = t
                    t = bbox, phossoul_result, score
                    print(t)
                phossoul_result.sort(key=lambda x: x[0][0][0])

                phossoul_text = ' '.join([text for bbox, text, score in phossoul_result])
                #phossoul_text = phossoul_text.replace(' , ', '.')
                #phossoul_text = phossoul_text.replace(' ', '.')
                print(phossoul_text)
                if ',' not in phossoul_text and '_' not in phossoul_text:
                    if '  '  in phossoul_text:
                            phossoul_text = phossoul_text.replace('  ', '.') 
                    elif ' ' in phossoul_text:
                            phossoul_text = phossoul_text.replace(' ', '.')  # Replace single spaces with periods
                    
                print(phossoul_text)
                temp = []
                for word in phossoul_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    phossoul_text = ''.join(temp)
                    print(phossoul_text)
                    phossoul_text = correct_text_errors(phossoul_text,value_handling_error2)
                if phossoul_text.endswith('.'):
                    phossoul_text += '0'
                print(phossoul_text)
            else:
                phossoul_text = placeholder
                #additional_info_results.append(('phossoul', text))

            copper_result_cor = extract_info(stdout_str, 'copper result')
            if copper_result_cor:
                print("copper")
                x1,y1,x2,y2 = copper_result_cor

                print(copper_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                cup_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                copper_result = reader.readtext(cup_roi)
                for t_, t in enumerate(copper_result):
                    bbox, text, score = t
                    t = bbox, copper_result, score
                    print(t)
                copper_result.sort(key=lambda x: x[0][0][0])
                copp_text = ' '.join([text for bbox, text, score in copper_result])
                if ',' not in copp_text and '_' not in copp_text:
                    if '  '  in copp_text:
                            copp_text = copp_text.replace('  ', '.') 
                    elif ' ' in copp_text:
                            copp_text = copp_text.replace(' ', '.') 
                temp = []
                for word in copp_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    copp_text = ''.join(temp)
                    copp_text = correct_text_errors(copp_text,value_handling_error2)
                if copp_text.endswith('.'):
                    copp_text += '0'
            else:
                copp_text = placeholder

                #additional_info_results.append(('copper', text))

            zinc_result_cor = extract_info(stdout_str, 'zinc result')
            if zinc_result_cor:
                print("zinc")
                x1,y1,x2,y2 = zinc_result_cor

                print(zinc_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                zinc_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                zinc_result = reader.readtext(zinc_roi)
                for t_, t in enumerate(zinc_result):
                    bbox, text, score = t
                    t = bbox, zinc_result, score
                    print(t)
                zinc_result.sort(key=lambda x: x[0][0][0])
                zinc_text = ' '.join([text for bbox, text, score in zinc_result])
                if ',' not in zinc_text and '_' not in zinc_text:
                    if '  '  in zinc_text:
                            zinc_text = zinc_text.replace('  ', '.') 
                    elif ' ' in zinc_text:
                            zinc_text = zinc_text.replace(' ', '.') 
                temp = []
                for word in zinc_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    zinc_text = ''.join(temp)
                    zinc_text = correct_text_errors(zinc_text,value_handling_error2)
                if zinc_text.endswith('.'):
                    zinc_text += '0'
            else:
                zinc_text = placeholder
            
                #additional_info_results.append(('zinc', text))
        if laboratory_sample_result_cor:
            print(laboratory_sample_result_cor)
            x1, y1, x2, y2 = laboratory_sample_result_cor

            tl_corner = (int(x1), int(y1))
            br_corner = (int(x2), int(y2))
            print(tl_corner)
            print(br_corner)
            # Extract the region of interest (ROI) from the frame
            lab_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
            #cv2.imshow('ana ROI', ana_roi)
            laboratory_sample_result = reader.readtext(lab_roi)
            for t_, t in enumerate(laboratory_sample_result):
                bbox, text, score = t
                t = bbox, laboratory_sample_result, score
                print(t)
            laboratory_sample_result_text = ''.join([text for bbox, text, score in laboratory_sample_result])
        # laboratory_sample_result_text = correct_text_errors(laboratory_sample_result_text,value_handling_error2)
            analysis_number_text = laboratory_sample_result_text
        else:
            laboratory_sample_result_text = placeholder
        if laboratory_sample_result_text == placeholder:
            print("No laboratory sample detected.")
            
        else:

                ##additional_info_results.append(('Analysis number', text))
            date_cor = extract_info(stdout_str, 'date')
            print(date_cor)
            if date_cor:
                print("date")
                print(date_cor)
                x1, y1, x2, y2 = date_cor
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                print(tl_corner)
                print(br_corner)
            # Extract the region of interest (ROI) from the frame
                date_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                #cv2.imshow('Date ROI', date_roi)
                date = reader.readtext(date_roi)
                for t_, t in enumerate(date):
                    bbox, text, score = t
                    t = bbox, date, score
                    print(t)
                date.sort(key=lambda x: x[0][0][0])
                date_text = ' '.join([text for bbox, text, score in date])
                temp = []
                for word in date_text.split():
                    temp.append(num_date.get(word, word))
                    date_text = ''.join(temp)
                date_text = datetime.strptime(date_text, "%d-%b-%Y")
                date_text = date_text.strftime("%Y-%m-%d")

            else:
                date_text = placeholder
            information_sheet_cor = extract_info(stdout_str, 'information sheet result')
            if information_sheet_cor:
                print("information sheet result")
                x1,y1,x2,y2 = information_sheet_cor
                print(information_sheet_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                info_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                information_sheet_result = reader.readtext(info_roi)
                for t_, t in enumerate(information_sheet_result):
                    bbox, text, score = t
                    t = bbox, information_sheet_result, score
                    print(t)
                information_sheet_result.sort(key=lambda x: x[0][0][0])
                information_sheet_text = ' '.join([text for bbox, text, score in information_sheet_result])
                if ',' not in information_sheet_text and '_' not in information_sheet_text:
                    if '  '  in information_sheet_text:
                            information_sheet_text = information_sheet_text.replace('  ', '.') 
                    elif ' ' in information_sheet_text:
                            information_sheet_text = information_sheet_text.replace(' ', '.') 
                temp = []
                for word in information_sheet_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    information_sheet_text = ''.join(temp)
                    information_sheet_text = correct_text_errors(information_sheet_text,value_handling_error2)
                if information_sheet_text.endswith('.'):
                    information_sheet_text += '0'
            else:
                information_sheet_text = placeholder
            submitted_for_cor = extract_info(stdout_str, 'submitted for result')
            if submitted_for_cor:
                print("submitted for")
                x1,y1,x2,y2 = submitted_for_cor
                print(submitted_for_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                submitted_for_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                submitted_for_result = reader.readtext(submitted_for_roi)
                for t_, t in enumerate(submitted_for_result):
                    bbox, text, score = t
                    t = bbox, submitted_for_result, score
                    print(t)
                submitted_for_result.sort(key=lambda x: x[0][0][0])
                submitted_for_text = ' '.join([text for bbox, text, score in submitted_for_result])
                if ',' not in submitted_for_text and '_' not in submitted_for_text:
                    if '  '  in submitted_for_text:
                            submitted_for_text = submitted_for_text.replace('  ', '.') 
                    elif ' ' in submitted_for_text:
                            submitted_for_text = submitted_for_text.replace(' ', '.') 
                temp = []
                for word in submitted_for_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    submitted_for_text = ''.join(temp)
                    submitted_for_text = correct_text_errors(submitted_for_text,value_handling_error2)
                if submitted_for_text.endswith('.'):
                    submitted_for_text += '0'
            else:
                submitted_for_text = placeholder
            submitted_by_cor = extract_info(stdout_str, 'submitted by result')
            if submitted_by_cor:
                print("submitted by result")
                x1,y1,x2,y2 = submitted_by_cor
                print(submitted_by_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                submitted_by_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                submitted_by_result = reader.readtext(submitted_by_roi)
                for t_, t in enumerate(submitted_by_result):
                    bbox, text, score = t
                    t = bbox, submitted_by_result, score
                    print(t)
                submitted_by_result.sort(key=lambda x: x[0][0][0])
                submitted_by_text = ' '.join([text for bbox, text, score in submitted_by_result])
                if ',' not in submitted_by_text and '_' not in submitted_by_text:
                    if '  '  in submitted_by_text:
                            submitted_by_text = submitted_by_text.replace('  ', '.') 
                    elif ' ' in submitted_by_text:
                            submitted_by_text = submitted_by_text.replace(' ', '.') 
                temp = []
                for word in submitted_by_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    submitted_by_text = ''.join(temp)
                    submitted_by_text = correct_text_errors(submitted_by_text,value_handling_error2)
                if submitted_by_text.endswith('.'):
                    submitted_by_text += '0'
            else:
                submitted_by_text = placeholder
            crop_cor = extract_info(stdout_str, 'crop result')
            if crop_cor:
                print("crop result")
                x1,y1,x2,y2 = crop_cor
                print(crop_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                crop_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                crop_result = reader.readtext(crop_roi)
                for t_, t in enumerate(crop_result):
                    bbox, text, score = t
                    t = bbox, crop_result, score
                    print(t)
                crop_result.sort(key=lambda x: x[0][0][0])
                crop_text = ' '.join([text for bbox, text, score in crop_result])
                #for word in crop_text.split():
                        #temp.append(spelling_errors.get(word, word))
                        #temp.append(num_date.get(word, word))
                        #crop_text = ' '.join(temp)
                    
                if crop_text.endswith('.'):
                    crop_text += '0'
                NOP_text = crop_text
            else:
                crop_text = placeholder
            sample_marked_cor = extract_info(stdout_str, 'sample marked result')
            if sample_marked_cor:
                print("sample marked result")
                x1,y1,x2,y2 = sample_marked_cor
                print(sample_marked_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                sample_marked_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                sample_marked_result = reader.readtext(sample_marked_roi)
                for t_, t in enumerate(sample_marked_result):
                    bbox, text, score = t
                    t = bbox, sample_marked_result, score
                    print(t)
                sample_marked_result.sort(key=lambda x: x[0][0][0])
                sample_marked_text = ' '.join([text for bbox, text, score in sample_marked_result])
                
                temp = []
                for word in sample_marked_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    sample_marked_text = ''.join(temp)
                    #sample_marked_text = correct_text_errors(sample_marked_text,value_handling_error2)
                if sample_marked_text.endswith('.'):
                    sample_marked_text += '0'
            else:
                sample_marked_text = placeholder
            sample_will_be_stored_result_cor = extract_info(stdout_str, 'sample will be stored result')
            if sample_will_be_stored_result_cor:
                print("sample will be stored result")
                x1,y1,x2,y2 = sample_will_be_stored_result_cor
                print(sample_will_be_stored_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                sample_will_be_stored_result_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                sample_will_be_stored_result = reader.readtext(sample_will_be_stored_result_roi)
                for t_, t in enumerate(sample_will_be_stored_result):
                    bbox, text, score = t
                    t = bbox, sample_will_be_stored_result, score
                    print(t)
                sample_will_be_stored_result.sort(key=lambda x: x[0][0][0])
                sample_stored_text = ' '.join([text for bbox, text, score in sample_will_be_stored_result])
                if ',' not in sample_stored_text and '_' not in sample_stored_text:
                    if '  '  in sample_stored_text:
                            sample_stored_text = sample_stored_text.replace('  ', '.') 
                    elif ' ' in sample_stored_text:
                            sample_stored_text = sample_stored_text.replace(' ', '.') 
                temp = []
                for word in sample_stored_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    sample_stored_text = ''.join(temp)
                    #sample_stored_text = correct_text_errors(sample_stored_text,value_handling_error2)
                if sample_stored_text.endswith('.'):
                    sample_stored_text += '0'
            else:
                sample_stored_text = placeholder
            Nitrogen_cor = extract_info(stdout_str, 'Nitrogen result')
            if Nitrogen_cor:
                print("Nitrogen result")
                x1,y1,x2,y2 = Nitrogen_cor
                print(Nitrogen_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                nitro_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                nitro_result = reader.readtext(nitro_roi)
                for t_, t in enumerate(nitro_result):
                    bbox, text, score = t
                    t = bbox, nitro_result, score
                    print(t)
                nitro_result.sort(key=lambda x: x[0][0][0])
                nitro_text = ' '.join([text for bbox, text, score in nitro_result])
                if ',' not in nitro_text and '_' not in nitro_text:
                    if '  '  in nitro_text:
                            nitro_text = nitro_text.replace('  ', '.') 
                    elif ' ' in nitro_text:
                            nitro_text = nitro_text.replace(' ', '.') 
                temp = []
                for word in nitro_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    nitro_text = ''.join(temp)
                    nitro_text = correct_text_errors(nitro_text,value_handling_error2)
                if nitro_text.endswith('.'):
                    nitro_text += '0'
            else:
                nitro_text = placeholder
            sulfer_cor = extract_info(stdout_str, 'sulfer result')
            if sulfer_cor:
                print("sulfer result")
                x1,y1,x2,y2 = sulfer_cor
                print(sulfer_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                sulfer_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                sulfer_result = reader.readtext(sulfer_roi)
                for t_, t in enumerate(sulfer_result):
                    bbox, text, score = t
                    t = bbox, sulfer_result, score
                    print(t)
                sulfer_result.sort(key=lambda x: x[0][0][0])
                sulfer_text = ' '.join([text for bbox, text, score in sulfer_result])
                if ',' not in sulfer_text and '_' not in sulfer_text:
                    if '  '  in sulfer_text:
                            sulfer_text = sulfer_text.replace('  ', '.') 
                    elif ' ' in sulfer_text:
                            sulfer_text = sulfer_text.replace(' ', '.') 
                temp = []
                for word in sulfer_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    sulfer_text = ''.join(temp)
                    sulfer_text = correct_text_errors(sulfer_text,value_handling_error2)
                if sulfer_text.endswith('.'):
                    sulfer_text += '0'
            else:
                sulfer_text = placeholder
            boron_cor = extract_info(stdout_str, 'boron result')
            if boron_cor:
                print("boron result")
                x1,y1,x2,y2 = boron_cor
                print(boron_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                boron_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                boron = reader.readtext(boron_roi)
                for t_, t in enumerate(boron):
                    bbox, text, score = t
                    t = bbox, boron, score
                    print(t)
                boron.sort(key=lambda x: x[0][0][0])
                boron_text = ' '.join([text for bbox, text, score in boron])
                if ',' not in boron_text and '_' not in boron_text:
                    if '  '  in boron_text:
                            boron_text = boron_text.replace('  ', '.') 
                    elif ' ' in boron_text:
                            boron_text = boron_text.replace(' ', '.') 
                temp = []
                for word in boron_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    boron_text = ''.join(temp)
                    boron_text = correct_text_errors(boron_text,value_handling_error2)
                if boron_text.endswith('.'):
                    boron_text += '0'
            else:
                boron_text = placeholder
            Aluminium_cor = extract_info(stdout_str, 'Aluminium result')
            if Aluminium_cor:
                print("Aluminium result")
                x1,y1,x2,y2 = Aluminium_cor
                print(Aluminium_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                Aluminium_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                Aluminium = reader.readtext(Aluminium_roi)
                for t_, t in enumerate(Aluminium):
                    bbox, text, score = t
                    t = bbox, Aluminium, score
                    print(t)
                Aluminium.sort(key=lambda x: x[0][0][0])
                Aluminium_text = ' '.join([text for bbox, text, score in Aluminium])
                if ',' not in Aluminium_text and '_' not in Aluminium_text:
                    if '  '  in Aluminium_text:
                            Aluminium_text = Aluminium_text.replace('  ', '.') 
                    elif ' ' in Aluminium_text:
                            Aluminium_text = Aluminium_text.replace(' ', '.') 
                temp = []
                for word in Aluminium_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    Aluminium_text = ''.join(temp)
                    Aluminium_text = correct_text_errors(Aluminium_text,value_handling_error2)
                if Aluminium_text.endswith('.'):
                    Aluminium_text += '0'
            else:
                Aluminium_text = placeholder
            phosphore_result_cor = extract_info(stdout_str, 'phosphore result')
            if phosphore_result_cor:
                print("phosphore_result")
                print(phosphore_result_cor)
                x1, y1, x2, y2 = phosphore_result_cor
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                phos_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                phosphore_result = reader.readtext(phos_roi)
                for t_, t in enumerate(phosphore_result):
                    bbox, text, score = t
                    t = bbox, phosphore_result, score
                    print(t)
                phosphore_result.sort(key=lambda x: x[0][0][0])
                phos_text = ' '.join([text for bbox, text, score in phosphore_result])
                if ',' not in phos_text and '_' not in phos_text:
                    if '  '  in phos_text:
                            phos_text = phos_text.replace('  ', '.') 
                    elif ' ' in phos_text:
                            phos_text = phos_text.replace(' ', '.') 
                temp = []
                for word in phos_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    phos_text = ''.join(temp) 
                    phos_text = correct_text_errors(phos_text,value_handling_error2)
                if phos_text.endswith('.'):
                    phos_text += '0'
            else:
                phos_text = placeholder
            # #additional_info_results.append(('phosphore', text))

            potassium_result_cor = extract_info(stdout_str, 'potassium result')
            if potassium_result_cor:
                print("potassium result")
                print(potassium_result_cor)
                try:
                    x1, y1, x2, y2 = potassium_result_cor
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    pota_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    potassium_result = reader.readtext(pota_roi)
                    for t_, t in enumerate(potassium_result):
                        bbox, text, score = t
                        t = bbox, potassium_result, score
                        print(t)
                    potassium_result.sort(key=lambda x: x[0][0][0])
                    pota_text = ' '.join([text for bbox, text, score in potassium_result])
                    if ',' not in pota_text and '_' not in pota_text:
                        if '  '  in pota_text:
                                pota_text = pota_text.replace('  ', '.') 
                        elif ' ' in pota_text:
                                pota_text = pota_text.replace(' ', '.') 
                    temp = []
                    for word in pota_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        pota_text = ''.join(temp)
                        pota_text = correct_text_errors(pota_text,value_handling_error2)
                    if pota_text.endswith('.'):
                        pota_text += '0'
                except ValueError:
                        pota_text = placeholder
            else:
                pota_text = placeholder
                #additional_info_results.append(('potassium', text))

            magnesium_result_cor = extract_info(stdout_str, 'magnesium result')
            if magnesium_result_cor:
                print("magnesium_result")
                print(magnesium_result_cor)
                x1,y1,x2,y2 = magnesium_result_cor
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                mag_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                magnesium_result = reader.readtext(mag_roi)
                for t_, t in enumerate(magnesium_result):
                    bbox, text, score = t
                    t = bbox, magnesium_result, score
                    print(t)
                magnesium_result.sort(key=lambda x: x[0][0][0])
                mag_text = ' '.join([text for bbox, text, score in magnesium_result])
                if ',' not in mag_text and '_' not in mag_text:
                    if '  '  in mag_text:
                            mag_text = mag_text.replace('  ', '.') 
                    elif ' ' in mag_text:
                            mag_text = mag_text.replace(' ', '.') 
                temp = []
                for word in mag_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    mag_text = ''.join(temp)
                
                    mag_text = correct_text_errors(mag_text,value_handling_error2) 
                if mag_text.endswith('.'):
                    mag_text += '0'
            else:
                mag_text = placeholder
                #additional_info_results.append(('magnesium', text))

            sodium_result_cor = extract_info(stdout_str, 'sodium result')
            if sodium_result_cor:
                print("sodium_result")
                x1,y1,x2,y2 = sodium_result_cor
                print(sodium_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
            # Extract the region of interest (ROI) from the frame
                sod_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                sodium_result = reader.readtext(sod_roi)
                for t_, t in enumerate(sodium_result):
                    bbox, text, score = t
                    t = bbox, sodium_result, score
                    print(t)
                sodium_result.sort(key=lambda x: x[0][0][0])
                sod_text = ' '.join([text for bbox, text, score in sodium_result])
                if ',' not in sod_text and '_' not in sod_text:
                    if '  '  in sod_text:
                            sod_text = sod_text.replace('  ', '.') 
                    elif ' ' in sod_text:
                            sod_text = sod_text.replace(' ', '.') 
                temp = []
                for word in sod_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    sod_text = ''.join(temp)
                    sod_text = correct_text_errors(sod_text,value_handling_error2)
                if sod_text.endswith('.'):
                    sod_text += '0'
            else:
                sod_text = placeholder
                #additional_info_results.append(('sodium', text))

            calcium_result_cor = extract_info(stdout_str, 'calcium result')
            if calcium_result_cor:
                print("calcium result")
                x1,y1,x2,y2 = calcium_result_cor
                print(calcium_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                cal_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                calcium_result = reader.readtext(cal_roi)
                for t_, t in enumerate(calcium_result):
                    bbox, text, score = t
                    t = bbox, calcium_result, score
                    print(t)
                calcium_result.sort(key=lambda x: x[0][0][0])
                cal_text = ' '.join([text for bbox, text, score in calcium_result])
                if ',' not in cal_text and '_' not in cal_text:
                    if '  '  in cal_text:
                            cal_text = cal_text.replace('  ', '.') 
                    elif ' ' in cal_text:
                            cal_text = cal_text.replace(' ', '.') 
                temp = []
                for word in cal_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    cal_text = ''.join(temp)
                    cal_text = correct_text_errors(cal_text,value_handling_error2)
                if cal_text.endswith('.'):
                    cal_text += '0'
            else:
                cal_text = placeholder
            manganese_result_cor = extract_info(stdout_str, 'manganese result')
            if manganese_result_cor:
                print("manganese result")
                print(manganese_result_cor)
                try:
                    x1,y1,x2,y2 = manganese_result_cor

                    
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    man_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    manganese_result = reader.readtext(man_roi)
                    for t_, t in enumerate(manganese_result):
                        bbox, text, score = t
                        t = bbox, manganese_result, score
                        print(t)
                    manganese_result.sort(key=lambda x: x[0][0][0])
                    man_text = ' '.join([text for bbox, text, score in manganese_result])
                    print(man_text)
                    if ',' not in man_text and '_' not in man_text:
                        if '  '  in man_text:
                                man_text = man_text.replace('  ', '.') 
                        elif ' ' in man_text:
                                man_text = man_text.replace(' ', '.') 
                    print(man_text)

                    temp = []
                    for word in man_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        man_text = ''.join(temp)
                        man_text = correct_text_errors(man_text,value_handling_error2)
                    if man_text.endswith('.'):
                        man_text += '0'
                except ValueError:
                    man_text = placeholder
            else:
                man_text = placeholder
                #additional_info_results.append(('manganese', text))

            Iron_result_cor = extract_info(stdout_str, 'Iron result')
            if Iron_result_cor:
                print("iron")
                x1,y1,x2,y2 = Iron_result_cor

                print(Iron_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                iron_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                Iron_result = reader.readtext(iron_roi)
                for t_, t in enumerate(Iron_result):
                    bbox, text, score = t
                    t = bbox, Iron_result, score
                    print(t)
                Iron_result.sort(key=lambda x: x[0][0][0])
                iron_text = ' '.join([text for bbox, text, score in Iron_result])
                if ',' not in iron_text and '_' not in iron_text:
                    if '  '  in iron_text:
                            iron_text = iron_text.replace('  ', '.') 
                    elif ' ' in iron_text:
                            iron_text = iron_text.replace(' ', '.') 
                temp = []
                for word in iron_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    iron_text = ''.join(temp)
                    iron_text = correct_text_errors(iron_text,value_handling_error2)
                if iron_text.endswith('.'):
                    iron_text += '0'
            else:
                iron_text = placeholder
                #additional_info_results.append(('calcium', text))
            copper_result_cor = extract_info(stdout_str, 'copper result')
            if copper_result_cor:
                print("copper")
                x1,y1,x2,y2 = copper_result_cor

                print(copper_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                cup_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                copper_result = reader.readtext(cup_roi)
                for t_, t in enumerate(copper_result):
                    bbox, text, score = t
                    t = bbox, copper_result, score
                    print(t)
                copper_result.sort(key=lambda x: x[0][0][0])
                copp_text = ' '.join([text for bbox, text, score in copper_result])
                if ',' not in copp_text and '_' not in copp_text:
                    if '  '  in copp_text:
                            copp_text = copp_text.replace('  ', '.') 
                    elif ' ' in copp_text:
                            copp_text = copp_text.replace(' ', '.') 
                temp = []
                for word in copp_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    copp_text = ''.join(temp)
                    copp_text = correct_text_errors(copp_text,value_handling_error2)
                if copp_text.endswith('.'):
                    copp_text += '0'
            else:
                copp_text = placeholder

                #additional_info_results.append(('copper', text))

            zinc_result_cor = extract_info(stdout_str, 'zinc result')
            if zinc_result_cor:
                print("zinc")
                x1,y1,x2,y2 = zinc_result_cor

                print(zinc_result_cor)
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                zinc_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                zinc_result = reader.readtext(zinc_roi)
                for t_, t in enumerate(zinc_result):
                    bbox, text, score = t
                    t = bbox, zinc_result, score
                    print(t)
                zinc_result.sort(key=lambda x: x[0][0][0])
                zinc_text = ' '.join([text for bbox, text, score in zinc_result])
                if ',' not in zinc_text and '_' not in zinc_text:
                    if '  '  in zinc_text:
                            zinc_text = zinc_text.replace('  ', '.') 
                    elif ' ' in zinc_text:
                            zinc_text = zinc_text.replace(' ', '.') 
                temp = []
                for word in zinc_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    zinc_text = ''.join(temp)
                    zinc_text = correct_text_errors(zinc_text,value_handling_error2)
                if zinc_text.endswith('.'):
                    zinc_text += '0'
            else:
                zinc_text = placeholder

            
        if  analysis_chemical_results_cor or zone_area_cor:
            if analysis_chemical_results_cor:
                x1, y1, x2, y2 = analysis_chemical_results_cor 

                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                print(tl_corner)
                print(br_corner)
                # Extract the region of interest (ROI) from the frame
                anach_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                #cv2.imshow('ana ROI', ana_roi)
                Analysisch_number_result = reader.readtext(anach_roi)
                for t_, t in enumerate(Analysisch_number_result):
                    bbox, text, score = t
                    t = bbox, Analysisch_number_result, score
                    print(t)
                analysisch_number_text = ''.join([text for bbox, text, score in Analysisch_number_result])
                analysisch_number_text = correct_text_errors(analysisch_number_text,value_handling_error3)
                analysis_number_text = analysisch_number_text
            elif zone_area_cor:
                print('no analysis chemical number but info still important')
                analysis_number_text = "N/A"+ str(y)
                print("yo loook here!!!!" +str(y))
            date_cor = extract_info(stdout_str, 'date')
            print(date_cor)
            if date_cor:
                print("date")
                print(date_cor)
                x1, y1, x2, y2 = date_cor
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
                print(tl_corner)
                print(br_corner)
            # Extract the region of interest (ROI) from the frame
                date_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                #cv2.imshow('Date ROI', date_roi)
                date = reader.readtext(date_roi)
                for t_, t in enumerate(date):
                    bbox, text, score = t
                    t = bbox, date, score
                    print(t)
                date.sort(key=lambda x: x[0][0][0])
                date_text = ' '.join([text for bbox, text, score in date])
                temp = []
                for word in date_text.split():
                    temp.append(num_date.get(word, word))
                    date_text = ''.join(temp)
                try:
                    # Attempt to convert the date to the specified format
                    date_text = datetime.strptime(date_text, "%d/%m/%Y")
                    # If successful, format the date in YYYY-MM-DD
                    date_text = date_text.strftime("%Y-%m-%d")
                except ValueError:
                    # If the date is not in the specified format, assign "N/A"
                    date_text = placeholder
            else:
                date_text = placeholder
            Name_of_the_plot_cor = extract_info(stdout_str, 'Name of the plot')
            if Name_of_the_plot_cor:
                print("name of plot:")
                print(Name_of_the_plot_cor)
                try:
                    x1, y1, x2, y2 = Name_of_the_plot_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    NOP_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    Name_of_the_plot = reader.readtext(NOP_roi)
                    for t_, t in enumerate(Name_of_the_plot):
                        bbox, text, score = t
                        t = bbox, Name_of_the_plot, score
                        print(t)
                    NOP_text = ' '.join([text for bbox, text, score in Name_of_the_plot])
                    temp = []
                    for word in NOP_text.split():
                        temp.append(spelling_errors.get(word, word))
                        #temp.append(num_date.get(word, word))
                        NOP_text = ' '.join(temp)
                except ValueError:
                    NOP_text = placeholder          
            else:
                NOP_text = placeholder 
            depth_cor = extract_info(stdout_str, 'Depth_result')
            if depth_cor:
                print("Depth:")
                print(depth_cor)
                
                x1, y1, x2, y2 = depth_cor 
                tl_corner = (int(x1), int(y1))
                br_corner = (int(x2), int(y2))
            # Extract the region of interest (ROI) from the frame
                depth_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                depth = reader.readtext(depth_roi)
                for t_, t in enumerate(depth):
                    bbox, text, score = t
                    t = bbox, depth, score
                    print(t)
                depth.sort(key=lambda x: x[0][0][0])
                depth_text = ' '.join([text for bbox, text, score in depth])
                if ',' not in depth_text and '_' not in depth_text:
                    if '  '  in depth_text:
                            depth_text = depth_text.replace('  ', '.') 
                    elif ' ' in depth_text:
                            depth_text = depth_text.replace(' ', '.') 
                temp = []
                for word in depth_text.split():
                    temp.append(value_handling_error.get(word, word))
                    #temp.append(num_date.get(word, word))
                    depth_text = ''.join(temp)
                    depth_text = correct_text_errors(depth_text,value_handling_error2)
                if depth_text.endswith('.'):
                    depth_text += '0'
                else:
                    depth_text = placeholder  
            zone_cor = extract_info(stdout_str, 'zone result')
            phosphore_result_cors = extract_info(stdout_str, 'phosphore result')
            potassium_result_cors = extract_info(stdout_str, 'potassium result')
            magnesium_result_cor = extract_info(stdout_str, 'magnesium result')
            calcium_result_cors = extract_info(stdout_str, 'calcium result')
            PH_at_KCL_results_cors = extract_info(stdout_str, 'PH at KCL results')
            HUMUS_percent_result_cors = extract_info(stdout_str, 'HUMUS % result')
            Iron_result_cors = extract_info(stdout_str, 'Iron result')
            magnesium_result_cors = extract_info(stdout_str, 'magnesium result')
            manganese_result_cors = extract_info(stdout_str, 'manganese result')
            sodium_result_cors = extract_info(stdout_str, 'sodium result')
            nt_cors = extract_info(stdout_str, 'Nt result')
            ph_acetate_cors = extract_info(stdout_str, 'ph acetate result')
            taux_argile_cors = extract_info(stdout_str, 'Taux d argile result')
            CEC_cors = extract_info(stdout_str, 'CEC (cmol/kg) result')
            reportCN_cors = extract_info(stdout_str, 'report C/N result')
            reportKMG_cors = extract_info(stdout_str, 'Repport K/Mg result')
            reportCAMG_cors = extract_info(stdout_str, 'Repport Ca/Mg result')

            if zone_cor:
                print("zone result:")
                print(zone_cor)
                num_sets = len(zone_cor) // 4
                print(num_sets)
                if num_sets > 1:
                    analysis_number_text = "N/A"+ str(y)
                    print("yo loook here!!!!" +str(y))
                    coordinate_tackle(y, api_data, page,frame,extract_info,stdout_str,reader, num_sets,zone_cor,phosphore_result_cors,potassium_result_cors,magnesium_result_cors,calcium_result_cors,PH_at_KCL_results_cors,HUMUS_percent_result_cors,Iron_result_cors,manganese_result_cors,sodium_result_cors,nt_cors,ph_acetate_cors,taux_argile_cors,CEC_cors,reportCN_cors,reportKMG_cors,reportCAMG_cors)
                    json_flag = True
                elif num_sets == 1:
                    analysis_number_text = "N/A"+ str(y)
                    print("yo loook here!!!!" +str(y))
                    json_flag = False
                    x1, y1, x2, y2 = zone_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    zone_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    zone = reader.readtext(zone_roi)
                    for t_, t in enumerate(zone):
                        bbox, text, score = t
                        t = bbox, zone, score
                        print(t)
                    zone_text = ' '.join([text for bbox, text, score in zone])
                    temp = []
                    for word in zone_text.split():
                        temp.append(spelling_errors.get(word, word))
                        #temp.append(num_date.get(word, word))
                        zone_text = ' '.join(temp)
                    NOP_text = zone_text           
                else:
                    zone_text = placeholder
            if json_flag == False:    
                phosphore_result_cor = extract_info(stdout_str, 'phosphore result')
                if phosphore_result_cor:
                    print("phosphore_result")
                    print(phosphore_result_cor)
                    try:
                        x1, y1, x2, y2 = phosphore_result_cor
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                        # Extract the region of interest (ROI) from the frame
                        phos_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        phosphore_result = reader.readtext(phos_roi)
                        for t_, t in enumerate(phosphore_result):
                            bbox, text, score = t
                            t = bbox, phosphore_result, score
                            print(t)
                        phosphore_result.sort(key=lambda x: x[0][0][0])
                        phos_text = ' '.join([text for bbox, text, score in phosphore_result])
                        if ',' not in phos_text and '_' not in phos_text:
                            if '  '  in phos_text:
                                    phos_text = phos_text.replace('  ', '.') 
                            elif ' ' in phos_text:
                                    phos_text = phos_text.replace(' ', '.') 
                        temp = []
                        for word in phos_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            phos_text = ''.join(temp) 
                            phos_text = correct_text_errors(phos_text,value_handling_error2)
                        if phos_text.endswith('.'):
                            phos_text += '0'
                    except ValueError:
                        phos_text = placeholder
                else:
                    phos_text = placeholder
                potassium_result_cor = extract_info(stdout_str, 'potassium result')
                if potassium_result_cor:
                    print("potassium result")
                    print(potassium_result_cor)
                    try:
                        x1, y1, x2, y2 = potassium_result_cor
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                        # Extract the region of interest (ROI) from the frame
                        pota_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        potassium_result = reader.readtext(pota_roi)
                        for t_, t in enumerate(potassium_result):
                            bbox, text, score = t
                            t = bbox, potassium_result, score
                            print(t)
                        potassium_result.sort(key=lambda x: x[0][0][0])
                        pota_text = ' '.join([text for bbox, text, score in potassium_result])
                        if ',' not in pota_text and '_' not in pota_text:
                            if '  '  in pota_text:
                                    pota_text = pota_text.replace('  ', '.') 
                            elif ' ' in pota_text:
                                    pota_text = pota_text.replace(' ', '.') 
                        temp = []
                        for word in pota_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            pota_text = ''.join(temp)
                            pota_text = correct_text_errors(pota_text,value_handling_error2)
                        if pota_text.endswith('.'):
                            pota_text += '0'
                    except ValueError:
                            pota_text = placeholder
                else:
                    pota_text = placeholder
                magnesium_result_cor = extract_info(stdout_str, 'magnesium result')
                if magnesium_result_cor:
                    print("magnesium_result")
                    print(magnesium_result_cor)
                    try:
                        x1,y1,x2,y2 = magnesium_result_cor
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                        # Extract the region of interest (ROI) from the frame
                        mag_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        magnesium_result = reader.readtext(mag_roi)
                        for t_, t in enumerate(magnesium_result):
                            bbox, text, score = t
                            t = bbox, magnesium_result, score
                            print(t)
                        magnesium_result.sort(key=lambda x: x[0][0][0])
                        mag_text = ' '.join([text for bbox, text, score in magnesium_result])
                        if ',' not in mag_text and '_' not in mag_text:
                            if '  '  in mag_text:
                                    mag_text = mag_text.replace('  ', '.') 
                            elif ' ' in mag_text:
                                    mag_text = mag_text.replace(' ', '.') 
                        temp = []
                        for word in mag_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            mag_text = ''.join(temp)
                        
                            mag_text = correct_text_errors(mag_text,value_handling_error2) 
                        if mag_text.endswith('.'):
                            mag_text += '0'
                    except ValueError:
                        mag_text = placeholder
                else:
                    mag_text = placeholder
                calcium_result_cor = extract_info(stdout_str, 'calcium result')
                if calcium_result_cor:
                    print("calcium result")
                    try:
                        x1,y1,x2,y2 = calcium_result_cor
                        print(calcium_result_cor)
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                        # Extract the region of interest (ROI) from the frame
                        cal_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        calcium_result = reader.readtext(cal_roi)
                        for t_, t in enumerate(calcium_result):
                            bbox, text, score = t
                            t = bbox, calcium_result, score
                            print(t)
                        calcium_result.sort(key=lambda x: x[0][0][0])
                        cal_text = ' '.join([text for bbox, text, score in calcium_result])
                        if ',' not in cal_text and '_' not in cal_text:
                            if '  '  in cal_text:
                                    cal_text = cal_text.replace('  ', '.') 
                            elif ' ' in cal_text:
                                    cal_text = cal_text.replace(' ', '.') 
                        temp = []
                        for word in cal_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            cal_text = ''.join(temp)
                            cal_text = correct_text_errors(cal_text,value_handling_error2)
                        if cal_text.endswith('.'):
                            cal_text += '0'
                    except ValueError:
                        cal_text = placeholder
                else:
                    cal_text = placeholder
                PH_at_KCL_results_cor = extract_info(stdout_str, 'PH at KCL results')
                if PH_at_KCL_results_cor:
                    print("ph at kcl")
                    try:
                        x1,y1,x2,y2 = PH_at_KCL_results_cor
                        print(PH_at_KCL_results_cor)
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                        # Extract the region of interest (ROI) from the frame
                        ph_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        PH_at_KCL_results = reader.readtext(ph_roi)
                        for t_, t in enumerate(PH_at_KCL_results):
                            bbox, text, score = t
                            t = bbox, PH_at_KCL_results, score
                            print(t)
                        PH_at_KCL_results.sort(key=lambda x: x[0][0][0])
                        ph_text = ' '.join([text for bbox, text, score in PH_at_KCL_results])
                        if ',' not in ph_text and '_' not in ph_text:
                            if '  '  in ph_text:
                                    ph_text = ph_text.replace('  ', '.') 
                            elif ' ' in ph_text:
                                    ph_text = ph_text.replace(' ', '.') 
                        temp = []
                        for word in ph_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            ph_text = ''.join(temp)
                            ph_text = correct_text_errors(ph_text,value_handling_error2)
                        if ph_text.endswith('.'):
                            ph_text += '0'
                    except ValueError:
                        ph_text = placeholder
                else:
                    ph_text = placeholder
                HUMUS_percent_result_cor = extract_info(stdout_str, 'HUMUS % result')
                if HUMUS_percent_result_cor:
                    print("humus percent")
                    try:
                        x1,y1,x2,y2 = HUMUS_percent_result_cor

                        print(HUMUS_percent_result_cor)
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                        # Extract the region of interest (ROI) from the frame
                        hum_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        HUMUS_result = reader.readtext(hum_roi)
                        for t_, t in enumerate(HUMUS_result):
                            bbox, text, score = t
                            t = bbox, HUMUS_result, score
                            print(t)
                        HUMUS_result.sort(key=lambda x: x[0][0][0])
                        humus_text = ' '.join([text for bbox, text, score in HUMUS_result])
                        if ',' not in humus_text and '_' not in humus_text:
                            if '  '  in humus_text:
                                    humus_text = humus_text.replace('  ', '.') 
                            elif ' ' in humus_text:
                                    humus_text = humus_text.replace(' ', '.') 
                        temp = []
                        for word in humus_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            humus_text = ''.join(temp)
                            humus_text = correct_text_errors(humus_text,value_handling_error2)
                        if humus_text.endswith('.'):
                            humus_text += '0'
                    except ValueError:
                        humus_text = placeholder
                else:
                    humus_text = placeholder
                Iron_result_cor = extract_info(stdout_str, 'Iron result')
                if Iron_result_cor:
                    print("iron")
                    try:
                        x1,y1,x2,y2 = Iron_result_cor

                        print(Iron_result_cor)
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                        # Extract the region of interest (ROI) from the frame
                        iron_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        Iron_result = reader.readtext(iron_roi)
                        for t_, t in enumerate(Iron_result):
                            bbox, text, score = t
                            t = bbox, Iron_result, score
                            print(t)
                        Iron_result.sort(key=lambda x: x[0][0][0])
                        iron_text = ' '.join([text for bbox, text, score in Iron_result])
                        if ',' not in iron_text and '_' not in iron_text:
                            if '  '  in iron_text:
                                    iron_text = iron_text.replace('  ', '.') 
                            elif ' ' in iron_text:
                                    iron_text = iron_text.replace(' ', '.') 
                        temp = []
                        for word in iron_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            iron_text = ''.join(temp)
                            iron_text = correct_text_errors(iron_text,value_handling_error2)
                        if iron_text.endswith('.'):
                            iron_text += '0'
                    except ValueError:
                        iron_text = placeholder
                else:
                    iron_text = placeholder     
                manganese_result_cor = extract_info(stdout_str, 'manganese result')
                if manganese_result_cor:
                    print("manganese result")
                    print(manganese_result_cor)
                    try:
                        x1,y1,x2,y2 = manganese_result_cor

                        
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                        # Extract the region of interest (ROI) from the frame
                        man_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        manganese_result = reader.readtext(man_roi)
                        for t_, t in enumerate(manganese_result):
                            bbox, text, score = t
                            t = bbox, manganese_result, score
                            print(t)
                        manganese_result.sort(key=lambda x: x[0][0][0])
                        man_text = ' '.join([text for bbox, text, score in manganese_result])
                        print(man_text)
                        if ',' not in man_text and '_' not in man_text:
                            if '  '  in man_text:
                                    man_text = man_text.replace('  ', '.') 
                            elif ' ' in man_text:
                                    man_text = man_text.replace(' ', '.') 
                        print(man_text)

                        temp = []
                        for word in man_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            man_text = ''.join(temp)
                            man_text = correct_text_errors(man_text,value_handling_error2)
                        if man_text.endswith('.'):
                            man_text += '0'
                    except ValueError:
                        man_text = placeholder
                else:
                    man_text = placeholder 
                sodium_result_cor = extract_info(stdout_str, 'sodium result')
                if sodium_result_cor:
                    print("sodium_result")
                    try:
                        x1,y1,x2,y2 = sodium_result_cor
                        print(sodium_result_cor)
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                        sod_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        sodium_result = reader.readtext(sod_roi)
                        for t_, t in enumerate(sodium_result):
                            bbox, text, score = t
                            t = bbox, sodium_result, score
                            print(t)
                        sodium_result.sort(key=lambda x: x[0][0][0])
                        sod_text = ' '.join([text for bbox, text, score in sodium_result])
                        if ',' not in sod_text and '_' not in sod_text:
                            if '  '  in sod_text:
                                    sod_text = sod_text.replace('  ', '.') 
                            elif ' ' in sod_text:
                                    sod_text = sod_text.replace(' ', '.') 
                        temp = []
                        for word in sod_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            sod_text = ''.join(temp)
                            sod_text = correct_text_errors(sod_text,value_handling_error2)
                        if sod_text.endswith('.'):
                            sod_text += '0'
                    except ValueError:
                        sod_text = placeholder
                else:
                    sod_text = placeholder
                
                nt_cor = extract_info(stdout_str, 'Nt result')
                if nt_cor:
                    print("Nt result:")
                    print(nt_cor)
                    try:
                        x1, y1, x2, y2 = nt_cor 
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                        nt_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        nt = reader.readtext(nt_roi)
                        for t_, t in enumerate(nt):
                            bbox, text, score = t
                            t = bbox, nt, score
                            print(t)
                        nt.sort(key=lambda x: x[0][0][0])
                        nt_text = ' '.join([text for bbox, text, score in nt])
                        if ',' not in nt_text and '_' not in nt_text:
                            if '  '  in nt_text:
                                    nt_text = nt_text.replace('  ', '.') 
                            elif ' ' in nt_text:
                                    nt_text = nt_text.replace(' ', '.') 
                        temp = []
                        for word in nt_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            nt_text = ''.join(temp)
                            nt_text = correct_text_errors(nt_text,value_handling_error2)
                        if nt_text.endswith('.'):
                            nt_text += '0'
                    except ValueError:
                        nt_text = placeholder
                else:
                    nt_text = placeholder 
                ph_acetate_cor = extract_info(stdout_str, 'ph acetate result')
                if ph_acetate_cor:
                    print("ph acetate result:")
                    print(ph_acetate_cor)
                    try:
                        x1, y1, x2, y2 = ph_acetate_cor 
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                        ph_acetate_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        ph_acetate = reader.readtext(ph_acetate_roi)
                        for t_, t in enumerate(ph_acetate):
                            bbox, text, score = t
                            t = bbox, ph_acetate, score
                            print(t)
                        ph_acetate_text = ' '.join([text for bbox, text, score in ph_acetate])
                        temp = []
                        ph_acetate.sort(key=lambda x: x[0][0][0])
                        ph_acetate_text = ' '.join([text for bbox, text, score in ph_acetate])
                        if ',' not in ph_acetate_text and '_' not in ph_acetate_text:
                            if '  '  in ph_acetate_text:
                                    ph_acetate_text = ph_acetate_text.replace('  ', '.') 
                            elif ' ' in ph_acetate_text:
                                    ph_acetate_text = ph_acetate_text.replace(' ', '.') 
                        temp = []
                        for word in ph_acetate_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            ph_acetate_text = ''.join(temp)
                            ph_acetate_text = correct_text_errors(ph_acetate_text,value_handling_error2)
                        if ph_acetate_text.endswith('.'):
                            ph_acetate_text += '0'
                    except ValueError:
                        ph_acetate_text = placeholder
                else:
                    ph_acetate_text = placeholder 
                taux_argile_cor = extract_info(stdout_str, 'Taux d argile result')
                if taux_argile_cor:
                    print("Taux d argile result:")
                    print(taux_argile_cor)
                    try:
                        x1, y1, x2, y2 = taux_argile_cor 
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                        taux_argile_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        taux_argile = reader.readtext(taux_argile_roi)
                        for t_, t in enumerate(taux_argile):
                            bbox, text, score = t
                            t = bbox, taux_argile, score
                            print(t)
                        taux_argile_text = ' '.join([text for bbox, text, score in taux_argile])
                        temp = []
                        taux_argile.sort(key=lambda x: x[0][0][0])
                        taux_argile_text = ' '.join([text for bbox, text, score in taux_argile])
                        if ',' not in taux_argile_text and '_' not in taux_argile_text:
                            if '  '  in taux_argile_text:
                                    taux_argile_text = taux_argile_text.replace('  ', '.') 
                            elif ' ' in taux_argile_text:
                                    taux_argile_text = taux_argile_text.replace(' ', '.') 
                        temp = []
                        for word in taux_argile_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            taux_argile_text = ''.join(temp)
                            taux_argile_text = correct_text_errors(taux_argile_text,value_handling_error2)
                        if taux_argile_text.endswith('.'):
                            taux_argile_text += '0'
                    except ValueError:
                        taux_argile_text = placeholder
                else:
                    taux_argile_text = placeholder 
                CEC_cor = extract_info(stdout_str, 'CEC (cmol/kg) result')
                if CEC_cor:
                    print("CEC (cmol/kg) result:")
                    print(CEC_cor)
                    try:
                        x1, y1, x2, y2 = CEC_cor 
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                        CEC_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        CEC = reader.readtext(CEC_roi)
                        for t_, t in enumerate(CEC):
                            bbox, text, score = t
                            t = bbox, CEC, score
                            print(t)
                        CEC_text = ' '.join([text for bbox, text, score in CEC])
                        temp = []
                        CEC.sort(key=lambda x: x[0][0][0])
                        CEC_text = ' '.join([text for bbox, text, score in CEC])
                        if ',' not in CEC_text and '_' not in CEC_text:
                            if '  '  in CEC_text:
                                    CEC_text = CEC_text.replace('  ', '.') 
                            elif ' ' in CEC_text:
                                    CEC_text = CEC_text.replace(' ', '.') 
                        temp = []
                        for word in CEC_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            CEC_text = ''.join(temp)
                            CEC_text = correct_text_errors(CEC_text,value_handling_error2)
                        if CEC_text.endswith('.'):
                            CEC_text += '0'
                    except ValueError:
                        CEC_text = placeholder
                else:
                    CEC_text = placeholder 
                reportCN_cor = extract_info(stdout_str, 'report C/N result')
                if reportCN_cor:
                    print("report C/N result:")
                    print(reportCN_cor)
                    try:
                        x1, y1, x2, y2 = reportCN_cor 
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                        reportCN_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        reportCN = reader.readtext(reportCN_roi)
                        for t_, t in enumerate(reportCN):
                            bbox, text, score = t
                            t = bbox, reportCN, score
                            print(t)
                        reportCN_text = ' '.join([text for bbox, text, score in reportCN])
                        temp = []
                        reportCN.sort(key=lambda x: x[0][0][0])
                        reportCN_text = ' '.join([text for bbox, text, score in reportCN])
                        if ',' not in reportCN_text and '_' not in reportCN_text:
                            if '  '  in reportCN_text:
                                    reportCN_text = reportCN_text.replace('  ', '.') 
                            elif ' ' in reportCN_text:
                                    reportCN_text = reportCN_text.replace(' ', '.') 
                        temp = []
                        for word in reportCN_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            reportCN_text = ''.join(temp)
                            reportCN_text = correct_text_errors(reportCN_text,value_handling_error2)
                        if reportCN_text.endswith('.'):
                            reportCN_text += '0'
                    except ValueError:
                        reportCN_text = placeholder
                else:
                    reportCN_text = placeholder 
                reportKMG_cor = extract_info(stdout_str, 'Repport K/Mg result')
                if reportKMG_cor:
                    print("Repport K/Mg result:")
                    print(reportKMG_cor)
                    try:
                        x1, y1, x2, y2 = reportKMG_cor 
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                        reportKMG_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        reportKMG = reader.readtext(reportKMG_roi)
                        for t_, t in enumerate(reportKMG):
                            bbox, text, score = t
                            t = bbox, reportKMG, score
                            print(t)
                        reportKMG_text = ' '.join([text for bbox, text, score in reportKMG])
                        temp = []
                        reportKMG.sort(key=lambda x: x[0][0][0])
                        reportKMG_text = ' '.join([text for bbox, text, score in reportKMG])
                        if ',' not in reportKMG_text and '_' not in reportKMG_text:
                            if '  '  in reportKMG_text:
                                    reportKMG_text = reportKMG_text.replace('  ', '.') 
                            elif ' ' in reportKMG_text:
                                    reportKMG_text = reportKMG_text.replace(' ', '.') 
                        temp = []
                        for word in reportKMG_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            reportKMG_text = ''.join(temp)
                            reportKMG_text = correct_text_errors(reportKMG_text,value_handling_error2)
                        if reportKMG_text.endswith('.'):
                            reportKMG_text += '0'
                    except ValueError:
                        reportKMG_text = placeholder
                else:
                    reportKMG_text = placeholder 
                reportCAMG_cor = extract_info(stdout_str, 'Repport Ca/Mg result')
                if reportCAMG_cor:
                    print("Repport Ca/Mg result:")
                    print(reportCAMG_cor)
                    try:
                        x1, y1, x2, y2 = reportCAMG_cor 
                        tl_corner = (int(x1), int(y1))
                        br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                        reportCAMG_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                        reportCAMG = reader.readtext(reportCAMG_roi)
                        for t_, t in enumerate(reportCAMG):
                            bbox, text, score = t
                            t = bbox, reportCAMG, score
                            print(t)
                        reportCAMG_text = ' '.join([text for bbox, text, score in reportCAMG])
                        temp = []
                        reportCAMG.sort(key=lambda x: x[0][0][0])
                        reportCAMG_text = ' '.join([text for bbox, text, score in reportCAMG])
                        if ',' not in reportCAMG_text and '_' not in reportCAMG_text:
                            if '  '  in reportCAMG_text:
                                    reportCAMG_text = reportCAMG_text.replace('  ', '.') 
                            elif ' ' in reportCAMG_text:
                                    reportCAMG_text = reportCAMG_text.replace(' ', '.') 
                        temp = []
                        for word in reportCAMG_text.split():
                            temp.append(value_handling_error.get(word, word))
                            #temp.append(num_date.get(word, word))
                            reportCAMG_text = ''.join(temp)
                            reportCAMG_text = correct_text_errors(reportCAMG_text,value_handling_error2)
                        if reportCAMG_text.endswith('.'):
                            reportCAMG_text += '0'
                    except ValueError:
                        reportCAMG_text = placeholder
                else:
                    reportCAMG_text = placeholder 
                
            carbone_organic_results_cor = extract_info(stdout_str, 'carbone organic result')
            if carbone_organic_results_cor:
                print("carbone organic result")
                try:
                    x1,y1,x2,y2 = carbone_organic_results_cor
                    print(carbone_organic_results_cor)
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    carbone_organic_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    carbone_organic_results = reader.readtext(carbone_organic_roi)
                    for t_, t in enumerate(carbone_organic_results):
                        bbox, text, score = t
                        t = bbox, carbone_organic_results, score
                        print(t)
                    carbone_organic_results.sort(key=lambda x: x[0][0][0])
                    carbone_organic_text = ' '.join([text for bbox, text, score in carbone_organic_results])
                    if ',' not in carbone_organic_text and '_' not in carbone_organic_text:
                        if '  '  in carbone_organic_text:
                                carbone_organic_text = carbone_organic_text.replace('  ', '.') 
                        elif ' ' in carbone_organic_text:
                                carbone_organic_text = carbone_organic_text.replace(' ', '.') 
                    temp = []
                    for word in carbone_organic_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        carbone_organic_text = ''.join(temp)
                        carbone_organic_text = correct_text_errors(carbone_organic_text,value_handling_error2)
                    if carbone_organic_text.endswith('.'):
                        carbone_organic_text += '0'
                except ValueError:
                    carbone_organic_text = placeholder
            else:
                carbone_organic_text = placeholder
        
            AZOTE_TOTAL_percent_result_cor = extract_info(stdout_str, 'AZOTE TOTAL % result')
            if AZOTE_TOTAL_percent_result_cor:
                print("azote total ")
                try:
                    x1,y1,x2,y2 = AZOTE_TOTAL_percent_result_cor

                    print(AZOTE_TOTAL_percent_result_cor)
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    azote_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    AZOTE_TOTAL_result = reader.readtext(azote_roi)
                    for t_, t in enumerate(AZOTE_TOTAL_result):
                        bbox, text, score = t
                        t = bbox, AZOTE_TOTAL_result, score
                        print(t)
                    AZOTE_TOTAL_result.sort(key=lambda x: x[0][0][0])
                    azote_text = ' '.join([text for bbox, text, score in AZOTE_TOTAL_result])
                    if ',' not in azote_text and '_' not in azote_text:
                        if '  '  in azote_text:
                                azote_text = azote_text.replace('  ', '.') 
                        elif ' ' in azote_text:
                                azote_text = azote_text.replace(' ', '.') 
                    temp = []
                    for word in azote_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        azote_text = ''.join(temp)
                        azote_text = correct_text_errors(azote_text,value_handling_error2)
                    if azote_text.endswith('.'):
                        azote_text += '0'
                #additional_info_results.append(('azote', text))
                except ValueError:
                    azote_text = placeholder
            else:
                azote_text = placeholder 
            car_azo_cor = extract_info(stdout_str, 'Carbone/Azote Result')
            if car_azo_cor:
                print("Carbone/Azote Result")
                try:
                    x1,y1,x2,y2 = car_azo_cor

                    print(car_azo_cor)
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    car_azo_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    car_azo = reader.readtext(car_azo_roi)
                    for t_, t in enumerate(car_azo):
                        bbox, text, score = t
                        t = bbox, car_azo, score
                        print(t)
                    car_azo.sort(key=lambda x: x[0][0][0])
                    car_azo_text = ' '.join([text for bbox, text, score in car_azo])
                    if ',' not in car_azo_text and '_' not in car_azo_text:
                        if '  '  in car_azo_text:
                                car_azo_text = car_azo_text.replace('  ', '.') 
                        elif ' ' in car_azo_text:
                                car_azo_text = car_azo_text.replace(' ', '.') 
                    temp = []
                    for word in car_azo_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        car_azo_text = ''.join(temp)
                        car_azo_text = correct_text_errors(car_azo_text,value_handling_error2)
                    if car_azo_text.endswith('.'):
                        car_azo_text += '0'
                except ValueError:
                    car_azo_text = placeholder
            else:
                car_azo_text = placeholder  
            chl_de_sod_cor = extract_info(stdout_str, 'chlorure de sodium result')
            if chl_de_sod_cor:
                print("chlorure de sodium result")
                try:
                    x1,y1,x2,y2 = chl_de_sod_cor

                    print(chl_de_sod_cor)
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    chl_de_sod_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    chl_de_sod = reader.readtext(chl_de_sod_roi)
                    for t_, t in enumerate(chl_de_sod):
                        bbox, text, score = t
                        t = bbox, chl_de_sod, score
                        print(t)
                    chl_de_sod.sort(key=lambda x: x[0][0][0])
                    chl_de_sod_text = ' '.join([text for bbox, text, score in chl_de_sod])
                    if ',' not in chl_de_sod_text and '_' not in chl_de_sod_text:
                        if '  '  in chl_de_sod_text:
                                chl_de_sod_text = chl_de_sod_text.replace('  ', '.') 
                        elif ' ' in chl_de_sod_text:
                                chl_de_sod_text = chl_de_sod_text.replace(' ', '.') 
                    temp = []
                    for word in chl_de_sod_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        chl_de_sod_text = ''.join(temp)
                        chl_de_sod_text = correct_text_errors(chl_de_sod_text,value_handling_error2)
                    if chl_de_sod_text.endswith('.'):
                        chl_de_sod_text += '0'
                except ValueError:
                    chl_de_sod_text = placeholder
            else:
                chl_de_sod_text = placeholder  
            conduct_cor = extract_info(stdout_str, 'conductivite result')
            if conduct_cor:
                print("conductivite result")
                try:
                    x1,y1,x2,y2 = conduct_cor

                    print(conduct_cor)
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    conduct_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    conduct = reader.readtext(conduct_roi)
                    for t_, t in enumerate(conduct):
                        bbox, text, score = t
                        t = bbox, conduct, score
                        print(t)
                    conduct.sort(key=lambda x: x[0][0][0])
                    conduct_text = ' '.join([text for bbox, text, score in conduct])
                    if ',' not in conduct_text and '_' not in conduct_text:
                        if '  '  in conduct_text:
                                conduct_text = conduct_text.replace('  ', '.') 
                        elif ' ' in conduct_text:
                                conduct_text = conduct_text.replace(' ', '.') 
                    temp = []
                    for word in conduct_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        conduct_text = ''.join(temp)
                        conduct_text = correct_text_errors(conduct_text,value_handling_error2)
                    if conduct_text.endswith('.'):
                        conduct_text += '0'
                except ValueError:
                    conduct_text = placeholder
            else:
                conduct_text = placeholder  
            cap_cat_cor = extract_info(stdout_str, 'Capacite d echange cationque result')
            if cap_cat_cor:
                print("Capacite d echange cationque result")
                try:
                    x1,y1,x2,y2 = cap_cat_cor

                    print(cap_cat_cor)
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                    # Extract the region of interest (ROI) from the frame
                    cap_cat_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    cap_cat = reader.readtext(cap_cat_roi)
                    for t_, t in enumerate(cap_cat):
                        bbox, text, score = t
                        t = bbox, cap_cat, score
                        print(t)
                    cap_cat.sort(key=lambda x: x[0][0][0])
                    cap_cat_text = ' '.join([text for bbox, text, score in cap_cat])
                    if ',' not in cap_cat_text and '_' not in cap_cat_text:
                        if '  '  in cap_cat_text:
                                cap_cat_text = cap_cat_text.replace('  ', '.') 
                        elif ' ' in cap_cat_text:
                                cap_cat_text = cap_cat_text.replace(' ', '.') 
                    temp = []
                    for word in cap_cat_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        cap_cat_text = ''.join(temp)
                        cap_cat_text = correct_text_errors(cap_cat_text,value_handling_error2)
                    if cap_cat_text.endswith('.'):
                        cap_cat_text += '0'
                except ValueError:
                    cap_cat_text = placeholder    
            else:
                cap_cat_text = placeholder  
            
        
        
            PH_cor = extract_info(stdout_str, 'PH result')
            if PH_cor:
                print("PH result:")
                print(PH_cor)
                try:
                    x1, y1, x2, y2 = PH_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    PH_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    PH_only = reader.readtext(PH_roi)
                    for t_, t in enumerate(PH_only):
                        bbox, text, score = t
                        t = bbox, PH_only, score
                        print(t)
                    PH_only_text = ' '.join([text for bbox, text, score in PH_only])
                    temp = []
                    PH_only.sort(key=lambda x: x[0][0][0])
                    PH_only_text = ' '.join([text for bbox, text, score in PH_only])
                    if ',' not in PH_only_text and '_' not in PH_only_text:
                        if '  '  in PH_only_text:
                                PH_only_text = PH_only_text.replace('  ', '.') 
                        elif ' ' in PH_only_text:
                                PH_only_text = PH_only_text.replace(' ', '.') 
                    temp = []
                    for word in PH_only_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        PH_only_text = ''.join(temp)
                        PH_only_text = correct_text_errors(PH_only_text,value_handling_error2)
                    if PH_only_text.endswith('.'):
                        PH_only_text += '0'
                except ValueError:
                    PH_only_text = placeholder
            else:
                PH_only_text = placeholder 
            hardness_total_cor = extract_info(stdout_str, 'hardness total result')
            if hardness_total_cor:
                print("hardness total result:")
                print(hardness_total_cor)
                try:
                    x1, y1, x2, y2 = hardness_total_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    hardness_total_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    hardness_total = reader.readtext(hardness_total_roi)
                    for t_, t in enumerate(hardness_total):
                        bbox, text, score = t
                        t = bbox, hardness_total, score
                        print(t)
                    hardness_total_text = ' '.join([text for bbox, text, score in hardness_total])
                    temp = []
                    hardness_total.sort(key=lambda x: x[0][0][0])
                    hardness_total_text = ' '.join([text for bbox, text, score in hardness_total])
                    if ',' not in hardness_total_text and '_' not in hardness_total_text:
                        if '  '  in hardness_total_text:
                                hardness_total_text = hardness_total_text.replace('  ', '.') 
                        elif ' ' in hardness_total_text:
                                hardness_total_text = hardness_total_text.replace(' ', '.') 
                    temp = []
                    for word in hardness_total_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        hardness_total_text = ''.join(temp)
                        hardness_total_text = correct_text_errors(hardness_total_text,value_handling_error2)
                    if hardness_total_text.endswith('.'):
                        hardness_total_text += '0'
                except ValueError:
                    hardness_total_text = placeholder
            else:
                hardness_total_text = placeholder 
            PO4_cor = extract_info(stdout_str, 'PO4 result')
            if PO4_cor:
                print("PO4 result:")
                print(PO4_cor)
                try:
                    x1, y1, x2, y2 = PO4_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    PO4_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    PO4 = reader.readtext(PO4_roi)
                    for t_, t in enumerate(PO4):
                        bbox, text, score = t
                        t = bbox, PO4, score
                        print(t)
                    PO4_text = ' '.join([text for bbox, text, score in PO4])
                    temp = []
                    PO4.sort(key=lambda x: x[0][0][0])
                    PO4_text = ' '.join([text for bbox, text, score in PO4])
                    if ',' not in PO4_text and '_' not in PO4_text:
                        if '  '  in PO4_text:
                                PO4_text = PO4_text.replace('  ', '.') 
                        elif ' ' in PO4_text:
                                PO4_text = PO4_text.replace(' ', '.') 
                    temp = []
                    for word in PO4_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        PO4_text = ''.join(temp)
                        PO4_text = correct_text_errors(PO4_text,value_handling_error2)
                    if PO4_text.endswith('.'):
                        PO4_text += '0'
                except ValueError:
                    PO4_text = placeholder
            else:
                PO4_text = placeholder 
            SO4_cor = extract_info(stdout_str, 'SO4 result')
            if SO4_cor:
                print("SO4 result:")
                print(SO4_cor)
                try:
                    x1, y1, x2, y2 = SO4_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    SO4_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    SO4 = reader.readtext(SO4_roi)
                    for t_, t in enumerate(SO4):
                        bbox, text, score = t
                        t = bbox, SO4, score
                        print(t)
                    SO4_text = ' '.join([text for bbox, text, score in SO4])
                    temp = []
                    SO4.sort(key=lambda x: x[0][0][0])
                    SO4_text = ' '.join([text for bbox, text, score in SO4])
                    if ',' not in SO4_text and '_' not in SO4_text:
                        if '  '  in SO4_text:
                                SO4_text = SO4_text.replace('  ', '.') 
                        elif ' ' in SO4_text:
                                SO4_text = SO4_text.replace(' ', '.') 
                    temp = []
                    for word in SO4_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        SO4_text = ''.join(temp)
                        SO4_text = correct_text_errors(SO4_text,value_handling_error2)
                    if SO4_text.endswith('.'):
                        SO4_text += '0'
                except ValueError:
                    SO4_text = placeholder
            else:
                SO4_text = placeholder 
            Cl_cor = extract_info(stdout_str, 'Cl result')
            if Cl_cor:
                print("Cl result:")
                print(Cl_cor)
                try:
                    x1, y1, x2, y2 = Cl_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    Cl_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    Cl = reader.readtext(Cl_roi)
                    for t_, t in enumerate(Cl):
                        bbox, text, score = t
                        t = bbox, Cl, score
                        print(t)
                    Cl_text = ' '.join([text for bbox, text, score in Cl])
                    temp = []
                    Cl.sort(key=lambda x: x[0][0][0])
                    Cl_text = ' '.join([text for bbox, text, score in Cl])
                    if ',' not in Cl_text and '_' not in Cl_text:
                        if '  '  in Cl_text:
                                Cl_text = Cl_text.replace('  ', '.') 
                        elif ' ' in Cl_text:
                                Cl_text = Cl_text.replace(' ', '.') 
                    temp = []
                    for word in Cl_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        Cl_text = ''.join(temp)
                        Cl_text = correct_text_errors(Cl_text,value_handling_error2)

                        if Cl_text.endswith('.'):
                            Cl_text += '0'
                    chlo_text = Cl_text
                except ValueError:
                    chlo_text = placeholder
            else:
                chlo_text = placeholder 
            sables_50_100_cor = extract_info(stdout_str, 'sables de 50 a 100 micro result')
            if sables_50_100_cor:
                print("sables de 50 a 100 micro result:")
                print(sables_50_100_cor)
                try:
                    x1, y1, x2, y2 = sables_50_100_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    sables_50_100_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    sables_50_100 = reader.readtext(sables_50_100_roi)
                    for t_, t in enumerate(sables_50_100):
                        bbox, text, score = t
                        t = bbox, sables_50_100, score
                        print(t)
                    sables_50_100_text = ' '.join([text for bbox, text, score in sables_50_100])
                    temp = []
                    sables_50_100.sort(key=lambda x: x[0][0][0])
                    sables_50_100_text = ' '.join([text for bbox, text, score in sables_50_100])
                    if ',' not in sables_50_100_text and '_' not in sables_50_100_text:
                        if '  '  in sables_50_100_text:
                                sables_50_100_text = sables_50_100_text.replace('  ', '.') 
                        elif ' ' in sables_50_100_text:
                                sables_50_100_text = sables_50_100_text.replace(' ', '.') 
                    temp = []
                    for word in sables_50_100_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        sables_50_100_text = ''.join(temp)
                        sables_50_100_text = correct_text_errors(sables_50_100_text,value_handling_error2)
                    if sables_50_100_text.endswith('.'):
                        sables_50_100_text += '0'
                except ValueError:
                    sables_50_100_text = placeholder
            else:
                sables_50_100_text = placeholder 
            sables_100_200_cor = extract_info(stdout_str, 'sables de 100 a 200 micro result')
            if sables_100_200_cor:
                print("sables de 100 a 200 micro result:")
                print(sables_100_200_cor)
                try:
                    x1, y1, x2, y2 = sables_100_200_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    sables_100_200_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    sables_100_200 = reader.readtext(sables_100_200_roi)
                    for t_, t in enumerate(sables_100_200):
                        bbox, text, score = t
                        t = bbox, sables_100_200, score
                        print(t)
                    sables_100_200_text = ' '.join([text for bbox, text, score in sables_100_200])
                    temp = []
                    sables_100_200.sort(key=lambda x: x[0][0][0])
                    sables_100_200_text = ' '.join([text for bbox, text, score in sables_100_200])
                    if ',' not in sables_100_200_text and '_' not in sables_100_200_text:
                        if '  '  in sables_100_200_text:
                                sables_100_200_text = sables_100_200_text.replace('  ', '.') 
                        elif ' ' in sables_100_200_text:
                                sables_100_200_text = sables_100_200_text.replace(' ', '.') 
                    temp = []
                    for word in sables_100_200_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        sables_100_200_text = ''.join(temp)
                        sables_100_200_text = correct_text_errors(sables_100_200_text,value_handling_error2)
                    if sables_100_200_text.endswith('.'):
                        sables_100_200_text += '0'
                except ValueError:
                    sables_100_200_text = placeholder
            else:
                sables_100_200_text = placeholder 
            sables_200_500_cor = extract_info(stdout_str, 'sables de 200 a 500 micro result')
            if sables_200_500_cor:
                print("sables de 200 a 500 micro result:")
                print(sables_200_500_cor)
                try:
                    x1, y1, x2, y2 = sables_200_500_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    sables_200_500_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    sables_200_500 = reader.readtext(sables_200_500_roi)
                    for t_, t in enumerate(sables_200_500):
                        bbox, text, score = t
                        t = bbox, sables_200_500, score
                        print(t)
                    sables_200_500_text = ' '.join([text for bbox, text, score in sables_200_500])
                    temp = []
                    sables_200_500.sort(key=lambda x: x[0][0][0])
                    sables_200_500_text = ' '.join([text for bbox, text, score in sables_200_500])
                    if ',' not in sables_200_500_text and '_' not in sables_200_500_text:
                        if '  '  in sables_200_500_text:
                                sables_200_500_text = sables_200_500_text.replace('  ', '.') 
                        elif ' ' in sables_200_500_text:
                                sables_200_500_text = sables_200_500_text.replace(' ', '.') 
                    temp = []
                    for word in sables_200_500_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        sables_200_500_text = ''.join(temp)
                        sables_200_500_text = correct_text_errors(sables_200_500_text,value_handling_error2)
                    if sables_200_500_text.endswith('.'):
                        sables_200_500_text += '0'
                except ValueError:
                    sables_200_500_text = placeholder
            else:
                sables_200_500_text = placeholder 
            sables_500_1000_cor = extract_info(stdout_str, 'sables de 500 a 1000 micro result')
            if sables_500_1000_cor:
                print("sables de 500 a 1000 micro resultt:")
                print(sables_500_1000_cor)
                try:
                    x1, y1, x2, y2 = sables_500_1000_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    sables_500_1000_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    sables_500_1000 = reader.readtext(sables_500_1000_roi)
                    for t_, t in enumerate(sables_500_1000):
                        bbox, text, score = t
                        t = bbox, sables_500_1000, score
                        print(t)
                    sables_500_1000_text = ' '.join([text for bbox, text, score in sables_500_1000])
                    temp = []
                    sables_500_1000.sort(key=lambda x: x[0][0][0])
                    sables_500_1000_text = ' '.join([text for bbox, text, score in sables_500_1000])
                    if ',' not in sables_500_1000_text and '_' not in sables_500_1000_text:
                        if '  '  in sables_500_1000_text:
                                sables_500_1000_text = sables_500_1000_text.replace('  ', '.') 
                        elif ' ' in sables_500_1000_text:
                                sables_500_1000_text = sables_500_1000_text.replace(' ', '.') 
                    temp = []
                    for word in sables_500_1000_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        sables_500_1000_text = ''.join(temp)
                        sables_500_1000_text = correct_text_errors(sables_500_1000_text,value_handling_error2)
                    if sables_500_1000_text.endswith('.'):
                        sables_500_1000_text += '0'
                except ValueError:
                    sables_500_1000_text = placeholder
            else:
                sables_500_1000_text = placeholder 
            sables_1000_2000_cor = extract_info(stdout_str, 'sables de 1000 a 2000 micro result')
            if sables_1000_2000_cor:
                print("sables de 1000 a 2000 micro result:")
                print(sables_1000_2000_cor)
                try:
                    x1, y1, x2, y2 = sables_1000_2000_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    sables_1000_2000_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    sables_1000_2000 = reader.readtext(sables_1000_2000_roi)
                    for t_, t in enumerate(sables_1000_2000):
                        bbox, text, score = t
                        t = bbox, sables_1000_2000, score
                        print(t)
                    sables_1000_2000_text = ' '.join([text for bbox, text, score in sables_1000_2000])
                    temp = []
                    sables_1000_2000.sort(key=lambda x: x[0][0][0])
                    sables_1000_2000_text = ' '.join([text for bbox, text, score in sables_1000_2000])
                    if ',' not in sables_1000_2000_text and '_' not in sables_1000_2000_text:
                        if '  '  in sables_1000_2000_text:
                                sables_1000_2000_text = sables_1000_2000_text.replace('  ', '.') 
                        elif ' ' in sables_1000_2000_text:
                                sables_1000_2000_text = sables_1000_2000_text.replace(' ', '.') 
                    temp = []
                    for word in sables_1000_2000_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        sables_1000_2000_text = ''.join(temp)
                        sables_1000_2000_text = correct_text_errors(sables_1000_2000_text,value_handling_error2)
                    if sables_1000_2000_text.endswith('.'):
                        sables_1000_2000_text += '0'
                except ValueError:
                    sables_1000_2000_text = placeholder
            else:
                sables_1000_2000_text = placeholder 
            sables_larger_2000_cor = extract_info(stdout_str, 'sables larger than 2000 micro result')
            if sables_larger_2000_cor:
                print("sables larger than 2000 micro result:")
                print(sables_larger_2000_cor)
                try:
                    x1, y1, x2, y2 = sables_larger_2000_cor 
                    tl_corner = (int(x1), int(y1))
                    br_corner = (int(x2), int(y2))
                # Extract the region of interest (ROI) from the frame
                    sables_larger_2000_roi = frame[tl_corner[1]:br_corner[1], tl_corner[0]:br_corner[0]]
                    sables_larger_2000 = reader.readtext(sables_larger_2000_roi)
                    for t_, t in enumerate(sables_larger_2000):
                        bbox, text, score = t
                        t = bbox, sables_larger_2000, score
                        print(t)
                    sables_larger_2000_text = ' '.join([text for bbox, text, score in sables_larger_2000])
                    temp = []
                    sables_larger_2000.sort(key=lambda x: x[0][0][0])
                    sables_larger_2000_text = ' '.join([text for bbox, text, score in sables_larger_2000])
                    if ',' not in sables_larger_2000_text and '_' not in sables_larger_2000_text:
                        if '  '  in sables_larger_2000_text:
                                sables_larger_2000_text = sables_larger_2000_text.replace('  ', '.') 
                        elif ' ' in sables_larger_2000_text:
                                sables_larger_2000_text = sables_larger_2000_text.replace(' ', '.') 
                    temp = []
                    for word in sables_larger_2000_text.split():
                        temp.append(value_handling_error.get(word, word))
                        #temp.append(num_date.get(word, word))
                        sables_larger_2000_text = ''.join(temp)
                        sables_larger_2000_text = correct_text_errors(sables_larger_2000_text,value_handling_error2)
                    if sables_larger_2000_text.endswith('.'):
                        sables_larger_2000_text += '0'
                except ValueError:
                    sables_1000_2000_text = placeholder
            else:
                sables_larger_2000_text = placeholder 
        y=y+1
        if json_flag == False:

            image_data = {
            "name": NOP_text,
            "sample_marked": sample_marked_text,
            "analysis_date": date_text,
            "analysis_number": analysis_number_text,
            "page_number": str(page),
            "substances": [
                {"substance_name": "phosphorus", "value": phos_text, "depth": depth_text},#
                {"substance_name": "Potassium", "value": pota_text, "depth": depth_text},#
                {"substance_name": "Magnesium", "value": mag_text, "depth": depth_text},#
                {"substance_name": "Sodium", "value": sod_text, "depth": depth_text},#
                {"substance_name": "Aluminium", "value": Aluminium_text, "depth": depth_text},#
                {"substance_name": "Calcium", "value": cal_text, "depth": depth_text},#
                {"substance_name": "PH at KCL", "value": ph_text, "depth": depth_text},#
                {"substance_name": "carbone organic result", "value": carbone_organic_text, "depth": depth_text},#
                {"substance_name": "Humus", "value": humus_text, "depth": depth_text},#
                {"substance_name": "Manganese", "value": man_text, "depth": depth_text},#
                {"substance_name": "Iron", "value": iron_text, "depth": depth_text},#
                {"substance_name": "Nitrogen", "value": nitro_text, "depth": depth_text},
                {"substance_name": "boron", "value": boron_text, "depth": depth_text},#
                {"substance_name": "sulfer", "value": sulfer_text, "depth": depth_text},#
                {"substance_name": "Carbone", "value": carb_text, "depth": depth_text},
                {"substance_name": "Azote", "value": azote_text, "depth": depth_text},#
                {"substance_name": "Carbone/Azote", "value": car_azo_text, "depth": depth_text},#
                {"substance_name": "chlorure de sodium", "value": chl_de_sod_text, "depth": depth_text},#
                {"substance_name": "conductivite", "value": conduct_text, "depth": depth_text},#
                {"substance_name": "Capacite d echange cationque", "value": cap_cat_text, "depth": depth_text},#
                #{"substance_name": "zone", "value": zone_text, "depth": depth_text},#
                {"substance_name": "Nt", "value": nt_text, "depth": depth_text},#
                {"substance_name": "ph acetate", "value": ph_acetate_text, "depth": depth_text},#
                {"substance_name": "Taux d argile", "value": taux_argile_text, "depth": depth_text},#
                {"substance_name": "CEC (cmol/kg)", "value": CEC_text, "depth": depth_text},#
                {"substance_name": "report C/N", "value": reportCN_text, "depth": depth_text},#
                {"substance_name": "Repport K/Mg", "value": reportKMG_text, "depth": depth_text},#
                {"substance_name": "Repport Ca/Mg", "value": reportCAMG_text, "depth": depth_text},#
                #{"substance_name": "Repport Ca/Mg", "value": reportCAMG_text, "depth": depth_text},#
                {"substance_name": "PH result", "value": PH_only_text, "depth": depth_text},#
                {"substance_name": "hardness total", "value": hardness_total_text, "depth": depth_text},#
                {"substance_name": "PO4", "value": PO4_text, "depth": depth_text},#
                {"substance_name": "SO4", "value": SO4_text, "depth": depth_text},#
                {"substance_name": "Chlorine", "value": chlo_text, "depth": depth_text},#
                {"substance_name": "sables de 50 a 100 micro", "value": sables_50_100_text, "depth": depth_text},#
                {"substance_name": "sables de 100 a 200 micro", "value": sables_100_200_text, "depth": depth_text},#
                {"substance_name": "sables de 200 a 500 micro", "value": sables_200_500_text, "depth": depth_text},#
                {"substance_name": "sables de 500 a 1000 micro", "value": sables_500_1000_text, "depth": depth_text},#
                {"substance_name": "sables de 1000 a 2000 micro", "value": sables_1000_2000_text, "depth": depth_text},#
                {"substance_name": "sables larger than 2000 micro", "value": sables_larger_2000_text, "depth": depth_text},#
                {"substance_name": "Salt concentration", "value": salt_text, "depth": depth_text},
                {"substance_name": "Phossoul", "value": phossoul_text, "depth": depth_text},
                {"substance_name": "Copper", "value": copp_text, "depth": depth_text},#
                {"substance_name": "Zinc", "value": zinc_text, "depth": depth_text}#
            ]
        }
            #cpu_usage = psutil.cpu_percent(interval=1)

            # # Get GPU usage
           # gpus = GPUtil.getGPUs()
            #total_gpu_usage = sum(gpu.load for gpu in gpus) / len(gpus) if gpus else 0

            #print(f"CPU Usage: {cpu_usage}%")
            #print(f"Average GPU Usage: {total_gpu_usage * 100}%")
            
        # print(image_data)
            api_data['data'].append(image_data)
        # print(api_data)
        #answers = []
            output_file = 'soil_json_file/soil_analysis_api_data_2.0.json'
            with open(output_file, 'w') as f:
                json.dump(api_data, f, indent=4)

            print(f"API input data written to {output_file}")
    else:
        #cpu_usage = psutil.cpu_percent(interval=1)

        # # Get GPU usage
       # gpus = GPUtil.getGPUs()
       # total_gpu_usage = sum(gpu.load for gpu in gpus) / len(gpus) if gpus else 0

       # print(f"CPU Usage: {cpu_usage}%")
        #print(f"Average GPU Usage: {total_gpu_usage * 100}%")
        print("no data here, move to the next page")
        
    