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
# Define a function to sort coordinates based on y-coordinate
def sort_coordinates(coords):
    # Sort coordinates based on y-coordinate
    sorted_coords = sorted(zip(coords[::2], coords[1::2]), key=lambda x: x[1])
    # Flatten the sorted coordinates list
    flattened_sorted_coords = [val for sublist in sorted_coords for val in sublist]
    return flattened_sorted_coords
def coordinate_tackle(y,api_data, page, frame,extract_info,stdout_str,reader, num_sets,zone_cor,phosphore_result_cors,potassium_result_cors,magnesium_result_cors,calcium_result_cors,PH_at_KCL_results_cors,HUMUS_percent_result_cors,Iron_result_cors,manganese_result_cors,sodium_result_cors,nt_cors,ph_acetate_cors,taux_argile_cors,CEC_cors,reportCN_cors,reportKMG_cors,reportCAMG_cors):
    
    zone_cor = sort_coordinates(zone_cor)
    phosphore_result_cors = sort_coordinates(phosphore_result_cors)
    potassium_result_cors = sort_coordinates(potassium_result_cors)
    magnesium_result_cors = sort_coordinates(magnesium_result_cors)
    calcium_result_cors = sort_coordinates(calcium_result_cors)
    PH_at_KCL_results_cors = sort_coordinates(PH_at_KCL_results_cors)
    HUMUS_percent_result_cors = sort_coordinates(HUMUS_percent_result_cors)
    Iron_result_cors = sort_coordinates(Iron_result_cors)
    manganese_result_cors = sort_coordinates(manganese_result_cors)
    sodium_result_cors = sort_coordinates(sodium_result_cors)
    nt_cors = sort_coordinates(nt_cors)
    ph_acetate_cors = sort_coordinates(ph_acetate_cors)
    taux_argile_cors = sort_coordinates(taux_argile_cors)
    CEC_cors = sort_coordinates(CEC_cors)
    reportCN_cors = sort_coordinates(reportCN_cors)
    reportKMG_cors = sort_coordinates(reportKMG_cors)
    reportCAMG_cors = sort_coordinates(reportCAMG_cors)

    for i in range(num_sets):

        # Extract coordinates for each variable
        zone_cor = zone_cor[i*4 : (i+1)*4]
        phosphore_result_cor = phosphore_result_cors[i*4 : (i+1)*4]
        potassium_result_cor = potassium_result_cors[i*4 : (i+1)*4]
        magnesium_result_cor = magnesium_result_cors[i*4 : (i+1)*4]
        calcium_result_cor = calcium_result_cors[i*4 : (i+1)*4]
        PH_at_KCL_results_cor = PH_at_KCL_results_cors[i*4 : (i+1)*4]
        HUMUS_percent_result_cor = HUMUS_percent_result_cors[i*4 : (i+1)*4]
        Iron_result_cor = Iron_result_cors[i*4 : (i+1)*4]
        manganese_result_cor = manganese_result_cors[i*4 : (i+1)*4]
        sodium_result_cor = sodium_result_cors[i*4 : (i+1)*4]
        nt_cor = nt_cors[i*4 : (i+1)*4]
        ph_acetate_cor = ph_acetate_cors[i*4 : (i+1)*4]
        taux_argile_cor = taux_argile_cors[i*4 : (i+1)*4]
        CEC_cor = CEC_cors[i*4 : (i+1)*4]
        reportCN_cor = reportCN_cors[i*4 : (i+1)*4]
        reportKMG_cor = reportKMG_cors[i*4 : (i+1)*4]
        reportCAMG_cor = reportCAMG_cors[i*4 : (i+1)*4]
        for coordinates in [zone_cor, phosphore_result_cor, potassium_result_cor, magnesium_result_cor, calcium_result_cor, PH_at_KCL_results_cor, HUMUS_percent_result_cor, Iron_result_cor, manganese_result_cor, sodium_result_cor, nt_cor, ph_acetate_cor, taux_argile_cor, CEC_cor, reportCN_cor, reportKMG_cor, reportCAMG_cor]:
            analysis_number_text = "N/A"+ str(y)
            if zone_cor:
                print("zone result:")
                print(zone_cor)
                
                
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
            else:
                zone_text = "N/A"
                        
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
                phos_text = "N/A"
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
                        pota_text = "N/A"
            else:
                pota_text = "N/A"
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
                mag_text = "N/A"
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
                cal_text = "N/A"
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
                ph_text = "N/A"
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
                humus_text = "N/A"
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
                iron_text = "N/A"     
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
                    man_text = "N/A"
            else:
                man_text = "N/A" 
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
                sod_text = "N/A"
            
            if nt_cor:
                print("Nt result:")
                print(nt_cor)
            
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
            else:
                nt_text = "N/A" 
            if ph_acetate_cor:
                print("ph acetate result:")
                print(ph_acetate_cor)
            
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
            else:
                ph_acetate_text = "N/A" 
            if taux_argile_cor:
                print("Taux d argile result:")
                print(taux_argile_cor)
            
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
            else:
                taux_argile_text = "N/A" 
            if CEC_cor:
                print("CEC (cmol/kg) result:")
                print(CEC_cor)
            
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
            else:
                CEC_text = "N/A" 
            if reportCN_cor:
                print("report C/N result:")
                print(reportCN_cor)
            
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
            else:
                reportCN_text = "N/A" 
            if reportKMG_cor:
                print("Repport K/Mg result:")
                print(reportKMG_cor)
            
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
            else:
                reportKMG_text = "N/A" 
            if reportCAMG_cor:
                print("Repport Ca/Mg result:")
                print(reportCAMG_cor)
            
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
            else:
                reportCAMG_text = "N/A" 
        y = y+1
        image_data = {
        "name": zone_text,
        "analysis_date": "N/A",
        "analysis_number": analysis_number_text,
        "page_number": str(page),
        "substances": [
            {"substance_name": "phosphorus", "value": phos_text, "depth": "N/A"},#
            {"substance_name": "Potassium", "value": pota_text, "depth": "N/A"},#
            {"substance_name": "Magnesium", "value": mag_text, "depth": "N/A"},#
            {"substance_name": "Sodium", "value": sod_text, "depth": "N/A"},#
            {"substance_name": "Aluminium", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "Calcium", "value": cal_text, "depth": "N/A"},#
            {"substance_name": "PH at KCL", "value": ph_text, "depth": "N/A"},#
            {"substance_name": "carbone organic result", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "Humus", "value": humus_text, "depth": "N/A"},#
            {"substance_name": "Manganese", "value": man_text, "depth": "N/A"},#
            {"substance_name": "Iron", "value": iron_text, "depth": "N/A"},#
            {"substance_name": "Nitrogen", "value": "N/A", "depth": "N/A"},
            {"substance_name": "boron", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "sulfer", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "Carbone", "value": "N/A", "depth": "N/A"},
            {"substance_name": "Azote", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "Carbone/Azote", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "chlorure de sodium", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "conductivite", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "Capacite d echange cationque", "value": "N/A", "depth": "N/A"},#
            #{"substance_name": "zone", "value": zone_text, "depth": "N/A"},#
            {"substance_name": "Nt", "value": nt_text, "depth": "N/A"},#
            {"substance_name": "ph acetate", "value": ph_acetate_text, "depth": "N/A"},#
            {"substance_name": "Taux d argile", "value": taux_argile_text, "depth": "N/A"},#
            {"substance_name": "CEC (cmol/kg)", "value": CEC_text, "depth": "N/A"},#
            {"substance_name": "report C/N", "value": reportCN_text, "depth": "N/A"},#
            {"substance_name": "Repport K/Mg", "value": reportKMG_text, "depth": "N/A"},#
            {"substance_name": "Repport Ca/Mg", "value": reportCAMG_text, "depth": "N/A"},#
            #{"substance_name": "Repport Ca/Mg", "value": reportCAMG_text, "depth": "N/A"},#
            {"substance_name": "PH result", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "hardness total", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "PO4", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "SO4", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "Chlorine", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "sables de 50 a 100 micro", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "sables de 100 a 200 micro", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "sables de 200 a 500 micro", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "sables de 500 a 1000 micro", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "sables de 1000 a 2000 micro", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "sables larger than 2000 micro", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "Salt concentration", "value": "N/A", "depth": "N/A"},
            {"substance_name": "Phossoul", "value": "N/A", "depth": "N/A"},
            {"substance_name": "Copper", "value": "N/A", "depth": "N/A"},#
            {"substance_name": "Zinc", "value": "N/A", "depth": "N/A"}#
        ]
    }
# print(image_data)
        api_data['data'].append(image_data)
        #print(api_data)
    #answers = []
        output_file = 'soil_json_file/soil_analysis_api_data_2.0.json'
        with open(output_file, 'w') as f:
            json.dump(api_data, f, indent=4)

        print(f"API input data written to {output_file}")
        