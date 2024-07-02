import PyPDF2
import json

def pdf_to_json(pdf_path, json_path):
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    data = {'text': text}
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)

# Example usage:
pdf_to_json('/home/youssefokab/catkin_ws/src/yolov7/pdf_files_for_training/20221125082130386 - 2016 ok with depth 1.pdf', '/home/youssefokab/catkin_ws/src/yolov7/example.json')
