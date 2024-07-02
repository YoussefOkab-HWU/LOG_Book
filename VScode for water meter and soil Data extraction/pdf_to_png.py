import os
import fitz
from PIL import Image
import shutil

def pdf_to_png(pdf_path, output_path):
    """
    Convert a PDF file to PNG images.

    Parameters:
    pdf_path (str): Path to the PDF file.
    output_path (str): Path to save the PNG images.
    """
    # Delete existing files in the output directory
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page in the PDF
    for page_number in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_number)

        # Render the page as a PNG image
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, clip=None, dpi=300)

        # Convert the pixmap to a PIL image
        pil_image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

        # Save the PIL image as a PNG file
        image_path = os.path.join(output_path, f"page_{page_number+1}.png")
        pil_image.save(image_path)
        print(f"Page {page_number+1} converted to PNG: {image_path}")
    # Delete all files in the PDF path
       
    # Close the PDF file
    pdf_document.close()
    try: 
        if os.path.isfile(pdf_path):
            os.unlink(pdf_path)
    except Exception as e:
        print(e)




# Example usage
#pdf_file = "/home/youssefokab/catkin_ws/src/yolov7/pdf_file/20240305062122288 (1).pdf"   
# pdf_file ='/home/youssefokab/catkin_ws/src/yolov7/pdf_file/20240305061733629 (1).pdf'
# output_directory = "/home/youssefokab/catkin_ws/src/yolov7/pdf_images"
# pdf_to_png(pdf_file, output_directory)