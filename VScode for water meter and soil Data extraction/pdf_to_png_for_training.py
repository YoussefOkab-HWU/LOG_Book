import os
import fitz
from PIL import Image

def pdf_to_png(pdf_path, output_path):
    """
    Convert a PDF file to PNG images.

    Parameters:
    pdf_path (str): Path to the PDF file.
    output_path (str): Path to save the PNG images.
    """
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
        image_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_number+1}.png")
        pil_image.save(image_path)
        print(f"Page {page_number+1} of {os.path.basename(pdf_path)} converted to PNG: {image_path}")
        
    # Close the PDF file
    pdf_document.close()

def convert_pdfs_in_folder(folder_path, output_directory):
    """
    Convert all PDF files in a folder to PNG images.

    Parameters:
    folder_path (str): Path to the folder containing PDF files.
    output_directory (str): Path to save the PNG images.
    """
    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            # Construct the full path of the PDF file
            pdf_file_path = os.path.join(folder_path, file_name)
            
            # Convert the PDF to PNG images
            pdf_to_png(pdf_file_path, output_directory)

# Example usage
pdf_folder = "/home/youssefokab/catkin_ws/src/yolov7/pdf_files_for_training"
output_directory = "/home/youssefokab/catkin_ws/src/yolov7/pdf_images_for_training"
convert_pdfs_in_folder(pdf_folder, output_directory)