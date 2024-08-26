import fitz
import re
import pytesseract
from PIL import Image
import io
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

class Redactor:
    # Class variables for regex patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b(\+?\d{1,4}[\s-]?)?(?!0+\b)\d{10}\b'
    # NAME_PATTERN = r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b'
    NAME_PATTERN = r'Netaji'

    def __init__(self, path):
        self.path = path
        self.texts = ""
        self.t_blocks = None
        self.print_area = None
        self.use_ocr = True
        self.sensitive_datas = []
    
    @staticmethod
    def get_sensitive_data(text):
        """Function to get all sensitive data"""
        sensitive_data = []
        patterns = [Redactor.EMAIL_PATTERN, Redactor.PHONE_PATTERN, Redactor.NAME_PATTERN]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            sensitive_data.extend([match.group() for match in matches])
        
        return sensitive_data

    def get_text(self, page):
        text = page.get_text()
        # If no text is found, perform OCR
        if not text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = self.perform_ocr(img)

        return text

    def perform_ocr(self, image):
        """Perform OCR on the given image"""
        return ' '.join(pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)['text'])

    def find_text_locations(self, page, sensitive_data):
        """Find the locations of sensitive data on the page."""
        locations = []
        
        # Iterate over each piece of sensitive data
        for data in sensitive_data:
            # Using a custom search method to find the exact location of the text
            text_instances = page.search_for(data)
            
            # If the search finds text instances, add them to locations
            if not text_instances:
                locations.extend(text_instances)
            else:
                # Handle cases where the sensitive data is not found by the built-in search
                # Iterate through all text blocks on the page to find potential matches
                text_blocks = page.get_text("blocks")
                # self.t_blocks = self.get_text(page)
                self.t_blocks = text_blocks
                for block in text_blocks:
                    block_text = block[4]
                    if data in block_text:
                        # Approximate the location by matching the text within the block
                        block_rect = fitz.Rect(block[:4])
                        locations.append(block_rect)
        
        return locations

    def redact(self):
        """Main redactor code"""
        # Opening the PDF
        doc = fitz.open(self.path)
        
        # Iterating through pages
        for page_num  in range(len(doc)):
            page = doc.load_page(page_num)
            # _wrapContents is needed for fixing alignment issues with rect boxes
            page.wrap_contents()
            # # Get the text content of the page (this will work for text-based PDFs)
            text = page.get_text()
            
            # If no text is found, perform OCR
            if not text.strip():
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = self.perform_ocr(img)
            
            self.texts += text
            # Getting the sensitive data
            sensitive_data = self.get_sensitive_data(text)
            self.sensitive_datas.extend(sensitive_data)
            # First, try the built-in search function
            # areas = [area for data in sensitive_data for area in page.search_for(data)]
            areas = self.find_text_locations(page, sensitive_data)

            # If built-in search doesn't find all sensitive data, use our custom method
            # if len(areas) < len(sensitive_data):
            #     areas = self.find_text_locations(page, sensitive_data)
            self.print_area = areas
            # Drawing redaction annotations
            for area in areas:
                page.add_redact_annot(area, fill=(0, 0, 0))
            
            # Applying the redactions
            page.apply_redactions()
        
        # Constructing the output file path
        output_path = os.path.splitext(self.path)[0] + '_redacted.pdf'
        # Saving it to a new PDF
        doc.save(output_path)
        print(f"Successfully redacted and saved as {output_path}")

# Driver code for testing
if __name__ == "__main__":
    # Replace it with the name of the PDF file
    # path = r'tests\pdfs\Notice_Unauthorized_Freshers_Party.pdf'
    path = r'tests\pdfs\peeyush_resume.pdf'
    redactor = Redactor(path)
    redactor.redact()
    print(redactor.sensitive_datas)
    print(redactor.print_area)
    print(redactor.t_blocks)