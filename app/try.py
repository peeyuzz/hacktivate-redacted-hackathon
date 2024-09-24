# import fitz
# import re
# import pytesseract
# from PIL import Image
# import os
# import json
# import google.generativeai as genai
# # from google.generativeai import GenerativeModel
# from google.generativeai.types import HarmCategory, HarmBlockThreshold
# from dotenv import load_dotenv
# import whisper
# from pydub import AudioSegment
# import cv2
# from openvino.runtime import Core
# import numpy as np
# # import utils
# from faceDetection import FaceDetector
# import platform
# from scipy.ndimage import rotate
# from deskew import determine_skew

# load_dotenv()

# API_KEY = os.environ["API_KEY"]
# genai.configure(api_key=API_KEY)

# # Define tesseract command path based on the OS
# if platform.system() == "Windows":
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# elif platform.system() == "Linux":
#     # Typically Tesseract is installed in /usr/bin/tesseract on Linux
#     pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# class Redactor:
#     # Class variables for regex patterns
#     # EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
#     # PHONE_PATTERN = r'\b(?:\+?\d{1,4}[\s-]?)?(?!0+\b)\d{10}\b'
#     # AADHAR_PATTERN = r'[2-9]\d{3}\s?-?\d{4}\s?-?\d{4}'
#     # PASSPORT_PATTERN = r'\b[A-PR-WY][1-9]\d{7}\b'
#     # PAN_PATTERN = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
#     # BANK_ACC_PATTERN = r'\b\d{9,18}\b'

#     EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Z|a-z]{2,}\b'
#     PHONE_PATTERN = r'\b(?:\+?\d{1,4}[\s-]?)?(?:(?:\d{3}[\s-]?){2}\d{4}|\d{10})\b'
#     AADHAR_PATTERN = r'\b[2-9]\d{3}\s?-?\d{4}\s?-?\d{4}\b'
#     PASSPORT_PATTERN = r'\b[A-PR-WY][1-9]\d{7}\b'
#     PAN_PATTERN = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
#     BANK_ACC_PATTERN = r'\b\d{9,18}\b'
#     CREDIT_CARD_PATTERN = r'\b(?:4\d{3}|5[1-5]\d{2}|6011|65\d{2}|3[47]\d|30[012345]\d)\d{11}\b'
#     DRIVING_LICENSE_PATTERN = r'\b[A-Z]{2}\d{13}\b'
#     VEHICLE_REGISTRATION_PATTERN = r'\b[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}\b'

#     def __init__(self, path, plan_type="free", special_instructions=None):
#         model_path = os.path.join("model", "public", "ultra-lightweight-face-detection-rfb-320", "FP16", "ultra-lightweight-face-detection-rfb-320.xml")
#         self.path = path
#         self.texts = ""
#         self.special_instructions=special_instructions
#         self.plan_type = plan_type
#         self.sensitive_datas = []
#         self.face_detector = FaceDetector(model=model_path)

#     def get_text_without_ocr(self, page, get_blocks=False):
#         text = page.get_text()
#         if text.strip():
#             if get_blocks:
#                 return page.get_text("blocks")
#             else:
#                 return text
#         else:
#             return None
        
#     # def get_text_with_ocr(self, image, get_blocks=False):
#     #     B, G, R = cv2.split(image)
#     #     image = 0.299 * R + 0.587 * G + 0.114 * B
#     #     image = np.uint8(image)

#     #     if get_blocks:
#     #         ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#     #         blocks = []
#     #         for i in range(len(ocr_data['text'])):
#     #             if ocr_data['text'][i].strip():
#     #                 x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
#     #                 blocks.append([x, y, x+w, y+h, ocr_data['text'][i]])
#     #         return blocks
#     #     else:
#     #         return pytesseract.image_to_string(image)


#     def get_text_with_ocr(self, image, get_blocks=False):
#         # Convert to grayscale
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image

#         # Detect and correct skew
#         angle = determine_skew(gray)
#         rotated = rotate(gray, angle, reshape=False, mode='constant', cval=255, order=3)

#         # Binarization
#         # thresh = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#         kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#         sharpened = cv2.filter2D(rotated, -1, kernel)

#         # Perform OCR
#         if get_blocks:
#             ocr_data = pytesseract.image_to_data(sharpened, output_type=pytesseract.Output.DICT)
#             blocks = []
#             for i in range(len(ocr_data['text'])):
#                 if ocr_data['text'][i].strip():
#                     x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
#                     blocks.append([x, y, x+w, y+h, ocr_data['text'][i]])
#             return blocks
#         else:
#             return pytesseract.image_to_string(sharpened)

#     def get_sensitive_data(self, text):
        
#         patterns = [
#             re.compile(self.EMAIL_PATTERN, re.IGNORECASE),
#             re.compile(self.PHONE_PATTERN, re.IGNORECASE),
#             re.compile(self.AADHAR_PATTERN),
#             re.compile(self.PASSPORT_PATTERN),
#             re.compile(self.PAN_PATTERN),
#             re.compile(self.BANK_ACC_PATTERN),
#             re.compile(self.CREDIT_CARD_PATTERN),
#             re.compile(self.DRIVING_LICENSE_PATTERN),
#             re.compile(self.VEHICLE_REGISTRATION_PATTERN)
#         ]                               

#         if self.plan_type == 'pro':
#             with open('prompt.txt', 'r', encoding='utf-8') as file:
#                 prompt = file.read()
#             if self.special_instructions:
#                 prompt += f"""\n SPECIAL INSTRUCTIONS FROM THE USER, THESE HAVE THE HIGHEST PRIORITY: {self.special_instructions}"""
#                 # print(prompt)
#             model = genai.GenerativeModel("gemini-1.5-flash")
#             response = model.generate_content(
#                 f"<DOCUMENT>\n{text}\n</DOCUMENT>\n\n{prompt}",
#                 safety_settings={
#                     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#                     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#                     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#                     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#                 })
#             try:
#                 response_text = response.text
#                 match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
#                 if match:
#                     json_str = match.group(1).strip()
#                     try:
#                         sensitive_data = json.loads(json_str)
#                         return sensitive_data
#                     except json.JSONDecodeError as e:
#                         print(f"JSON decoding error: {e}")
#                 else:
#                     print("No JSON content found")

#             except json.JSONDecodeError as ve:
#                 print(f"Error here: {ve}")

#         else:
#             sensitive_data = []
#             for pattern in patterns:
#                 matches = pattern.finditer(text, re.IGNORECASE)
#                 matches = pattern.finditer(text)
#                 for match in matches:
#                     sensitive_data.append({"text": match.group(), "reason": "regex match", "level": "high"})
#             return sensitive_data
        
#         return []

        
    
#     def find_text_locations_without_ocr(self, page, sensitve_data):

#         locations = []
        
#         for data in sensitve_data:
#             text_instances = page.search_for(data['text'])
#             if text_instances:
#                 locations.extend(text_instances)
#             else:
#                 text_blocks = self.get_text_without_ocr(page, get_blocks=True)
#                 for block in text_blocks:
#                     block_text = block[4]
#                     if data['text'].lower() in block_text.lower():
#                         block_rect = fitz.Rect(block[:4])
#                         locations.append(block_rect)
        
#         return locations
    
#     # def find_text_locations_with_ocr(self, image, sensitive_data):

#     #     locations = []
#     #     text_blocks = self.get_text_with_ocr(image, get_blocks=True)
#     #     for data in sensitive_data:
#     #         data_componenets = data['text'].lower().split()
#     #         for block_idx in range(len(text_blocks)):
#     #             block = text_blocks[block_idx]
#     #             block_text = block[4]
#     #             if block_text.lower().replace(" ", "") in data_componenets:
#     #                 block_rect = fitz.Rect(block[:4])
#     #                 locations.append(block_rect)
#     #     return locations


#     def find_text_locations_with_ocr(self, image, sensitive_data):
#         locations = []
        
#         # Step 1: Get the text blocks from the image using OCR (similar to your existing code)
#         text_blocks = self.get_text_with_ocr(image, get_blocks=True)

#         # Step 2: Combine text from all blocks into a single sequence of words
#         words = [block[4] for block in text_blocks]  # Extracting words from blocks
#         paragraph_text = " ".join(words).lower()  # Combine into a single text
        
#         # Step 3: Create a list of sensitive data keywords
#         sensitive_keywords = [data['text'].lower() for data in sensitive_data]
        
#         # Step 4: Create a regex pattern to match any of the sensitive keywords
#         regex_pattern = re.compile(r"\b(" + "|".join(map(re.escape, sensitive_keywords)) + r")\b", re.IGNORECASE)

#         # Step 5: Match sensitive data in the combined paragraph text
#         matches = []
#         for match in regex_pattern.finditer(paragraph_text):
#             matched_text = match.group(0)
#             start_index = match.start()
#             end_index = match.end()

#             # Find which blocks this match came from
#             words_in_match = matched_text.split()
#             start_word_index = paragraph_text[:start_index].split().__len__()
#             end_word_index = start_word_index + len(words_in_match) - 1
            
#             # Step 6: Map the match back to the corresponding text blocks
#             matched_blocks = text_blocks[start_word_index:end_word_index + 1]
            
#             # Step 7: Get the bounding rectangle for each matched block
#             for block in matched_blocks:
#                 block_rect = fitz.Rect(block[:4])  # Assuming block[:4] is the coordinates
#                 locations.append(block_rect)
        
#         return locations

    
#     def get_face_location(self, image, use_fitz_format=False):
#         faces, _ = self.face_detector.inference(image)
#         if use_fitz_format:
#             locations = []
#             for face in faces:
#                 x0, y0, x1, y1 = face
#                 locations.append(fitz.Rect(x0, y0, x1, y1))
#                 return locations
#         else:
#             return faces
    
#     def preprocess_image(self, image):
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # Apply bilateral filter to remove noise while keeping edges sharp
#         gray = cv2.bilateralFilter(gray, 9, 75, 75)
#         # Use adaptive thresholding to improve contrast
#         thresh = cv2.adaptiveThreshold(
#             gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#         # Optionally, apply sharpening to enhance edges
#         kernel = np.array([[0, -1, 0],
#                         [-1, 5, -1],
#                         [0, -1, 0]])
#         sharpened = cv2.filter2D(thresh, -1, kernel)
#         return sharpened

#     def get_QR_code_location(self, image, use_fitz_format=False):
        
#         # preprocessing
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # A kernel that enhances the edges
#         kernel = np.array([[0, -1, 0],
#                         [-1, 5,-1],
#                         [0, -1, 0]])
#         # Apply the sharpening kernel to the image
#         sharpened = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)

#         qrCodeDetector = cv2.QRCodeDetector()
#         retval, decoded_info, points, straight_qrcode = qrCodeDetector.detectAndDecodeMulti(sharpened)
#         # print(points)
#         locations = []
#         if points is None:
#             return locations
#         for point in points:
#             x0, y0, x1, y1 = point[0][0], point[0][1], point[2][0], point[2][1]
#             if use_fitz_format:
#                 locations.append(fitz.Rect(x0, y0, x1, y1))
#             else:
#                 locations.append((x0, y0, x1, y1))
#         return locations

#     def redact_pdf(self):
#         doc = fitz.open(self.path)
#         for page in doc:
#             zoom = 2  # Increase resolution by factor of 2
#             mat = fitz.Matrix(zoom, zoom)
#             pix = page.get_pixmap(matrix=mat) 
#             image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             use_OCR = False
#             text = self.get_text_without_ocr(page)
#             if text is None:
#                 use_OCR = True
#                 text = self.get_text_with_ocr(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR))
#             print(text)
#             sensitive_data = self.get_sensitive_data(text)
#             print(sensitive_data)
#             locations = []
#             if not use_OCR:
#                 locations.extend([location for location in self.find_text_locations_without_ocr(page, sensitive_data)])
#             else:
#                 locations.extend([location/2 for location in self.find_text_locations_with_ocr(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR), sensitive_data)])
#             face_locations = self.get_face_location(np.array(image), use_fitz_format=True)
#             if face_locations:
#                 locations.extend([face_location/2 for face_location in face_locations]) 
#             qr_locations = self.get_QR_code_location(np.array(image), use_fitz_format=True)
#             if qr_locations:
#                 locations.extend([qr_location/2 for qr_location in qr_locations]) 
#             for location in locations:
#                 page.add_redact_annot(location, fill=(0, 0, 0))
#             page.apply_redactions()

#         output_path = os.path.splitext(self.path)[0] + '_redacted.pdf'
#         doc.save(output_path)
#         print(f"Successfully redacted and saved as {output_path}")
#         return locations


#     def redact_image(self):
#         image = cv2.imread(self.path)
#         # Get the original dimensions
#         height, width = image.shape[:2]
#         # Calculate new dimensions for 2x zoom
#         new_dimensions = (width * 2, height * 2)
#         # Resize the image
#         zoomed_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)
#         locations = []
#         # text = self.get_text_with_ocr(Image.fromarray(image))
#         text = self.get_text_with_ocr(zoomed_image)
#         print(text)
#         if text.strip():
#             sensitive_data = self.get_sensitive_data(text)
#             text_locations = self.find_text_locations_with_ocr(zoomed_image, sensitive_data)
#             if text_locations:
#                 locations.extend(text_locations)
#         face_locations = self.get_face_location(np.array(zoomed_image), use_fitz_format=False)
#         QR_locations = self.get_QR_code_location(np.array(zoomed_image), use_fitz_format=False)
#         if face_locations is not None:
#             locations.extend(face_locations) 
#         if QR_locations is not None:
#             locations.extend(QR_locations)
#         for location in locations:
#             x_min, y_min, x_max, y_max = np.array(location/2, dtype=np.int32)
#             cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)

#         output_path = os.path.splitext(self.path)[0] + '_redacted.jpg'
#         cv2.imwrite(output_path, image)
#         print(f"Successfully redacted image and saved as {output_path}")
#         return locations


import fitz
import re
import pytesseract
from PIL import Image
import os
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import cv2
import numpy as np
from faceDetection import FaceDetector
import platform
from scipy.ndimage import rotate
from deskew import determine_skew
import whisper
from pydub import AudioSegment

load_dotenv()

API_KEY = os.environ["API_KEY"]
genai.configure(api_key=API_KEY)

# Set tesseract command path based on OS
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
elif platform.system() == "Linux":
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

class Redactor:
    # Regex patterns (unchanged)
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b(?:\+?\d{1,4}[\s-]?)?(?:(?:\d{3}[\s-]?){2}\d{4}|\d{10})\b'
    AADHAR_PATTERN = r'\b[2-9]\d{3}\s?-?\d{4}\s?-?\d{4}\b'
    PASSPORT_PATTERN = r'\b[A-PR-WY][1-9]\d{7}\b'
    PAN_PATTERN = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
    BANK_ACC_PATTERN = r'\b\d{9,18}\b'
    CREDIT_CARD_PATTERN = r'\b(?:4\d{3}|5[1-5]\d{2}|6011|65\d{2}|3[47]\d|30[012345]\d)\d{11}\b'
    DRIVING_LICENSE_PATTERN = r'\b[A-Z]{2}\d{13}\b'
    VEHICLE_REGISTRATION_PATTERN = r'\b[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}\b'

    def __init__(self, path, plan_type="free", special_instructions=None):
        self.path = path
        self.special_instructions = special_instructions
        self.plan_type = plan_type
        model_path = os.path.join("model", "public", "ultra-lightweight-face-detection-rfb-320", "FP16", "ultra-lightweight-face-detection-rfb-320.xml")
        self.face_detector = FaceDetector(model=model_path)
        self.zoom_factor = 2  # Consolidated zoom factor

    def preprocess_image(self, image):
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect and correct skew
        angle = determine_skew(gray)
        rotated = rotate(gray, angle, reshape=False, mode='constant', cval=255, order=3)

        # Apply sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(rotated, -1, kernel)

        return sharpened

    def get_text_with_ocr(self, image):
        preprocessed = self.preprocess_image(image)
        ocr_data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
        
        full_text = " ".join(ocr_data['text'])
        blocks = []
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                blocks.append([x, y, x+w, y+h, ocr_data['text'][i]])
        
        return full_text, blocks

    def get_sensitive_data(self, text):
        if self.plan_type == 'pro':
            with open('prompt.txt', 'r', encoding='utf-8') as file:
                prompt = file.read()
            if self.special_instructions:
                prompt += f"\n SPECIAL INSTRUCTIONS FROM THE USER, THESE HAVE THE HIGHEST PRIORITY: {self.special_instructions}"
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"<DOCUMENT>\n{text}\n</DOCUMENT>\n\n{prompt}",
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                })
            
            try:
                response_text = response.text
                match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    return json.loads(json_str)
                else:
                    print("No JSON content found")
                    return []
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                return []
        else:
            patterns = [
                re.compile(pattern, re.IGNORECASE) for pattern in [
                    self.EMAIL_PATTERN, self.PHONE_PATTERN, self.AADHAR_PATTERN,
                    self.PASSPORT_PATTERN, self.PAN_PATTERN, self.BANK_ACC_PATTERN,
                    self.CREDIT_CARD_PATTERN, self.DRIVING_LICENSE_PATTERN,
                    self.VEHICLE_REGISTRATION_PATTERN
                ]
            ]
            
            sensitive_data = []
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    sensitive_data.append({"text": match.group(), "reason": "regex match", "level": "high"})
            return sensitive_data

    def find_text_locations(self, text_blocks, sensitive_data):
        locations = []
        full_text = " ".join([block[4].lower() for block in text_blocks])
        
        for data in sensitive_data:
            data_text = data['text'].lower()
            start = full_text.find(data_text)
            if start != -1:
                end = start + len(data_text)
                word_start = full_text[:start].count(' ')
                word_end = word_start + data_text.count(' ') + 1
                
                for block in text_blocks[word_start:word_end]:
                    x0, y0, x1, y1 = block[:4]
                #    locations.append(fitz.Rect(x0, y0, x1, y1))
                    locations.append(fitz.Rect(x0, y0, x1, y1))
        
        return locations

    def get_face_and_qr_locations(self, image):
        faces, _ = self.face_detector.inference(image)
        
        # face_locations = [fitz.Rect(face) for face in faces]
        face_locations = []
        for face in faces:
            x0, y0, x1, y1 = face
            face_locations.append(fitz.Rect(x0, y0, x1, y1))

        qr_detector = cv2.QRCodeDetector()
        _, _, points, _ = qr_detector.detectAndDecodeMulti(image)
        qr_locations = []
        if points is not None:
            for point in points:
                x0, y0, x1, y1 = point[0][0], point[0][1], point[2][0], point[2][1]
                qr_locations.append(fitz.Rect(x0, y0, x1, y1))

        return face_locations + qr_locations

    def redact_pdf(self):
        doc = fitz.open(self.path)
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom_factor, self.zoom_factor))
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            text, text_blocks = self.get_text_with_ocr(image_np)
            sensitive_data = self.get_sensitive_data(text)
            
            locations = self.find_text_locations(text_blocks, sensitive_data)
            locations.extend(self.get_face_and_qr_locations(image_np))

            for location in locations:
                page.add_redact_annot(location / self.zoom_factor, fill=(0, 0, 0))
            page.apply_redactions()

        output_path = os.path.splitext(self.path)[0] + '_redacted.pdf'
        doc.save(output_path)
        print(f"Successfully redacted and saved as {output_path}")
        return locations

    def redact_image(self):
        image = cv2.imread(self.path)
        height, width = image.shape[:2]
        zoomed_image = cv2.resize(image, (width * self.zoom_factor, height * self.zoom_factor), interpolation=cv2.INTER_LINEAR)

        text, text_blocks = self.get_text_with_ocr(zoomed_image)
        sensitive_data = self.get_sensitive_data(text)
        
        locations = self.find_text_locations(text_blocks, sensitive_data)
        locations.extend(self.get_face_and_qr_locations(zoomed_image))

        for location in locations:
            x_min, y_min, x_max, y_max = np.array(location / self.zoom_factor, dtype=np.int32)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)

        output_path = os.path.splitext(self.path)[0] + '_redacted.jpg'
        cv2.imwrite(output_path, image)
        print(f"Successfully redacted image and saved as {output_path}")
        return locations


    # def redact_video(self):
    #     video = cv2.VideoCapture(self.path)
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #     output_path = os.path.splitext(self.path)[0] + '_redacted.mp4'
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #     while True:
    #         ret, frame = video.read()
    #         if not ret:
    #             break
            
    #         locations = []

    #         # text = self.get_text_with_ocr(Image.fromarray(frame))
    #         # if text.strip():
    #         #     print(text)
    #         #     sensitive_data = self.get_sensitive_data(text)
    #         #     text_locations = self.find_text_locations_with_ocr(Image.fromarray(frame), sensitive_data)
    #         #     if text_locations:
    #         #         locations.extend(text_locations)

    #         face_locations = self.get_face_location(np.array(frame), use_fitz_format=False)
    #         if face_locations is not None:
    #             locations.extend(face_locations) 
        
    #         for location in locations:
    #             x_min, y_min, x_max, y_max = np.array(location, dtype=np.int32)
    #             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
                
    #         out.write(frame)

    #     video.release()
    #     out.release()
    #     print(f"Successfully redacted video and saved as {output_path}")
    
    # def transcribe_audio_with_whisper(self):
    #     model = whisper.load_model("tiny")
    #     transcript = model.transcribe(
    #         word_timestamps=True,
    #         audio=self.path
    #     )
    #     return transcript
    
    # def find_audio_locations(self, transcript, sensitive_data):
    #     locations = []
    #     for data in sensitive_data:
    #         for segment in transcript['segments']:
    #             if data['text'] in segment['text']:
    #                 print(data['text'])
    #                 print(segment['text'])
    #                 start_time = segment['start']
    #                 end_time = segment['end']
    #                 for word in segment['words']:
    #                     if word['word'].replace(" ", "")[:-2] not in data['text']:
    #                         print(f"{word['word']} not in data")
    #                         start_time = word['end']
    #                     else:
    #                         print(f"{word['word']} in data")
    #                         break
    #                 for word in reversed(segment['words']):
    #                     if word['word'].replace(" ", "")[:-2] not in data['text']:
    #                         print(f"{word['word']} not in data")
    #                         end_time = word['start']
    #                     else:
    #                         print(f"{word['word']} in data")
    #                         break
    #                 locations.append((start_time, end_time))
    #     return locations
    
    # def redact_audio(self):
    #     audio = AudioSegment.from_file(self.path)
    #     audio_length = len(audio)
    #     transcript = self.transcribe_audio_with_whisper()
    #     text = ' '.join([segment['text'] for segment in transcript['segments']])
    #     sensitive_data = self.get_sensitive_data(text)
    #     locations = self.find_audio_locations(transcript, sensitive_data)
    #     beep = AudioSegment.from_wav("beep.wav")
    #     locations.sort(key=lambda x: x[0])

    #     redacted_audio = AudioSegment.empty()
    #     last_end = 0

    #     for start_time, end_time in locations:
    #         start_ms = int(start_time * 1000)
    #         end_ms = int(end_time * 1000)
    #         redacted_audio += audio[last_end:start_ms]
    #         redacted_audio += beep[:end_ms - start_ms]
    #         last_end = end_ms
    #     redacted_audio += audio[last_end:]
        
    #     output_path = os.path.splitext(self.path)[0] + '_redacted.wav'
    #     redacted_audio.export(output_path, format="wav")
    #     print(f"Successfully redacted and saved as {output_path}")

    # def redact(self):
    #     file_extension = os.path.splitext(self.path)[1].lower()
    #     if file_extension == ".pdf":
    #         return self.redact_pdf()
    #     elif file_extension in [".mp3", ".wav"]:
    #         return self.redact_audio()
    #     elif file_extension in [".jpg", ".jpeg", ".png"]:
    #         return self.redact_image()
    #     elif file_extension in [".mp4", ".avi", ".mov"]:
    #         return self.redact_video()
    #     else:
    #         print(f"Unsupported file type: {file_extension}")



    def redact_video(self):
        video = cv2.VideoCapture(self.path)
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.splitext(self.path)[0] + '_redacted.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Zoom the frame
            zoomed_frame = cv2.resize(frame, (width * self.zoom_factor, height * self.zoom_factor), interpolation=cv2.INTER_LINEAR)
            
            # Get text and sensitive data (commented out for performance, uncomment if needed)
            # text, text_blocks = self.get_text_with_ocr(zoomed_frame)
            # sensitive_data = self.get_sensitive_data(text)
            # text_locations = self.find_text_locations(text_blocks, sensitive_data)
            
            # Get face locations
            face_locations = self.get_face_and_qr_locations(zoomed_frame)
            
            # Combine all locations
            # locations = text_locations + face_locations
            locations = face_locations  # If using only face detection
            
            # Apply redaction
            for location in locations:
                x_min, y_min, x_max, y_max = np.array(location / self.zoom_factor, dtype=np.int32)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
                
            out.write(frame)

        video.release()
        out.release()
        print(f"Successfully redacted video and saved as {output_path}")

    def transcribe_audio_with_whisper(self):
        model = whisper.load_model("tiny")
        transcript = model.transcribe(audio=self.path, word_timestamps=True)
        return transcript
    
    def find_audio_locations(self, transcript, sensitive_data):
        locations = []
        for data in sensitive_data:
            for segment in transcript['segments']:
                if data['text'].lower() in segment['text'].lower():
                    start_time = segment['start']
                    end_time = segment['end']
                    
                    # Fine-tune start and end times
                    for word in segment['words']:
                        if word['word'].lower().strip() in data['text'].lower():
                            start_time = word['start']
                            break
                    
                    for word in reversed(segment['words']):
                        if word['word'].lower().strip() in data['text'].lower():
                            end_time = word['end']
                            break
                    
                    locations.append((start_time, end_time))
        return locations
    
    def redact_audio(self):
        audio = AudioSegment.from_file(self.path)
        transcript = self.transcribe_audio_with_whisper()
        text = ' '.join([segment['text'] for segment in transcript['segments']])
        sensitive_data = self.get_sensitive_data(text)
        locations = self.find_audio_locations(transcript, sensitive_data)
        beep = AudioSegment.from_wav("beep.wav")
        
        redacted_audio = AudioSegment.empty()
        last_end = 0

        for start_time, end_time in sorted(locations):
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            redacted_audio += audio[last_end:start_ms]
            redacted_audio += beep[:end_ms - start_ms]
            last_end = end_ms
        redacted_audio += audio[last_end:]
        
        output_path = os.path.splitext(self.path)[0] + '_redacted.wav'
        redacted_audio.export(output_path, format="wav")
        print(f"Successfully redacted audio and saved as {output_path}")

    def redact(self):
        file_extension = os.path.splitext(self.path)[1].lower()
        redaction_methods = {
            ".pdf": self.redact_pdf,
            ".mp3": self.redact_audio,
            ".wav": self.redact_audio,
            ".jpg": self.redact_image,
            ".jpeg": self.redact_image,
            ".png": self.redact_image,
            ".mp4": self.redact_video,
            ".avi": self.redact_video,
            ".mov": self.redact_video
        }
        
        redact_method = redaction_methods.get(file_extension)
        if redact_method:
            return redact_method()
        else:
            print(f"Unsupported file type: {file_extension}")
            return None

path = r"C:\Users\admin\Documents\RMSI itnern\Intern Information Form Filled (1).pdf"
redactor = Redactor(path, plan_type="pro")
redactor.redact()