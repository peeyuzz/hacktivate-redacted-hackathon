import fitz
import re
import pytesseract
from PIL import Image
import os
import json
import google.generativeai as genai
# from google.generativeai import GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import whisper
from pydub import AudioSegment
import cv2
from openvino.runtime import Core
import numpy as np
from app import utils
import platform

load_dotenv()

API_KEY = os.environ["API_KEY"]
genai.configure(api_key=API_KEY)
# Define tesseract command path based on the OS
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
elif platform.system() == "Linux":
    # Typically Tesseract is installed in /usr/bin/tesseract on Linux
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
class FaceDetector:
    def __init__(self,
                 model,
                 confidence_thr=0.3,
                 overlap_thr=0.7):
        # load and compile the model
        core = Core()
        model = core.read_model(model=model)
        compiled_model = core.compile_model(model=model)
        self.model = compiled_model

        # 'Cause that model has more than one output,
        # We are saving the names in a more human frendly
        # variable to remember later how to recover the output we wish
        # In our case here, is a output for hte bbox and other for the score
        # /confidence. Have a look at the openvino documentation for more i
        self.output_scores_layer = self.model.output(0)
        self.output_boxes_layer = self.model.output(1)
        # confidence threshold
        self.confidence_thr = confidence_thr
        # threshold for the nonmaximum suppression
        self.overlap_thr = overlap_thr

    def preprocess(self, image):
        """
            input image is a numpy array image representation,
            in the BGR format of any shape.
        """
        # resize to match the expected by the model
        input_image = cv2.resize(image, dsize=[320, 240])
        # changing from [H, W, C] to [C, H, W]. "channels first"
        input_image = np.expand_dims(input_image.transpose(2, 0, 1), axis=0)
        return input_image

    def posprocess(self, pred_scores, pred_boxes, image_shape):
        # get all predictions with more than confidence_thr of confidence
        filtered_indexes = np.argwhere(
            pred_scores[0, :, 1] > self.confidence_thr).tolist()
        filtered_boxes = pred_boxes[0, filtered_indexes, :]
        filtered_scores = pred_scores[0, filtered_indexes, 1]

        if len(filtered_scores) == 0:
            return [], []

        # convert all boxes to image coordinates
        h, w = image_shape

        def _convert_bbox_format(*args):
            bbox = args[0]
            x_min, y_min, x_max, y_max = bbox
            x_min = int(w*x_min)
            y_min = int(h*y_min)
            x_max = int(w*x_max)
            y_max = int(h*y_max)
            return x_min, y_min, x_max, y_max

        bboxes_image_coord = np.apply_along_axis(
            _convert_bbox_format, axis=2, arr=filtered_boxes)

        # apply non-maximum supressions
        bboxes_image_coord, indexes = utils.non_max_suppression(bboxes_image_coord.reshape([-1, 4]),
                                                                overlapThresh=self.overlap_thr)
        filtered_scores = filtered_scores[indexes]
        return bboxes_image_coord, filtered_scores

    def draw_bboxes(self, image, bboxes, color=[0, 255, 0]):
        # Just for visualization
        # draw all bboxes on the input image
        for boxe in bboxes:
            x_min, y_min, x_max, y_max = boxe
            pt1 = (x_min, y_min)
            pt2 = (x_max, y_max)
            cv2.rectangle(image, pt1, pt2, color=color,
                          thickness=2, lineType=cv2.LINE_4)  # BGR

    def inference(self, image):
        input_image = self.preprocess(image)
        # inference
        pred_scores = self.model([input_image])[self.output_scores_layer]
        pred_boxes = self.model([input_image])[self.output_boxes_layer]

        image_shape = image.shape[:2]
        faces, scores = self.posprocess(pred_scores, pred_boxes, image_shape)
        return faces, scores


class Redactor:
    # Class variables for regex patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    PHONE_PATTERN = r'\b(?:\+?\d{1,4}[\s-]?)?(?!0+\b)\d{10}\b'
    AADHAR_PATTERN = r'[2-9]\d{3}\s?-?\d{4}\s?-?\d{4}'
    PASSPORT_PATTERN = r'\b[A-PR-WY][1-9]\d{7}\b'
    PAN_PATTERN = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
    BANK_ACC_PATTERN = r'\b\d{9,18}\b'
    def __init__(self, path, plan_type="free"):
        model_path = os.path.join("model", "public", "ultra-lightweight-face-detection-rfb-320", "FP16", "ultra-lightweight-face-detection-rfb-320.xml")
        self.path = path
        self.texts = ""
        self.plan_type = plan_type
        self.sensitive_datas = []
        self.face_detector = FaceDetector(model=model_path)

    
    def get_sensitive_data(self, text):
        """Function to get all sensitive data"""
        # sensitive_data = []

        patterns = [
            self.EMAIL_PATTERN,
            self.PHONE_PATTERN,
            self.AADHAR_PATTERN,
            self.PASSPORT_PATTERN,
            self.PAN_PATTERN,
            self.BANK_ACC_PATTERN
        ]

        if self.plan_type == "pro":
            with open('prompt.txt', 'r', encoding='utf-8') as file:
                prompt = file.read()
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
                # Assuming `response` is your API response object
                response_text = response.text
                match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    try:
                        sensitive_data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print("JSON decoding error:", e)
                else:
                    print("No JSON content found.")

                # sensitive_data.extend(json.loads(response.text.split('```')[1]))  # Assuming the response is a list of dicts

            except json.JSONDecodeError as ve:
                print(f"Error here: {ve}")
                # Handle the error, e.g., log it or notify the user
        else:
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                sensitive_data = [{"text": match.group(), "reason": "regex match", "level": "medium"} for match in matches]
        
        return sensitive_data

    def get_text(self, page, get_blocks=False):
        text = page.get_text()
        if not text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            if get_blocks:
                return self.perform_ocr(img, get_blocks=True)
            else:
                return self.perform_ocr(img)
        else:
            if get_blocks:
                return page.get_text("blocks")
            else:
                return text


    def perform_ocr(self, image, get_blocks=False):
        """Perform OCR on the given image"""
        if get_blocks:
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            blocks = []
            for i in range(len(ocr_data['text'])):
                if ocr_data['text'][i].strip():
                    x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                    blocks.append([x, y, x+w, y+h, ocr_data['text'][i]])
            return blocks
        else:
            return pytesseract.image_to_string(image)

    def find_text_locations(self, page, sensitive_data):
        """Find the locations of sensitive data on the page."""
        locations = []
        text_blocks = self.get_text(page, get_blocks=True)
        
        for data in sensitive_data:
            text_instances = page.search_for(data['text'])
            if text_instances:
                locations.extend(text_instances)
            else:
                if isinstance(text_blocks, list):  # OCR result
                    for block in text_blocks:
                        # print(block)
                        if block[4].lower().replace(" ", "  ") in data['text'].lower().split():
                            block_rect = fitz.Rect(block[:4])
                            locations.append(block_rect)
                else:  # Regular PDF text blocks
                    for block in text_blocks:
                        block_text = block[4]
                        if data['text'].lower() in block_text.lower():
                            block_rect = fitz.Rect(block[:4])
                            locations.append(block_rect)
        return locations

    def redact_pdf(self):
        """Main redactor code"""
        doc = fitz.open(self.path)
        for page in doc:
            page.wrap_contents()
            text = self.get_text(page)
            self.texts += text
            sensitive_data = self.get_sensitive_data(text)
            self.sensitive_datas.extend(sensitive_data)
            areas = self.find_text_locations(page, sensitive_data)
            for area in areas:
                page.add_redact_annot(area, fill=(0, 0, 0))
            page.apply_redactions()
        
        output_path = os.path.splitext(self.path)[0] + '_redacted.pdf'
        doc.save(output_path)
        print(f"Successfully redacted and saved as {output_path}")
        # return self.sensitive_datas
        return output_path

    def transcribe_audio_with_whisper(self):
        model = whisper.load_model("tiny")
        transcript = model.transcribe(
            word_timestamps=True,
            audio=self.path
        )
        return transcript
    
    def find_audio_locations(self, sensitive_data, transcript):
        locations = []
        for data in sensitive_data:
            for segment in transcript['segments']:
                # if data in segment['text']:
                #     for word in segment['words']:
                #         if word['word'].lower() in data['text'].lower():
                #             locations.append((word['start'], word['end']))
                
                if data['text'] in segment['text']:
                    print(data['text'])
                    print(segment['text'])
                    start_time = segment['start']
                    end_time = segment['end']
                    for word in segment['words']:
                        # print(word['word'])
                        if word['word'].replace(" ", "")[:-2] not in data['text']:
                            print(f"{word['word']} not in data")
                            start_time = word['end']
                        else:
                            print(f"{word['word']} in data")
                            break
                    for word in reversed(segment['words']):
                        if word['word'].replace(" ", "")[:-2] not in data['text']:
                            print(f"{word['word']} not in data")
                            end_time = word['start']
                        else:
                            print(f"{word['word']} in data")
                            break
                    locations.append((start_time, end_time))
        return locations
    
    def redact_audio(self):
        audio = AudioSegment.from_file(self.path)
        self.audio_length = len(audio)
        transcript = self.transcribe_audio_with_whisper()
        # print(transcript)
        self.texts = ' '.join([segment['text'] for segment in transcript['segments']])
        sensitive_data = self.get_sensitive_data(self.texts)
        self.sensitive_datas = sensitive_data
        areas = self.find_audio_locations(sensitive_data, transcript)
        beep = AudioSegment.from_wav("beep.wav")
        areas.sort(key=lambda x: x[0])
    
        redacted_audio = AudioSegment.empty()
        last_end = 0
        
        for start_time, end_time in areas:
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)            
            redacted_audio += audio[last_end:start_ms]
            redacted_audio += beep[:end_ms - start_ms]
            last_end = end_ms
        # Add any remaining audio after the last redacted area
        redacted_audio += audio[last_end:]

        output_path = os.path.splitext(self.path)[0] + '_redacted.wav'
        redacted_audio.export(output_path, format="wav")
        print(f"Successfully redacted and saved as {output_path}")
        return output_path

    def redact_image(self):
        image = cv2.imread(self.path)
        faces, _ = self.face_detector.inference(image)
        
        for face in faces:
            x_min, y_min, x_max, y_max = face
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
        
        output_path = os.path.splitext(self.path)[0] + '_redacted.jpg'
        cv2.imwrite(output_path, image)
        print(f"Successfully redacted image and saved as {output_path}")
        return output_path

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
            
            faces, _ = self.face_detector.inference(frame)
            
            for face in faces:
                x_min, y_min, x_max, y_max = face
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
            
            out.write(frame)
        
        video.release()
        out.release()
        print(f"Successfully redacted video and saved as {output_path}")

        # Redact audio in the video
        # self.redact_audio_in_video(output_path)
        return output_path

    def redact_audio_in_video(self, video_path):
        # Extract audio from video
        video = AudioSegment.from_file(video_path, format="mp4")
        audio = video.split_to_mono()[0]
        audio.export("temp_audio.wav", format="wav")

        # Redact the extracted audio
        self.path = "temp_audio.wav"
        self.redact_audio()

        # Combine redacted audio with redacted video
        redacted_audio = AudioSegment.from_wav(os.path.splitext(self.path)[0] + '_redacted.wav')
        video = video.overlay(redacted_audio)

        output_path = os.path.splitext(video_path)[0] + '_with_redacted_audio.mp4'
        video.export(output_path, format="mp4")

        # Clean up temporary files
        os.remove("temp_audio.wav")
        os.remove(os.path.splitext(self.path)[0] + '_redacted.wav')
        os.remove(video_path)

        print(f"Successfully redacted video with audio and saved as {output_path}")

    def redact(self):
        file_extension = os.path.splitext(self.path)[1].lower()
        if file_extension == ".pdf":
            return self.redact_pdf()
        elif file_extension in [".mp3", ".wav"]:
            return self.redact_audio()
        elif file_extension in [".jpg", ".jpeg", ".png"]:
            return self.redact_image()
        elif file_extension in [".mp4", ".avi", ".mov"]:
            return self.redact_video()
        else:
            print(f"Unsupported file type: {file_extension}")

# if __name__ == "__main__":
#     path = r'tests\pdfs\Notice_Unauthorized_Freshers_Party.pdf'
#     redactor = Redactor(path, plan_type="pro")
#     sensitive_data = redactor.redact()
#     # print(redactor.sensitive_datas)