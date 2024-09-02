import fitz
import re
import pytesseract
from PIL import Image
import os
import json
import google.generativeai as genai
from google.generativeai import GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import whisper
from pydub import AudioSegment

load_dotenv()

API_KEY = os.environ["API_KEY"]
genai.configure(api_key=API_KEY)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

class Redactor:
    # Class variables for regex patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    PHONE_PATTERN = r'\b(?:\+?\d{1,4}[\s-]?)?(?!0+\b)\d{10}\b'
    AADHAR_PATTERN = r'[2-9]\d{3}\s?-?\d{4}\s?-?\d{4}'
    PASSPORT_PATTERN = r'\b[A-PR-WY][1-9]\d{7}\b'  
    PAN_PATTERN = r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'
    BANK_ACC_PATTERN = r'\b\d{9,18}\b'

    def __init__(self, path, plan_type = "free"):
        self.path = path
        self.texts = ""
        self.plan_type = plan_type
        self.sensitive_datas = []
    
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

    def get_text(self, page):
        text = page.get_text()
        if not text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = self.perform_ocr(img)
        return text

    def perform_ocr(self, image):
        """Perform OCR on the given image"""
        return pytesseract.image_to_string(image)

    def find_text_locations(self, page, sensitive_data):
        """Find the locations of sensitive data on the page."""
        locations = []
        for data in sensitive_data:
            text_instances = page.search_for(data['text'])
            if text_instances:
                locations.extend(text_instances)
            else:
                text_blocks = page.get_text("blocks")
                for block in text_blocks:
                    block_text = block[4]
                    if data['text'] in block_text:
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
    

    def redact(self):
        if self.path.endswith("pdf"):
            self.redact_pdf()
        elif self.path.endswith((".mp3", "wav")):
            self.redact_audio()

if __name__ == "__main__":
    path = r'tests\audio\even_more_sensitive.mp3'
    redactor = Redactor(path, plan_type="pro")
    sensitive_data = redactor.redact()
    print(redactor.sensitive_datas)