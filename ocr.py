import fitz
from PIL import Image
import io
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


def extract_text_and_boxes(pdf_path):
    doc = fitz.open(pdf_path)
    results = []

    for page_num  in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        results.append(ocr_result['text'])
        # # print(ocr_result['text'])
        # for i, word in enumerate(ocr_result['text']):
        #     if word.strip():
        #         x, y, w, h = (
        #             ocr_result['left'][i],
        #             ocr_result['top'][i],
        #             ocr_result['width'][i],
        #             ocr_result['height'][i]
        #         )
        #         results.append({
        #             'text': word,
        #             'box': (x, y, x+w, y+h),
        #             'page': page_num
        #         })
                
    return results


print(extract_text_and_boxes(r'tests\pdfs\Notice_Unauthorized_Freshers_Party.pdf'))