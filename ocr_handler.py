import os
import requests
import base64
from PIL import Image, ImageEnhance, ImageFile, ImageFilter
from io import BytesIO
from bs4 import BeautifulSoup
import pytesseract
import numpy as np
import cv2
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
import re

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

class OCRHandler:
    def __init__(self):
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in the .env file")
        
        self.client = OpenAI(api_key=openai_api_key)
        self.model = "gpt-4o-2024-11-20"
        self.system_prompt = """
        당신은 한국어로 작업하며, OCR 및 다양한 데이터 소스(이미지, HTML 등)에서 추출된 정보를 기반으로 상품의 형태와 생김새를 묘사하고 분석하고 상품 정보를 상세히 분석하고 정리하는 AI 에이전트입니다. 모든 입력 데이터에 기반하여 고객이 상품의 모든 특징을 정확히 이해할 수 있도록 상세하고 구체적인 정보를 제공합니다.

        핵심 규칙
        정확성 및 구체성:

        입력된 데이터를 기반으로만 작업하며, 추측이나 근거 없는 정보를 추가하지 않습니다.
        상품의 형태와 생김새에 대하여 자세히 묘사하고 설명합니다. 
        데이터를 기반으로 형태나 생김새에 맞는 넓이를 산출합니다.
        상품의 특징, 사양, 장단점, 사용법 등을 매우 세세하고 구체적으로 설명합니다.
        정확히 언급이 돼 있지 않더라도 데이터소스의 내용상 합리적 추론이 가능하다면 구체적으로 설명합니다. (예: 6각형 모양의 텐트의 한변의 길이가 250cm일경우 6각현 탠트의 넓이를 구합니다)
        "상세 페이지 참조" 또는 비슷한 표현은 사용 금지입니다.
        
        상품 정보 정리:

        상품 이름, 브랜드, 모델, 주요 특징, 사양, 재질, 사이즈, 옵션별 차이 등을 구분하여 정리합니다.
        사이즈 표기:
        "290x220x(h)195cm"은 가로 290, 세로 220, 높이 195(cm)를 의미합니다.
        텐트 등 특수한 경우: "290(240)x220x(h)195cm"는 긴쪽 가로 290, 짧은쪽 가로 240, 세로 220, 높이 195(cm)로 해석합니다.
        옵션이나 구성품이 있다면 각각 구분하여 설명합니다.
        데이터 통합:

        이미지 정보: 상품 외관, 색상, 로고, 텍스트 등.
        OCR 데이터: 라벨, 설명서, 광고 텍스트에서 추출한 정보.
        HTML 데이터: 기술 사양, 고객 리뷰, 가격, 할인 정보 등.
        모든 출처에서 얻은 정보를 종합해 일관된 결과를 제공합니다.
        고객 관점 반영:

        고객이 중요하게 여길 수 있는 항목(예: 이너텐트 사이즈, 호환성, 사용법, 내구성, 가격 대비 가치 등)을 강조합니다.
        유통기한 정보가 있다면, 가장 짧은 제품의 유통기한 기준을 명시합니다.
        구조화된 정보 제공:

        가독성을 높이기 위해 표, 리스트, 순서 목록 등을 활용합니다.
        필요한 경우 데이터 간의 비교를 추가해 정보를 명확히 전달합니다.
        호환성과 사용법:

        제품의 사용법, 호환성, 주의사항을 상세히 설명합니다.
        조립 및 사용 방법, 유지 관리 팁을 구체적으로 안내합니다.
        """

    @staticmethod
    def preprocess_image(img, upscale_factor=4):
        """
        Preprocesses the image for OCR by resizing, enhancing contrast, denoising, and thresholding.
        """
        try:
            # Resize the image for better OCR accuracy
            img = img.resize((img.width * upscale_factor, img.height * upscale_factor), Image.LANCZOS)
            
            # Enhance the contrast of the image
            enhancer = ImageEnhance.Contrast(img)
            enhanced_img = enhancer.enhance(2.0)  # Adjusted contrast factor
            
            # Convert to a NumPy array for OpenCV processing
            enhanced_img_np = np.array(enhanced_img)

            # Convert to grayscale
            gray_img = cv2.cvtColor(enhanced_img_np, cv2.COLOR_BGR2GRAY)

            # Apply Otsu's thresholding for better binarization
            _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Denoise the image
            denoised_img = cv2.fastNlMeansDenoising(binary_img, None, h=30, templateWindowSize=7, searchWindowSize=21)

            # Convert back to PIL Image for compatibility
            processed_img = Image.fromarray(denoised_img)
            return processed_img
        except Exception as e:
            raise RuntimeError(f"Error in image preprocessing: {e}")

    @staticmethod
    def split_image_vertically(img, max_height=3000, overlap=200):
        width, height = img.size
        if height <= max_height:
            return [img]
        
        split_images = []
        for i in range(0, height, max_height - overlap):
            box = (0, i, width, min(i + max_height, height))
            split_images.append(img.crop(box))
        
        return split_images

    @staticmethod
    def clean_ocr_output(ocr_text):
        """
        Clean and refine OCR output to remove unwanted characters and extract numeric values.
        """
        # Remove all characters except digits
        return re.sub(r'[^0-9]', '', ocr_text)

    @staticmethod
    def ocr_from_image(img):
        """
        Perform OCR on a preprocessed image using pytesseract with additional configurations.
        """
        try:
            # Preprocess the image
            preprocessed_img = OCRHandler.preprocess_image(img)

            # Split large images into smaller parts if necessary
            split_images = OCRHandler.split_image_vertically(preprocessed_img)

            # Perform OCR on each split image
            ocr_results = []
