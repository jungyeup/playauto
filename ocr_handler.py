import os
import requests
import base64
from PIL import Image, ImageEnhance, ImageFile
from io import BytesIO
from bs4 import BeautifulSoup
import pytesseract
import numpy as np
import cv2
from dotenv import load_dotenv
from openai import OpenAI
from typing import List

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

class OCRHandler:
    def __init__(self):
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in the .env file")
        
        self.client = OpenAI(api_key=openai_api_key)
        self.model = "gpt-4-turbo-2024-04-09"
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
        try:
            img = img.resize((img.width * upscale_factor, img.height * upscale_factor), Image.LANCZOS)
            enhancer = ImageEnhance.Contrast(img)
            enhanced_img = enhancer.enhance(64)
            enhanced_img_np = np.array(enhanced_img)
            gray_img = cv2.cvtColor(enhanced_img_np, cv2.COLOR_BGR2GRAY)
            thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            denoised_img = cv2.fastNlMeansDenoising(thresh_img, None, 30, 7, 21)
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
    def ocr_from_image(img):
        try:
            preprocessed_img = OCRHandler.preprocess_image(img)
            split_images = OCRHandler.split_image_vertically(preprocessed_img)
            ocr_results = []
            for idx, part_img in enumerate(split_images):
                text = pytesseract.image_to_string(part_img, lang='kor+eng')
                ocr_results.append(f"Part {idx + 1}:\n{text.strip()}")
            return "\n".join(ocr_results)
        except Exception as e:
            return f"Error in OCR processing: {e}"

    @staticmethod
    def handle_large_image(img, max_size=60000):
        """Handle large images by resizing if they exceed the maximum dimension."""
        max_dimension = max(img.width, img.height)
        if max_dimension > max_size:
            scale_factor = max_size / max_dimension
            new_width = max(int(img.width * scale_factor), 1)
            new_height = max(int(img.height * scale_factor), 1)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        return img

    @staticmethod
    def ocr_from_image_url(image_url):
        try:
            if 'data:image' in image_url:
                header, encoded = image_url.split(",", 1)
                img_data = base64.b64decode(encoded)
                img = Image.open(BytesIO(img_data))
            else:
                # Stream the image content in parts to ensure completeness
                with requests.get(image_url, stream=True) as response:
                    response.raise_for_status()
                    img_data = BytesIO()
                    for chunk in response.iter_content(chunk_size=8192):
                        img_data.write(chunk)
                    img_data.seek(0)  # Reset buffer position
                    img = Image.open(img_data)

            # Immediately resize large images
            img = OCRHandler.handle_large_image(img)

            return OCRHandler.ocr_from_image(img)
        
        except ValueError as ve:
            return str(ve)
        except requests.exceptions.RequestException as re:
            return f"Request Error in OCR from URL: {re}"
        except OSError as img_err:
            return f"Image handling error: {img_err}"
        except Exception as e:
            return f"General error in OCR from URL: {e}"

    @staticmethod
    def extract_image_urls_from_html(html_content):
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            image_tags = soup.find_all('img')

            image_urls = []
            for img in image_tags:
                src = img.get('src')
                data_src = img.get('data-src')
                if src:
                    image_urls.append(src)
                if data_src:
                    image_urls.append(data_src)
            return image_urls
        except Exception as e:
            raise RuntimeError(f"Error in extracting image URLs from HTML: {e}")

    @staticmethod
    def extract_text_from_html(html_content):
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            raise RuntimeError(f"Error in extracting text from HTML: {e}")

    def ocr_from_html_content(self, html_content):
        try:
            text = self.extract_text_from_html(html_content)
            image_urls = self.extract_image_urls_from_html(html_content)
            ocr_results_from_images = self.ocr_from_image_urls(image_urls)

            return [{"ocr_text": text}] + ocr_results_from_images
        except Exception as e:
            return [{'ocr_text': f"Error in OCR from HTML content: {e}"}]

    def analyze_images_with_function_calling(self, image_urls: List[str]) -> str:
        processed_images = []
        for url in image_urls:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                img_data = BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    img_data.write(chunk)
                img_data.seek(0)
                img = Image.open(img_data)

                # Handle large images by splitting them if needed
                large_img = self.handle_large_image(img)  # resize large images if necessary
                sub_images = self.split_image_vertically(large_img, max_height=3000, overlap=200)

                for sub_img in sub_images:
                    # Preprocess image
                    preprocessed_img = self.preprocess_image(sub_img)

                    # Save to buffer
                    buffered = BytesIO()
                    try:
                        preprocessed_img.save(buffered, format="JPEG")
                        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        processed_images.append(processed_image_base64)
                    except (ValueError, IOError) as saving_error:
                        print(f"Error saving processed image from {url}: {saving_error}")
                        continue  # Skip this image and go to the next

            except requests.exceptions.RequestException as request_error:
                print(f"Error fetching image from {url}: {request_error}")
                continue
            except OSError as img_error:
                print(f"Image handling error for {url}: {img_error}")
                continue
            except Exception as e:
                print(f"General error processing image from {url}: {e}")
                continue

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}} for image in processed_images]},
            ],
        }

        try:
            response = self.client.chat.completions.create(**payload)
            descriptions = [choice.message.content for choice in response.choices]
            print(descriptions)
            return "\n".join(descriptions)
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return "Failed to process images."

    def summarize_ocr_results(self, ocr_results, image_urls):
        summaries = []
        try:
            combined_text = "\n".join([result.get('ocr_text', '') for result in ocr_results if isinstance(result, dict)])
            image_urls_text = "\n".join(image_urls)
            prompt_text = f"Images URLs: {image_urls_text}\nText Content:\n{combined_text}"
            shortened_ocr_text = self.shorten_text_to_token_limit(prompt_text, 8192)

            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": shortened_ocr_text},
                ]
            )

            summary = chat_completion.choices[0].message.content
            summaries.append({'summary': summary})

            return summaries
        except Exception as e:
            summaries.append({'summary': f"Error in generating OCR summary: {e}"})
            return summaries

    @staticmethod
    def shorten_text_to_token_limit(text, token_limit):
        tokens = text.split()
        if len(tokens) > token_limit:
            return " ".join(tokens[:token_limit])
        return text

    def ocr_from_image_urls(self, image_urls):
        ocr_results = []
        for image_url in image_urls:
            ocr_text = self.ocr_from_image_url(image_url)
            if isinstance(ocr_text, str) and ocr_text:
                ocr_results.append({'image_url': image_url, 'ocr_text': ocr_text})
        return ocr_results

    def summarize_summaries(self, image_summaries):
        # Integrate OCR summaries and image analysis into a comprehensive summary
        ocr_texts = " ".join([
            summary.get('ocr_text', '')
            for summary in image_summaries if isinstance(summary, dict)
        ])
        
        # Analyze images and combine results
        image_analysis_descriptions = self.analyze_images_with_function_calling(
            [summary.get('image_url') for summary in image_summaries if 'image_url' in summary]
        )
        
        # Prepare a comprehensive prompt for GPT-4o
        comprehensive_prompt = f"OCR Results:\n{ocr_texts}\nImage Analysis Descriptions:\n{image_analysis_descriptions}"
        
        chat_completion = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": comprehensive_prompt},
            ]
        )
        
        combined_summary = chat_completion.choices[0].message.content
        print(combined_summary)
        return combined_summary