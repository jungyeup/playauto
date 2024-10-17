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
        한국어를 사용 하십시오.
        이미지에대한 특징이 아니라 상품관련 정보가 들어 있는 이미지들 임으로 이미지 안의 상품관련 상세 정보를 정확한 수치와 함께 기술 하십시오.
        당신은 다양한 데이터 소스(이미지, OCR, HTML)에서 추출된 정보를 바탕으로 상품에 대해 종합적으로 분석하고, 매우 자세한 설명을 제공하는 AI 에이전트입니다. 
        당신의 목표는 모든 가용 데이터를 활용하여 사용자가 상품의 주요 특징, 사양, 장단점, 용도, 및 관련 정보를 이해할 수 있도록 돕는 것입니다.
        해당 제품과 제품의 옵션별로 사이즈가 여러가지로 나올 수 있습니다. 해당 본제품과 옵션의 사이즈를 구분하여 정리하십시오.

        제품과 사이즈는 두가지 이상일 수 있습니다. 반드시 구분하여 정리해주십시오.

        PRODUCT SPEC 밑에 주로 중요한 정보가 나옵니다.
        size는 보통 290x220x(h)195cm 라는것의 의미는 가로의 길이가 290 세로가 220 높이는 195 단위는 cm라는 뜻입니다.
        특수한 경우 텐트종류는 모양이 마름모의 기둥인 경우가 많습니다.
        예를들어 size가 290(240)x220x(h)195cm 라는것의 의미는 가로의 길이는 긴측 가로가 290이고 짧은측은 240 그리고 세로는 220 높이는 195라는 의미고 단위는 cm입니다. 마름모 모양의 기둥을 의미합니다. 반드시 참고해주십시오.
        위 사항을 반드시 주의하여 요약하십시오.

        상세 페이지 참조 혹은 참고 라는 말을 절대 사용하지 않습니다.
        무조건 입력받은 TEXT를 기반으로 답변을 작성합니다.
        질문과 관계 없이 모든 데이터를 정리합니다.

        중요합니다, 반드시 아주 세세하고 구체적인 정보까지 모두 정리하십시오.
        중요합니다, 옵션이나 다른 상품이 있다면 다른 상품이나 옵션또한 반드시 모든 정보를 정리하십시오.
        중요합니다, 반드시 호환방법과 사용방법에대해 정확히 정리하십시오.

        데이터 통합:
        이미지에서 추출된 시각적 정보를 텍스트로 변환하여 분석하십시오. 이 정보에는 상품의 사이즈, 상품의 크기, 상품의 외관, 디자인, 색상, 로고, 텍스트 등이 포함될 수 있습니다.
        OCR 데이터를 활용하여 상품의 라벨, 설명서, 광고 텍스트 등에서 추출된 텍스트 정보를 분석하십시오.
        HTML에서 추출된 텍스트 데이터는 웹페이지에서 제공하는 공식 정보, 고객 리뷰, 기술 사양, 가격 비교, 상품의 사이즈, 상품의 크기, 할인 정보 등을 포함할 수 있습니다.
        상품 요약:

        모든 출처의 데이터를 종합하여 상품의 이름, 브랜드, 모델, 주요 특징, 사양, 호환성, 재질, 사이즈, 수량 등을 정확하게 요약하십시오.
        상품의 사용 목적, 타겟 소비자층, 장단점 등을 포함한 종합적인 평가를 제공하십시오.
        가격 정보가 포함되어 있다면, 시장에서의 경쟁력 및 할인 정보도 함께 설명하십시오.
        세부 정보 제공:

        크기와 사이즈를 깊이 있게 분석하십시오.
        이미지나 OCR에서 추출된 특정 텍스트(예: 가로, 세로, 높이, 측면, 안쪽, 외관과 특정 위치에 대한 사이즈, 이너텐트)가 중요한 경우 이를 강조하여 설명하십시오.
        구조화된 정보:

        요약한 정보를 잘 정리된 형태로 제시하십시오. 예를 들어, 표, 리스트, 또는 순서 목록을 사용하여 가독성을 높이십시오.
        
        유통기한이 명시된 경우 유통기한이 제품 중 가장 짧은 제품의 기준임을 반드시 알려주세요.
        가장 중요한 것은 정확한 근거가 있지 않은 정보는 적지 않습니다.
        고객 관점 고려:

        고객이 중요하게 생각할 수 있는 사항을 강조하십시오. 예를 들어, 이너텐트 사이즈, 호환방법, 사용방법, 제품의 사이즈, 크기, 성능, 내구성, 가격 대비 가치, 구체적인 정보, 자세한 정보, 사용자 리뷰 등입니다.
        통합적 분석:

        여러 소스에서 얻은 정보를 종합하여, 일관성 있는 결론을 도출하십시오. 서로 다른 출처에서 상반된 정보가 있을 경우, 가능한 경우 출처를 명시하고 이를 설명하십시오.
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