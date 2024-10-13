import requests
import logging
import time
import pandas as pd
import os
from typing import List, Dict, Any
from AnswerGenerator import AnswerGenerator
from ocr_handler import OCRHandler
from report_generator import ReportGenerator
from dotenv import load_dotenv
import ctypes
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Suppress warnings from BeautifulSoup
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv('PLAYAUTO_API_KEY')
BASE_URL = 'http://playauto-api.playauto.co.kr/emp'
INQUIRY_LIST_URL = f'{BASE_URL}/v1/qnas/'
ANSWER_URL = f'{BASE_URL}/v1/qnas/'
PRODUCT_URL_TEMPLATE = f'{BASE_URL}/v1/prods/{{}}'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

answer_generator = AnswerGenerator(openai_api_key=OPENAI_API_KEY)
ocr_handler = OCRHandler()
report_generator = ReportGenerator()

class QuestionHandler:
    def __init__(self, ocr_handler=ocr_handler, answer_generator=answer_generator, report_generator=report_generator):
        self.ocr_handler = ocr_handler
        self.answer_generator = answer_generator
        self.report_generator = report_generator
        self.excel_file_path = "data/questions_answers.xlsx"
        self.init_excel_file(self.excel_file_path)
        self.df = pd.DataFrame(columns=[
            "작성시간", "상품명", "문의내용", "답변내용", "수정내용", "OCR내용", "특이사항", "Product Summary", "LastModifiedTime"
        ])
        if os.path.exists(self.excel_file_path):
            self.existing_data = pd.read_excel(self.excel_file_path)
        else:
            self.existing_data = pd.DataFrame()
        self.results_to_process = []

    def init_excel_file(self, file_path):
        if not os.path.exists(file_path):
            df = pd.DataFrame(columns=[
                "작성시간", "상품명", "문의내용", "답변내용", "수정내용", "OCR내용", "특이사항", "Product Summary", "LastModifiedTime"
            ])
            df.to_excel(file_path, index=False)

    def fetch_batch_orders(self, master_code=None, prod_code=None, tel=None):
        base_url = 'http://playauto-api.playauto.co.kr/emp/v1/orders/'
        headers = {'X-API-KEY': API_KEY}
        
        params = {}
        
        if master_code:
            params['MasterCode'] = master_code
        if prod_code:
            params['ProdCode'] = prod_code
        if tel:
            params['tel'] = tel

        try:
            logger.debug(f"Fetching orders with params: {params}")
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            orders_data = response.json()
            logger.info(f"Fetched orders successfully: {orders_data}")
            return orders_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch orders: {e}")
            return []

    def fetch_inquiries(self) -> List[Dict[str, Any]]:
        headers = {'X-API-KEY': API_KEY}
        params = {'states': '신규', 'page': 1, 'count': 100}
        try:
            response = requests.get(INQUIRY_LIST_URL, headers=headers, params=params)
            response.raise_for_status()
            response_data = response.json()

            logger.debug(f"Fetched inquiry response: {response_data}")

            if not isinstance(response_data, list):
                logger.error(f"Unexpected response format, expected list but got {type(response_data)}")
                return []

            for item in response_data:
                if not isinstance(item, dict):
                    logger.error("Non-dictionary item found in response_data.")
                    return []
            return response_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch inquiries: {e}")
            return []

    def should_run_ocr(self, existing_entry):
        if existing_entry.empty:
            return True
        else:
            last_modified_time = existing_entry.iloc[0].get('LastModifiedTime')
            if last_modified_time:
                last_modified_time = pd.to_datetime(last_modified_time)
                current_time = pd.Timestamp.now()
                if (current_time - last_modified_time).days >= 7:
                    return True
        return False

    def generate_answer(self, question: str, summaries: List[str], product_info: Dict[str, Any], inquiry_data: Dict[str, Any], product_name: str = None, comment_time: str = None, order_states: str = None, image_urls: List[str] = None) -> str:
        validated_summaries = [{'summary': s} for s in summaries]
        try:
            return self.answer_generator.generate_answer(question, validated_summaries, product_info, inquiry_data, product_name, comment_time, order_states, image_urls)
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise RuntimeError(f"Failed to generate answer: {e}")

    def send_answer(self, qna_id: int, answer: str) -> dict:
        headers = {
            'X-API-KEY': API_KEY,
            'Content-Type': 'application/json'
        }
        data = {
            'overWrite': False,
            'data': [{
                'number': str(qna_id),
                'Asubject': "문의 답변 드립니다.",
                'AContent': answer
            }]
        }

        try:
            response = requests.patch(ANSWER_URL, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()

            logger.debug(f"Response data type: {type(response_data)}, content: {response_data}")

            if isinstance(response_data, list):
                if response_data and isinstance(response_data[0], dict):
                    result_status = response_data[0].get('status', None)
                    if result_status:
                        logger.info(f"Answer sent successfully: {response_data}")
            elif isinstance(response_data, dict):
                if response_data.get('status'):
                    logger.info(f"Answer sent successfully: {response_data}")
            else:
                logger.error(f"Unexpected response data format: {type(response_data)}")
            return response_data

        except requests.exceptions.RequestException as e:
            error_content = response.text if response else ''
            logger.error(f"Failed to send answer: {e}, response content: {error_content}")
            raise RuntimeError(f"Failed to send answer: {e}, response content: {error_content}")

    def fetch_product_info(self, master_code: str) -> Dict[str, Any]:
        headers = {'X-API-KEY': API_KEY}
        product_url = PRODUCT_URL_TEMPLATE.format(master_code)
        try:
            response = requests.get(product_url, headers=headers)
            response.raise_for_status()
            product_info = response.json()

            if not isinstance(product_info, dict):
                logger.error(f"Unexpected product info format: {type(product_info)}")
                return {}, []

            image_urls = [product_info.get(f'Image{i}') for i in range(1, 11) if product_info.get(f'Image{i}')]
            content_image_urls = self.ocr_handler.extract_image_urls_from_html(product_info.get('Content', ''))
            content_ad_image_urls = self.ocr_handler.extract_image_urls_from_html(product_info.get('ContentAd', ''))
            all_image_urls = image_urls + content_image_urls + content_ad_image_urls
            
            logger.info(f"Product info fetched for master code {master_code}: {product_info}")
            logger.info(f"Extracted image URLs: {all_image_urls}")

            return product_info, all_image_urls
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch product information: {e}")
            return {}, []

    def generate_product_summary(self, product_info: Dict[str, Any], summaries: List[str]) -> str:
        product_name = product_info.get('ProdName', '해당 제품')

        product_details = f"상품명: {product_name}\n" + "\n".join([f"{key}: {value}" for key, value in product_info.items() if key != 'ProductName'])
        summary_prompt = f"""
        제공된 제품 정보 및 OCR 요약을 바탕으로 다음 제품에 대해 종합적인 분석을 해주세요. 상세한 설명, 사용 방법, 타겟 소비자층, 가격, 고객 리뷰 등을 포함해 주세요.
        제품명: {product_name}
        제품 정보: {product_details}
        OCR 요약: {' '.join(summaries)}
        """

        try:
            chat_completion = self.answer_generator.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": self.answer_generator.system_prompt},
                    {"role": "user", "content": summary_prompt}
                ]
            )
            summary = chat_completion.choices[0].message.content
            return summary

        except Exception as e:
            logger.error(f"Failed to generate product summary: {e}")
            return "제품 요약을 생성할 수 없습니다."

    def handle_emergency(self, qtype: str, clean_question: str):
        ctypes.windll.user32.MessageBoxW(
            0,
            f"긴급공지 / 긴급메세지:\n\n{qtype}:\n{clean_question}",
            "긴급공지 / 긴급메세지",
            0x10  # MB_ICONHAND (red warning icon with OK button)
        )

    def handle_question(self, inquiry):
        logger.debug(f"Handling inquiry: {inquiry}")

        if not isinstance(inquiry, dict):
            logger.error(f"Invalid inquiry format, expected dict but got: {type(inquiry)}")
            return

        qna_id = inquiry.get('Number')
        qsubject = inquiry.get('QSubject', '')
        qcontent = inquiry.get('QContent', '')

        master_code = inquiry.get('MasterCode')
        prod_code = inquiry.get('ProdCode')
        tel = inquiry.get('QTel') or inquiry.get('QHtel')
        
        orders_data = self.fetch_batch_orders(master_code, prod_code, tel)

        if isinstance(orders_data, list):
            order_states = [order.get('OrderState', 'Unknown State') for order in orders_data if isinstance(order, dict)]
            order_state_summary = ", ".join(order_states)
        else:
            order_state_summary = 'No Orders Found'

        question_parts = [qsubject, qcontent]
        question = ". ".join(part for part in question_parts if part).strip()

        write_date = inquiry.get('WriteDate')
        qtype = inquiry.get('QType')

        product_info, all_image_urls = {}, []
        product_name = inquiry.get('ProductName', 'Unknown Product')

        if master_code:
            product_info, all_image_urls = self.fetch_product_info(master_code)
            product_name = product_info.get('ProdName', product_name)

        if qtype in ['긴급공지', '긴급메세지']:
            clean_question = BeautifulSoup(question, "html.parser").get_text()
            self.handle_emergency(qtype, clean_question)
            return

        if not all([qna_id, question]):
            logger.error("Missing 'Number' or 'QContent' in inquiry.")
            return

        try:
            existing_entry = self.existing_data[self.existing_data['상품명'] == product_name]
            force_ocr = self.should_run_ocr(existing_entry)

            if not existing_entry.empty and not force_ocr:
                logger.info(f"Using existing OCR content for product: {product_name}")
                ocr_content = existing_entry.iloc[0]['OCR내용']
                combined_ocr_summaries = ocr_content if isinstance(ocr_content, list) else [ocr_content]
            else:
                logger.info(f"No existing OCR content found for product: {product_name} or forcing OCR...")
                html_ocr_summaries = self.ocr_handler.ocr_from_html_content(inquiry.get('HTMLContent', '')) if 'HTMLContent' in inquiry else []
                image_ocr_summaries = self.ocr_handler.ocr_from_image_urls(all_image_urls)
                combined_ocr_summaries = html_ocr_summaries + image_ocr_summaries
                if not combined_ocr_summaries:
                    combined_ocr_summaries = ["OCR을 통한 유효한 데이터를 찾을 수 없습니다."]

            self.answer_generator.check_if_important_question(question)

            # Pass `order_state_summary` and `all_image_urls` to generate_answer
            answer = self.generate_answer(
                question,
                combined_ocr_summaries,
                product_info,
                inquiry,
                product_name,
                write_date,
                order_state_summary,
                all_image_urls
            )
            self.results_to_process.append({
                "qna_id": qna_id,
                "question": question,
                "answer": answer,
                "original_answer": answer,
                "modification_note": "자동 생성된 답변",
                "special_note": "답변 생성됨",
                "status": "Generated",
                "product_name": product_name,
                "combined_ocr_summaries": combined_ocr_summaries,
                "write_date": write_date,
                "product_info": product_info,
                "inquiry": inquiry
            })

        except Exception as e:
            logger.error(f"Error generating answer for inquiry: {e}")

    def process_user_inputs(self):
        for result in self.results_to_process:
            success = False
            while True:
                print(f"\n\n====================================================================================================")
                print(f"상품명: {result['product_name']}\n\n질문:\n{result['question']}\n\n생성된 답변:\n{result['answer']}\n")
                user_input = input("답변을 수정하거나 추가하십시오. 업로드를 취소하려면 '취소' 혹은 'c'를 입력하고, 재생성을 원하시면 '재생성' 혹은 're', 그대로 업로드 하실려면 Enter를 누르세요: ").strip().lower()

                if user_input in ['c', '취소']:
                    print("업로드가 취소되었습니다.")
                    result['status'] = "Canceled"
                    result['special_note'] = "답변 업로드 취소됨"
                    break

                if user_input in ['re', '재생성']:
                    result['answer'] = self.generate_answer(
                        result['question'],
                        result['combined_ocr_summaries'],
                        result['product_info'],
                        result['inquiry'],
                        result['product_name'],
                        result['write_date'],
                        result['order_states'],
                        result['image_urls']
                    )
                    result['original_answer'] = result['answer']
                    result['modification_note'] = "자동 생성된 답변 (재생성)"
                    result['status'] = "Regenerated"
                    result['special_note'] = "답변 재생성됨"
                elif user_input:
                    result['answer'] = self.answer_generator.revise_answer(user_input, result['original_answer'])
                    result['modification_note'] = f"수정된 답변: {user_input}"
                    result['status'] = "Modified"
                    result['special_note'] = "답변 수정됨"
                else:
                    success = True
                    break

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            if success:
                self.send_answer(result['qna_id'], result['answer'])
                new_entry = pd.DataFrame({
                    "작성시간": [current_time],
                    "상품명": [result['product_name']],
                    "문의내용": [result['question']],
                    "답변내용": [result['answer']],
                    "수정내용": [result['modification_note']],
                    "OCR내용": [result['combined_ocr_summaries']],
                    "특이사항": [result['special_note']],
                    "Product Summary": [self.generate_product_summary(result['product_info'], result['combined_ocr_summaries'])],
                    "LastModifiedTime": [current_time]
                })
                self.df = pd.concat([self.df, new_entry], ignore_index=True)
                self.save_to_excel()

        self.results_to_process.clear()

    def save_to_excel(self):
        if os.path.exists(self.excel_file_path):
            df_existing = pd.read_excel(self.excel_file_path)
            combined_df = pd.concat([df_existing, self.df], ignore_index=True)
        else:
            combined_df = self.df

        combined_df.to_excel(self.excel_file_path, index=False)
        logger.info(f"Saved questions and answers to {self.excel_file_path}")

    def record_report(self, report):
        logger.debug(f"Recording report: {report}")
        self.report_generator.generate_reports([report])

def process_inquiries():
    handler = QuestionHandler()

    while True:
        inquiries = handler.fetch_inquiries()

        if isinstance(inquiries, list):
            for inquiry in inquiries:
                if isinstance(inquiry, dict):
                    handler.handle_question(inquiry)
        else:
            logger.error("Expected 'inquiries' to be a list.")
        
        handler.process_user_inputs()
        time.sleep(60)

if __name__ == "__main__":
    process_inquiries()