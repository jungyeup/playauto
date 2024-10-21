import logging
import requests
import time
import pandas as pd
import os
from typing import List, Dict, Any
from AnswerGenerator import AnswerGenerator
from ocr_handler import OCRHandler
from report_generator import ReportGenerator
from dotenv import load_dotenv
import ctypes
import threading
import warnings
import mysql.connector
from mysql.connector import Error
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

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

answer_generator = AnswerGenerator(openai_api_key=OPENAI_API_KEY)
ocr_handler = OCRHandler()
report_generator = ReportGenerator()

class QuestionHandler:
    def __init__(self, ocr_handler=ocr_handler, answer_generator=answer_generator):
        self.ocr_handler = ocr_handler
        self.shown_messages = set()
        self.answer_generator = answer_generator
        self.excel_file_path = "data/questions_answers.xlsx"
        self.init_excel_file(self.excel_file_path)
        self.df = pd.DataFrame(columns=[
            "작성시간", "상품명", "모델명", "문의내용", 
            "답변초안", "답변내용", "수정내용", "OCR내용", 
            "특이사항", "LastModifiedTime"
        ])
        if os.path.exists(self.excel_file_path):
            self.existing_data = pd.read_excel(self.excel_file_path)
        else:
            self.existing_data = pd.DataFrame()
        self.results_to_process = []

        # Initialize database connection
        self.connection = self.create_db_connection()

    def create_db_connection(self):
        """Establish a connection to the MySQL database."""
        try:
            connection = mysql.connector.connect(**DB_CONFIG)
            if connection.is_connected():
                logger.info("Database connection established successfully.")
            return connection
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return None

    def init_excel_file(self, file_path):
        if not os.path.exists(file_path):
            df = pd.DataFrame(columns=[
                "작성시간", "상품명", "모델명", 
                "문의내용", "답변초안", "답변내용", 
                "수정내용", "OCR내용", "특이사항", 
                "LastModifiedTime"
            ])
            df.to_excel(file_path, index=False)

    def save_to_excel_and_db(self, new_data: pd.DataFrame):
        """Save the data to both Excel and MySQL database."""
        self.save_to_excel(new_data)
        self.save_to_db(new_data)

    def save_to_excel(self, new_data: pd.DataFrame):
        if os.path.exists(self.excel_file_path):
            df_existing = pd.read_excel(self.excel_file_path)
            combined_df = pd.concat([df_existing, new_data], ignore_index=True)
        else:
            combined_df = new_data

        combined_df.to_excel(self.excel_file_path, index=False)
        logger.info(f"Saved questions and answers to {self.excel_file_path}")

    def save_to_db(self, new_data: pd.DataFrame):
        """Insert the data from the DataFrame into the MySQL database."""
        if not self.connection:
            logger.error("No database connection available for saving data.")
            return

        cursor = self.connection.cursor()
        query = """
        INSERT INTO inquiries (
            작성시간, 상품명, 모델명, 문의내용, 
            답변초안, 답변내용, 수정내용, OCR내용, 
            특이사항, LastModifiedTime
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        for _, row in new_data.iterrows():
            try:
                cursor.execute(query, tuple(row))
                logger.info("Row inserted into database successfully.")
            except Error as e:
                logger.error(f"Failed to insert row into MySQL: {e}")

        self.connection.commit()
        cursor.close()

    def fetch_batch_orders(self, order_code=None, master_code=None, prod_code=None, tel=None):
        base_url = 'http://playauto-api.playauto.co.kr/emp/v1/orders/'
        headers = {'X-API-KEY': API_KEY}
        params = {}

        if order_code:
            params['OrderCode'] = order_code
        else:
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
                if (current_time - last_modified_time).days >= 30:
                    return True
        return False
    
    def update_answers_from_api(self):
        try:
            headers = {'X-API-KEY': API_KEY}
            params = {'states': '전송완료', 'page': 1, 'count': 100}
            
            response = requests.get(INQUIRY_LIST_URL, headers=headers, params=params)
            response.raise_for_status()
            inquiries = response.json()
            
            if not isinstance(inquiries, list):
                logger.error("Expected response to be a list.")
                return

            if not os.path.exists('data/questions_answers.xlsx'):
                logger.error("Excel file not found.")
                return

            df = pd.read_excel('data/questions_answers.xlsx')

            for inquiry in inquiries:
                if isinstance(inquiry, dict):
                    qsubject = inquiry.get('QSubject', '')
                    qcontent = inquiry.get('QContent', '')
                    acontent = inquiry.get('AContent', '')

                    combined_question = ". ".join(filter(None, [qsubject, qcontent])).strip()

                    matching_rows = df[df['문의내용'] == combined_question]

                    if not matching_rows.empty:
                        current_answer = matching_rows.iloc[0]['답변내용']
                        
                        if current_answer == acontent:
                            continue
                        
                        logger.info(f"Updating answer for question: {combined_question}")

                        df.loc[matching_rows.index, '답변내용'] = acontent

            df.to_excel('data/questions_answers.xlsx', index=False)
            logger.info("Excel file updated successfully.")

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def generate_answer(self, question: str, summaries: List[str], product_info: Dict[str, Any], inquiry_data: Dict[str, Any], product_name: str = None, comment_time: str = None, order_states: str = None, image_urls: List[str] = None) -> str:
        validated_summaries = [{'summary': s} for s in summaries]
        try:
            return self.answer_generator.generate_answer(
                question=question,
                combined_summary=summaries[0],
                product_info=product_info,
                inquiry_data=inquiry_data,
                product_name=product_name,
                comment_time=comment_time,
                order_states=order_states,
                image_urls=image_urls
            )
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
            elif isinstance(response_data, dict) and response_data.get('status'):
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
            
            filtered_image_urls = [url for url in all_image_urls if url.startswith('https://cdn.kzmoutdoor.com')]

            logger.info(f"Product info fetched for master code {master_code}: {product_info}")
            logger.info(f"Filtered image URLs: {filtered_image_urls}")

            return product_info, filtered_image_urls
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch product information: {e}")
            return {}, []

    def handle_emergency(self, qtype: str, clean_question: str):
        message_identifier = f"{qtype}:{clean_question}"
        if message_identifier in self.shown_messages:
            logger.info("Emergency message already shown, skipping pop-up.")
            return
        
        self.shown_messages.add(message_identifier)

        def show_message_box():
            ctypes.windll.user32.MessageBoxW(
                0,
                f"긴급공지 / 긴급메세지:\n\n{qtype}:\n{clean_question}",
                "긴급공지 / 긴급메세지",
                0x10  
            )

        message_thread = threading.Thread(target=show_message_box)
        message_thread.start()

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
        order_code = inquiry.get('OrderCode')  # Fetch the OrderCode if available

        orders_data = self.fetch_batch_orders(order_code, master_code, prod_code, tel)

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

        if not all([qna_id, question]):
            logger.error("Missing 'Number' or 'QContent' in inquiry.")
            return

        try:
            existing_entries = self.existing_data[self.existing_data['상품명'] == product_name]

            force_ocr = self.should_run_ocr(existing_entries)

            if not existing_entries.empty and not force_ocr:
                try:
                    if 'LastModifiedTime' in existing_entries and not existing_entries['LastModifiedTime'].isnull().all():
                        latest_ocr_entry = existing_entries.loc[existing_entries['LastModifiedTime'].idxmax()]
                        combined_summaries = latest_ocr_entry['OCR내용']
                        logger.info(f"Using existing OCR content for product: {product_name}")
                    else:
                        combined_summaries = "OCR을 통한 유효한 데이터를 찾을 수 없습니다."
                        logger.info(f"No valid 'LastModifiedTime' found for product: {product_name}")
                except IndexError:
                    logger.error("Failed to obtain latest OCR entry due to index error.")
                    combined_summaries = "OCR 내용을 가져오는 중 오류가 발생했습니다."
            else:
                logger.info(f"No recent OCR content found for product: {product_name} or forcing OCR...")
                html_content = inquiry.get('HTMLContent', '')
                html_ocr_summaries = self.ocr_handler.ocr_from_html_content(html_content)
                image_ocr_summaries = self.ocr_handler.ocr_from_image_urls(all_image_urls)
                combined_ocr_summaries = html_ocr_summaries + image_ocr_summaries

                combined_summaries = self.ocr_handler.summarize_summaries(combined_ocr_summaries)
                if not combined_ocr_summaries:
                    combined_summaries = "OCR을 통한 유효한 데이터를 찾을 수 없습니다."

            self.answer_generator.check_if_important_question(question)

            answer = self.generate_answer(
                question=question,
                summaries=[combined_summaries],
                product_info=product_info,
                inquiry_data=inquiry,
                product_name=product_name,
                comment_time=write_date,
                order_states=order_state_summary,
                image_urls=all_image_urls
            )

            self.results_to_process.append({
                "qna_id": qna_id,
                "question": question,
                "draft_answer": answer,  # Store the draft answer.
                "answer": "",  # Leave the formal answer empty initially.
                "original_answer": answer,
                "modification_note": "자동 생성된 답변",
                "special_note": "답변 생성됨",
                "status": "Generated",
                "product_name": product_name,
                "master_code": master_code,
                "combined_ocr_summaries": combined_summaries,
                "write_date": write_date,
                "product_info": product_info,
                "inquiry": inquiry
            })

        except Exception as e:
            logger.error(f"Error generating answer for inquiry: {e}")

    def save_product_info(self, master_codes: str):
        # Split the input string into individual master codes
        master_code_list = [code.strip() for code in master_codes.split(',') if code.strip()]

        for master_code in master_code_list:
            # Unpack the returned tuple from fetch_product_info
            product_info, filtered_image_urls = self.fetch_product_info(master_code=master_code)
            
            if not product_info:
                print(f"Failed to fetch product information for master code: {master_code}")
                continue

            # Run OCR regardless of existing content
            logger.info(f"Running OCR for master code: {master_code}...")

            html_content = product_info.get('Content', '')
            html_ocr_summaries = self.ocr_handler.ocr_from_html_content(html_content)
            image_ocr_summaries = self.ocr_handler.ocr_from_image_urls(filtered_image_urls)
            combined_ocr_summaries = html_ocr_summaries + image_ocr_summaries

            combined_summaries = self.ocr_handler.summarize_summaries(combined_ocr_summaries)
            if not combined_ocr_summaries:
                combined_summaries = "OCR을 통한 유효한 데이터를 찾을 수 없습니다."

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            # Check existing entries for the master code
            existing_entries = self.existing_data[self.existing_data['모델명'] == master_code]

            if not existing_entries.empty:
                # Update existing entry
                index_to_update = existing_entries.index[0]
                self.df.loc[index_to_update, 'OCR내용'] = combined_summaries
                self.df.loc[index_to_update, 'LastModifiedTime'] = current_time
                print(f"OCR content updated for master code: {master_code}")
            else:
                # Add new entry
                new_entry = pd.DataFrame({
                    "작성시간": [current_time],
                    "상품명": [product_info.get('ProdName', 'Unknown Product')],
                    "모델명": [master_code],
                    "수정내용": ["정보 수집 완료"],
                    "OCR내용": [combined_summaries],
                    "특이사항": ["상품 정보 업데이트"],
                    "LastModifiedTime": [current_time]
                })

                self.df = pd.concat([self.df, new_entry], ignore_index=True)

            # Save changes to Excel each loop iteration
            self.save_to_excel(self.df)
        print("Product information processing completed.")

    def process_user_inputs(self):
        for result in self.results_to_process:
            success = False

            # Automatically upload the answer without user input
            result['answer'] = result['draft_answer']
            success = True

            if success:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                self.send_answer(result['qna_id'], result['answer'])
                new_entry = pd.DataFrame({
                    "작성시간": [current_time],
                    "상품명": [result['product_name']],
                    "모델명": [result['master_code']],
                    "문의내용": [result['question']],
                    "답변초안": [result['draft_answer']],
                    "답변내용": [result['answer']],
                    "수정내용": [result['modification_note']],
                    "OCR내용": [result['combined_ocr_summaries']],
                    "특이사항": [result['special_note']],
                    "LastModifiedTime": [current_time]
                })

                self.df = pd.concat([self.df, new_entry], ignore_index=True)
                self.save_to_excel_and_db(new_entry)

        self.results_to_process.clear()

def main_menu():
    handler = QuestionHandler()

    while True:
        print("\nSelect an option:")
        print("1. 신규문의 답변생성 (Generate answers for new inquiries)")
        print("2. 상품 정보수집 (Collect product information)")
        print("3. 종료 (Exit)")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            handler.update_answers_from_api()
            inquiries = handler.fetch_inquiries()

            if isinstance(inquiries, list):
                for inquiry in inquiries:
                    if isinstance(inquiry, dict):
                        handler.handle_question(inquiry)
            else:
                logger.error("Expected 'inquiries' to be a list.")

            handler.process_user_inputs()

        elif choice == '2':
            master_code = input("Enter master_code for product information collection: ").strip()
            handler.save_product_info(master_code)

        elif choice == '3':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main_menu()