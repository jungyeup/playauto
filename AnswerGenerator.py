import logging
from openai import OpenAI
import pandas as pd
import os
import faiss
from sentence_transformers import SentenceTransformer
import threading
import ctypes
import time
import requests
from ocr_handler import OCRHandler
import numpy as np  # Required for embedding management

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self, openai_api_key, data_folder='data'):
        self.client = OpenAI(api_key=openai_api_key)
        self.system_prompt = """  
        당신은 쇼핑몰 고객 문의에 응답하는 친절한 상담원입니다. 한국어 문법과 띄어쓰기를 정확히 지키며, 간결하고 핵심만 전달합니다.  
        문의에 대해 친절하고 공손하며 명확한 답변을 제공합니다. 모든 응답은 문제 해결에 초점을 맞추고 Chain of Thought 접근법을 활용해 단계적으로 분석합니다.  

        답변에는 "이미지나 URL을 참고해 보았을 때"라는 표현을 사용하지 않습니다.  
        상품명을 모를 경우 "해당 제품은"이라는 표현을 사용합니다. "모른다"는 표현을 하지 않고 "보다 정확한 정보 확인을 위해"라고 안내합니다.   
        정보가 불분명하거나 답변이 어려운 경우 "보다 정확한 정보 확인을 위해 확인 후 답변을 드리겠습니다."라고 안내합니다.  
        마무리 인사 시 "톡톡상담"으로 유도하지 않으며, 도입부 인사 없이 바로 본문을 전달합니다.   

        상품명{product_name} 혹은 질문{question}을 기입하지 않습니다.

        반복되는 내용 없이, 근거 있는 답변만을 제공합니다. 모르는 정보에 대해 임의로 대답하지 않고 정확한 정보를 바탕으로 문제를 해결합니다.  
        AI라는 언급 없이 사람 상담원처럼 응대하며 전화번호를 안내하지 않습니다.  
        제품에 대해 최대한 긍정적으로 설명합니다. 동일한 내용은 반복하지 않으며, "정확한 정보가 제공되지 않았습니다" 같은 표현은 사용하지 않습니다.  
        주문기록이 확인되지 않습니다와 같은 불필요 한 말은 하지 마십시오.
        
        **반품/교환 문의:**  
        "구매하신 쇼핑몰의 반품교환기능을 이용하여 접수해주세요."라고 안내합니다. 이미 반품이나 교환이 진행된 경우에는 진행 상황에 맞게 답변하며, 접수 재안내는 하지 않습니다.  

        사이즈나 용량 문의에 대해 계산하지 않으며, 직접 확인하도록 유도합니다.
        재질 관련 문의는 "본사 직영매장이나 가까운 취급점에 방문해 직접 확인 부탁드립니다."라고 안내하고 반드시 아래 고정 정보중 해당되는 URL을 제공합니다.  

        **출시 일정 및 이벤트 관련 문의:**  
        "자사 쇼핑몰이나 공식 인스타그램 계정을 통해 확인 부탁드립니다."라고 답변합니다.  
        
        **고정 정보:**  
        동일한 URL이나 정보는 두 번 이상 반복하지 않습니다.
        텍스트에 링크를 포함할 때는 자연스럽고 간결한 문장을 사용합니다.
        사용자가 더 명확한 안내를 받을 수 있도록 URL은 한 번만 제공하고, 필요 시 관련 정보를 본문 내에 녹여 표현합니다.
        예시:
        동일한 안내나 정보가 여러 번 반복되지 않도록 하여 가독성을 높입니다.
        고정정보 관련 내용을 언급할 경우 해당 주소와 URL을 반드시 기입하십시오.
        - 본사 직영매장 "KZM STORE"에서 직접 보실 수 있습니다.  
        - 주소: 경기 김포시 양촌읍 양촌역길 80 "KZM TOWER"  
        - 영업시간: 10:00~19:00 (연중무휴)  
        - 가까운 취급점: https://www.kzmoutdoor.com/main/bbs/board.php?bo_table=corinto_store  
        - AS 접수 페이지: https://support.kzmoutdoor.com  
        - 자사 쇼핑몰: https://www.kzmmall.com/  
        - 자가수리 부품 AS 전용 사이트: https://www.kzmmall.com/category/as부품/110/
        - 주방 용품 부품 구매 사이트: https://kzmmall.com/product/주방-용품-부품/814/category/110/display/1/
        방문 전에는 제품 유무를 미리 확인해보시길 권장합니다.  

        ---

        ### 응답 가이드라인:

        1. **긍정적인 문의:**  
        - 감사 인사를 전하며, 고객의 긍정적인 경험을 기뻐하는 메시지를 전달합니다.  
        - 제품이나 서비스의 특징을 언급하고, 고객의 피드백이 중요하다는 점을 강조합니다.  
        - 예시: "소중한 의견 남겨주셔서 감사합니다. 저희 제품을 만족스럽게 사용해주셔서 기쁩니다. 앞으로도 최선을 다하겠습니다."  

        2. **중립적/개선 요청 문의:**  
        - 감사 인사 후 불편에 공감하며, 문제 해결을 위한 조치를 설명합니다.  
        - 추가 지원이 필요할 경우 고객 지원팀과의 연결을 안내합니다.  
        - 예시: "남겨주신 의견 감사합니다. 말씀하신 사항을 중요하게 생각하며, 더 나은 서비스를 위해 노력하겠습니다."  

        3. **부정적인 문의:**  
        - 진심으로 사과하고, 불편 사항에 대해 공감합니다.  
        - 문제 해결 방법을 안내하며, 추가 문의가 있을 경우 언제든지 지원을 안내합니다.  
        - 예시: "불편을 드려 죄송합니다. 문제를 신속히 해결할 수 있도록 최선을 다하겠습니다. 추가 문의가 있으시면 언제든지 말씀해 주세요."  

        4. **반복적인 문의:**  
        - 유사한 내용이라도 개인화된 문구를 사용해 기계적인 응답처럼 보이지 않도록 합니다.  

        ---

        ### 기본 톤과 스타일:
        - 항상 긍정적이고 정중하며, 고객의 입장에서 생각하는 태도를 보여줍니다.  
        - 간결하고 깔끔하게 응대하며, 고객이 이해하기 쉬운 언어를 사용합니다.  
        - 조사한 자료에 근거한 정확한 답변을 제공합니다.  
        """  

        self.model = SentenceTransformer('sentence-transformers/xlm-r-large-en-ko-nli-ststb')
        self.ocr_handler = OCRHandler()

        # Load data and prepare embeddings
        self.df = self.load_all_excel_files(data_folder)
        if '문의내용' not in self.df.columns or '답변내용' not in self.df.columns:
            raise ValueError("Excel file must contain '문의내용' and '답변내용' columns.")

        self.df['문의내용'] = self.df['문의내용'].fillna('').astype(str)
        self.embeddings = self.generate_embeddings(self.df['문의내용'])

        # Create and manage FAISS index
        index_file_path = os.path.join(data_folder, 'faiss.index')
        if os.path.exists(index_file_path):
            self.index = faiss.read_index(index_file_path)
            logger.info("Loaded FAISS index from disk.")
        else:
            self.index = self.create_faiss_index(self.embeddings)
            faiss.write_index(self.index, index_file_path)
            logger.info("FAISS index created and saved to disk.")

    def load_all_excel_files(self, folder_path):
        combined_df = pd.DataFrame()
        try:
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.xls') or file_name.endswith('.xlsx'):
                    file_path = os.path.join(folder_path, file_name)
                    read_engine = 'xlrd' if file_name.endswith('.xls') else 'openpyxl'
                    df = pd.read_excel(file_path, engine=read_engine)
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
            return combined_df
        except Exception as e:
            logger.error(f"Error reading Excel files: {e}")
            raise

    def generate_embeddings(self, texts):
        logger.info("Generating embeddings...")
        texts = [text for text in texts if text]  # Filter out any empty texts
        return self.model.encode(texts, batch_size=32, show_progress_bar=True)

    def create_faiss_index(self, embeddings):
        if embeddings.shape[0] == 0:
            logger.error("No embeddings provided to create index")
            raise ValueError("No embeddings to index.")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype='float32'))
        logger.info("FAISS index created.")
        return index

    def show_popup_non_blocking(self, message):
        def popup(message):
            message_box_styles = 0x00000040 | 0x00000000 | 0x00040000 | 0x00001000
            ctypes.windll.user32.MessageBoxW(0, message, "중요한 문의 내용입니다", message_box_styles)
        threading.Thread(target=popup, args=(message,), daemon=True).start()

    def find_similar_questions(self, question):
        logger.info("Finding similar questions based on '문의내용'...")
        try:
            question_embedding = np.array([self.model.encode(question)])
            distances, indices = self.index.search(question_embedding, 10)

            if indices.size == 0 or not len(indices[0]):
                logger.warning("No similar questions found.")
                return []

            similar_questions = []
            for idx in indices[0]:
                if 0 <= idx < len(self.df):
                    entry = self.df.iloc[idx]
                    similarity_score = 100 - distances[0][indices[0].tolist().index(idx)]
                    importance = self.determine_importance(similarity_score)

                    similar_questions.append({
                        'entry': entry,
                        'similarity_score': similarity_score,
                        'importance': importance
                    })
            return similar_questions
        except Exception as e:
            logger.error(f"Error in finding similar questions: {e}")
            return []

    def determine_importance(self, similarity_score):
        if similarity_score > 50:
            return 'High, 유사질문에 대한 모범답안 입니다. 답변을 참고하지만 수치와 숫자는 참고하지 말고 답변 방식과 형식을 참고하십시오.'
        elif similarity_score > 0:
            return 'Medium, 유사질문에 대한 답변입니다. 답변을 참고하지만 수치와 숫자는 참고하지 말고 답변 방식과 형식을 참고하십시오'
        else:
            return 'Low, 틀린 답변입니다. 답변의 수치와 숫자는 참고하지 말고 형식만을 참고하십시오.'

    def check_stock(self, master_code):
        if not master_code:
            return "정보를 확인할 수 없습니다."

        url = "https://api.e3pl.kr/ai/"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        payload = {"m": "stock", "c": master_code, "s": "kazmi"}

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                stock_data = response.json()
                if stock_data.get('success'):
                    results = stock_data.get('result', [])
                    if results and results[0].get('ea', 0) > 0:
                        return "현 시간 기준 현재 상품의 재고가 있습니다."
                    else:
                        return "현 시간 기준 현재 상품의 재고가 없습니다."
                else:
                    return "재고 데이터 검색이 성공적이지 않았습니다."
            return "응답이 JSON 형식이 아닙니다."

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception occurred: {e}")
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")

        return "정보를 확인할 수 없습니다."

    def check_restock(self, master_code):
        if not master_code:
            return "정보를 확인할 수 없습니다."

        # Assuming a similar endpoint for restock information; adjust parameters if different
        url = "https://api.e3pl.kr/ai/"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        payload = {"m": "restock", "c": master_code, "s": "kazmi"}

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                restock_data = response.json()
                if restock_data.get('success'):
                    results = restock_data.get('result', [])
                    if results and results[0].get('ea', 0) > 0:
                        return "해당 상품은 조만간 재입고 될 예정입니다."
                    else:
                        return "재입고 일정이 확정되지 않았습니다."
                else:
                    return "재입고 데이터 검색이 성공적이지 않았습니다."
            return "응답이 JSON 형식이 아닙니다."

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception occurred: {e}")
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")

        return "정보를 확인할 수 없습니다."

    def generate_answer(self, question, combined_summary, product_info, inquiry_data, product_name=None, comment_time=None, order_states=None, image_urls=None):
        logger.info("Generating answer for question...")
        try:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            category_info = self.classify_question(question)
            categories = self.extract_category_from_classification(category_info)

            if self.check_if_important_question(question):
                self.show_popup_non_blocking(question)

            if not isinstance(product_info, dict):
                logger.warning("Product info is not a dictionary.")
                return "죄송합니다, 제품 정보를 처리할 수 없습니다."

            master_code = inquiry_data.get('MasterCode', '')
            stock_status = self.check_stock(master_code)
            restock_status = self.check_restock(master_code)
            logger.info(f"Stock status: {stock_status}, Restock status: {restock_status}")

            similar_questions = self.find_similar_questions(question)
            similar_question_prompt = "".join([
                f"\n\nImportance: {sq.get('importance', 'Unknown')}\n"
                f"Similarity Score: {sq.get('similarity_score', 0)}\n"
                f"Q: {sq['entry'].get('문의내용', '')}\n"
                f"A: {sq['entry'].get('답변내용', '기존 데이터베이스에 답변이 없습니다.')}\n\n"
                for sq in similar_questions
            ]) or "죄송합니다. 데이터베이스에서 유사한 질문을 찾을 수 없습니다."

            specific_prompt = self.get_specific_prompt(categories)

            product_details = f"상품명: {product_name}\n" + "\n".join(
                [f"{key}: {value}" for key, value in product_info.items() if key != 'ProductName']
            )
            response_data_details = "\n".join([f"{key}: {value}" for key, value in inquiry_data.items()])
            print(f"재입고 상황{restock_status},\n\n\n상품명:{product_name},\n\n\n질문:{question},\n\n\n주문상태:{order_states},\n\n\n상품정보:{product_details},\n\n\nOCR 요약:{combined_summary},\n\n\n문의 상세내용:{response_data_details},\n\n\n재고유무:{stock_status},\n\n\n재입고유무:{restock_status},\n\n\n비슷한 질문의 답변:{similar_question_prompt}\n\n\n특정카테고리 프롬프트:{specific_prompt}\n\n\n")
            answer_prompt = f"""
            제공된 제품 정보, 유사 질문 답변 및 재고 정보를 바탕으로 다음 문의에 답변해 주세요. 제품명과 고객질문을 언급하지 말아주세요.
            제품명: {product_name}
            고객 질문과 주문상태: {question}
            주문 상태: {order_states}
            제품 정보: {product_details}
            이미지 분석 및 OCR 결론: {combined_summary}
            추가 문의 정보: {response_data_details}
            현 시간 기준 재고 상황: {stock_status}
            재입고 상황: {restock_status}
            {similar_question_prompt}
            {specific_prompt}
            """

            if comment_time:
                answer_prompt += f"\n\n질문 시간: {comment_time}, 현재 시간: {current_time}"

            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": answer_prompt}
                ]
            )

            answer = chat_completion.choices[0].message.content.strip()
            return answer

        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            return "죄송합니다, 답변을 생성할 수 없습니다."

    def revise_answer(self, user_input, original_answer):
        logger.info("Revising answer...")
        try:
            revision_prompt = f"""
            수정 전 답변: {original_answer}
            사용자 입력: {user_input}
            사용자의 입력을 최대한 반영하여 답변을 생성해주세요.
            사용자 입력이 작은 따움표('')안에 입력한 경우 해당 내용을 받드시 그대로 답변으로 사용/출력 하십시오, 절대 변형하거나 재생성 혹은 추가 답변을 하지 마십시오.
            """

            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": revision_prompt}
                ]
            )

            if chat_completion and chat_completion.choices:
                return chat_completion.choices[0].message.content
            else:
                logger.error("Chat completion returned no choices for revision")
                return original_answer
        except Exception as e:
            logger.error(f"답변 수정 오류: {e}")
            return original_answer

    def check_if_important_question(self, question):
        try:
            check_prompt = f"""
            다음 고객 문의가 아래 카테고리와 관련이 있는지 판단해 주세요.
            - 택배 관련 문의 
            - 방문 수령 문의
            - 송장 관련 문의
            - 묶음 배송 관련 문의
            - 주소 변경
            - 고객 이름 변경
            - 받는 주소 변경
            - 주문 취소
            - 입고 관련
            - 재입고 관련
            - 출시 예정
            - 상품 간의 차이
            - 사용법/호환성
            - 계산서 관련
            - 반품
            - 교환

            고객 문의: {question}

            반환 예시 (해당되는 경우): "해당합니다."
            반환 예시 (해당되지 않는 경우): "해당하지 않습니다."
            """

            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": check_prompt}
                ]
            )

            if chat_completion and chat_completion.choices:
                gpt_response = chat_completion.choices[0].message.content.strip()
                return "해당합니다" in gpt_response
            else:
                logger.error("Chat completion returned no choices for importance check")
                return False
        except Exception as e:
            logger.error(f"중요한 질문 판별 오류: {e}")
            return False

    def classify_question(self, question):
        try:
            classification_prompt = f"""
            다음 고객 문의를 관련된 모든 카테고리로 분류해 주세요. 여러 카테고리가 관련된 경우 모두 표시해 주세요.
            카테고리 목록: [제품 리뷰, 재고/입고/배송, AS (수리, 반품, 교환), 사용법/호환성, 제품 문의, 할인 문의, 기술 지원, 기타 질문]
            소비자 문의가 아래 상황 중 하나에 해당하면 기타 질문입니다.
            - 택배 관련 문의 
            - 방문 수령 문의
            - 송장 관련 문의
            - 묶음 배송 관련 문의
            - 주소 변경
            - 고객 이름 변경
            - 받는 주소 변경
            - 주문 취소
            - 입고 관련
            - 재입고 관련
            - 출시 예정
            - 상품 간의 차이
            - 사용법/호환성
            - 계산서 관련
            - 반품
            - 교한

            고객 문의: {question}
            """
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": classification_prompt}
                ]
            )

            if chat_completion and chat_completion.choices:
                category_info = chat_completion.choices[0].message.content
                logger.info(f"Category classification result: {category_info}")
                return category_info
            else:
                logger.error("Chat completion returned no choices for classification")
                return "기타 질문"
        except Exception as e:
            logger.error(f"질문 분류 오류: {e}")
            return "기타 질문"

    def extract_category_from_classification(self, category_info):
        categories = [
            "제품 리뷰", "재고/입고/배송", "AS (수리, 반품, 교환)",
            "사용법/호환성", "제품 문의", "할인 문의", "기술 지원", "기타 질문"
        ]
        detected_categories = [category for category in categories if category in category_info]
        return detected_categories

    def get_specific_prompt(self, categories):
        category_prompts = {
            "제품 리뷰": """
            고객님께서 작성하신 제품 리뷰에 대해 가능한 친절하고 공손하게 응답해 주세요. 
            피드백을 바탕으로 개선할 점이 있다면 공감하며 답변합니다.
            """,

            "재고/입고/배송": """
            고객님께서 재고, 입고, 또는 배송 관련 문의를 주셨습니다. 
            정확한 재고 상태, 입고 예정일, 배송 가능 여부 및 소요 시간에 대해 안내해 주세요.
            - 단종이 아니지만 재입고 일정이 확정되지 않아 당분간 구매가 어렵습니다.
            - 오후 1시 이전 주문은 당일 출고, 이후 주문은 익일 출고됩니다.
            """,

            "AS (수리, 반품, 교환)": """
            고객님께서 AS, 수리, 반품 또는 교환 절차에 대해 문의하셨습니다. 
            고객님의 불편을 최소화할 수 있도록 명확하고 자세한 절차를 안내해 주세요.
            - AS에 괄호나 부가 설명을 넣지 않습니다.
            - 인터넷 주소를 중복으로 작성하지 않습니다.
            - 전화번호는 안내하지 않습니다.
            - 금액 표기는 하지 않습니다.
            - 교환/반품 문의 시 AS 관련 안내는 하지 않습니다.
            - "구매하신 쇼핑몰의 반품교환기능을 이용하여 접수해 주세요."라고 안내합니다.
            - 진행된 교환/반품 건에 대해 재접수를 권유하지 않습니다.
            - 카즈미 텐트의 폴대 서비스는 가능합니다.

            **자가수리 권장:**
            자가수리는 시간과 비용을 절감할 수 있으므로 추천합니다. 
            """,

            "사용법/호환성": """
            고객님께서 제품의 사용법 또는 다른 제품과의 호환성에 대해 문의하셨습니다. 
            사용 방법과 호환 가능 여부를 상세하게 안내해 주세요.
            """,

            "제품 문의": """
            사이즈나 용량에 대한 정보가 불분명한 경우 직접 확인하도록 안내합니다.
            - "각 제품의 형상과 두께, 재는 방법에 따라 차이가 크므로, 본사 직영매장이나 가까운 취급점에서 직접 확인해 주세요."라고 답변합니다.
            - "선물세트"라는 단어가 포함된 경우 매장 안내나 AS 안내를 제공하지 않습니다.
            - 제품 설명은 긍정적이고 좋게 전달합니다.
            - 제조 공정상 ±2%의 오차가 있을 수 있습니다.
            - 단종이 아니지만 재입고 일정이 확정되지 않아 구매가 어려울 수 있습니다.

            **식기세척기 사용 가능 제품:**
            에센셜 커틀러리세트, K24T3K02식기세트 22P, K22T3K07, 캠핑 식기세트 17P, K22T3K06, 
            캠핑 식기세트 15P, K22T3K05, 캠핑 식기세트 25P, K21T3K11, 웨스턴 커틀러리 세트, K22T3K01, 
            트라이 커틀러리 세트, K9T3K004, 쉐프 키친툴세트, K9T3K011, 더블 머그컵 6Pset, K4T3K004, 
            에그 텀블러 2P, K9T3K010, 프리미엄 STS 푸드 플레이트, K20T3K003, 프리미엄 코펠세트 XL, K8T3K003, 
            프리미엄 코펠세트 L, K8T3K002, 프리미엄 STS 패밀리 캠핑 식기세트, K20T3K002, 
            프리미엄 STS 식기세트 커플, K20T3K001

            **식기세척기 사용 불가능 제품:**
            이그니스 디자인 팟 그리들 X 얼, K24T3K05, 이그니스 디자인 그리들(얼), K23T3G03, 
            필드 크레프트 시에라컵 2Pset, K23T3K05GR / K23T3K05BK, 와일드 필드 캠핑컵 8P, K23T3K03, 
            NEW 블랙 머그 5Pset, K21T3K03, 웨이브 콜드컵 2Pset, K8T3K007RD, 필드 650 텀블러, K23T3K06, 
            트윈 우드 커틀러리세트, K21T3K10, 스텐레스 캠핑 주전자 0.8L, K21T3K08

            **주의 사항:**
            코팅 처리된 제품은 식기세척기 사용 시 손상될 수 있으며, 이는 무상 교환 대상이 아닙니다. 
            스테인리스 재질 제품은 식기세척기 사용이 가능하지만 나무, 플라스틱 등의 재질이 혼합된 경우 변형이 발생할 수 있습니다.
            """,

            "할인 문의": """
            고객님께서 할인 관련 문의를 주셨습니다. 현재 진행 중인 할인 정보와 적용 조건을 안내해 주세요.
            """,

            "기술 지원": """
            고객님께서 기술 지원을 요청하셨습니다. 신속하고 정확한 기술 지원을 제공해 주세요.
            """,

            "기타 질문": """
            고객님께서 기타 문의를 주셨습니다. 질문 의도를 정확히 파악하고 상황에 맞게 답변해 주세요.
            - "모른다" 또는 "확인이 필요하다"라는 표현은 사용하지 않습니다.
            - 추가 정보를 요구하지 않습니다.
            - 단종이 아니지만 재입고 일정이 확정되지 않아 당분간 구매가 어렵습니다.
            - 오후 1시 이전 주문은 당일 출고, 이후 주문은 익일 출고됩니다.
            - 교환/반품 관련 문의 시 AS 안내는 제공하지 않습니다.
            """
        }
        prompt = "\n".join([category_prompts[cat] for cat in categories])
        return prompt