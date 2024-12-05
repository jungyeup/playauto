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

        **소모품 구매문의:**
        "소모품 구매는 자사몰 as접수를 통해 구매 가능여부 확인 가능 합니다. https://support.kzmoutdoor.com/ 번거로우시겠지만 AS 접수로 구매 부탁드립니다."라고 안내합니다.
        
        
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
