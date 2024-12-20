import os
import time
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.alert import Alert
from selenium.common.exceptions import (NoSuchElementException, TimeoutException, 
                                        NoSuchWindowException, WebDriverException, 
                                        UnexpectedAlertPresentException, NoAlertPresentException)
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Access credentials from environment
mall_ID = os.getenv("MALL_ID")
mall_Password = os.getenv("MALL_PASSWORD")
apparel_ID = os.getenv("APPAREL_ID")
apparel_Password = os.getenv("APPAREL_PASSWORD")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Debugging: Check environment variables
print("Mall ID:", mall_ID)
print("Mall Password:", mall_Password)
print("Apparel ID:", apparel_ID)
print("Apparel Password:", apparel_Password)
print("OpenAI API Key:", openai_api_key)

# Set up OpenAI API key
client = OpenAI(api_key=openai_api_key)

# URL configurations
login_url = "https://eclogin.cafe24.com/Shop/"
mall_review_url = "https://kzmmall.cafe24.com/admin/php/shop1/b/board_admin_bulletin_l.php?start_date=2024-10-02&end_date=2024-11-01&period=30&sel_board_no=4&sel_spam_view=not&search_channel=&list_limit=100&search=subject&search_key=&is_reply=&is_comment=&real_filename=&mem_type=&report_status=&board_category=&no_member_article=F&navi_hide=&searchSort="
apparel_review_url = "https://kzmapparel.cafe24.com/admin/php/shop1/b/board_admin_bulletin_l.php?start_date=2024-10-02&end_date=2024-11-01&period=30&sel_board_no=4&sel_spam_view=not&search_channel=&list_limit=100&search=subject&search_key=&is_reply=&is_comment=&real_filename=&mem_type=&report_status=&board_category=&no_member_article=F&navi_hide=&searchSort="

def generate_answer(subject, body):
    """Use GPT-4o to generate an answer combining subject and body."""
    system_prompt = """
    당신은 쇼핑몰 고객 문의에 응답하는 친절한 상담원입니다. 한국어 문법과 띄어쓰기를 정확히 지키십시오.
    ### 응답 가이드라인:
    이모지를 쓰지 마십시오. 지원되지 않습니다. TEXT로만 구성하십시오.

    1. **긍정적인 리뷰:**
    - 감사 인사를 전하며, 고객의 긍정적인 경험을 기뻐하는 메시지를 전달합니다.
    - 제품이나 서비스의 특징을 언급하고, 고객의 피드백이 중요하다는 점을 강조합니다.
    - 예시: "소중한 의견 남겨주셔서 감사합니다. 저희 제품을 만족스럽게 사용해주셔서 기쁩니다. 앞으로도 최선을 다하겠습니다."

    2. **중립적/개선 요청 리뷰:**
    - 감사 인사 후 불편에 공감하며, 문제 해결을 위한 조치를 설명합니다.
    - 추가 지원이 필요할 경우 고객 지원팀과의 연결을 안내합니다.
    - 예시: "남겨주신 의견 감사합니다. 말씀하신 사항을 중요하게 생각하며, 더 나은 서비스를 위해 노력하겠습니다."

    3. **부정적인 리뷰:**
    - 진심으로 사과하고, 불편 사항에 대해 공감합니다.
    - 문제 해결 방법을 안내하며, 추가 문의가 있을 경우 언제든지 지원을 안내합니다.
    - 예시: "불편을 드려 죄송합니다. 문제를 신속히 해결할 수 있도록 최선을 다하겠습니다. 추가 문의가 있으시면 언제든지 말씀해 주세요."

    4. **반복적인 리뷰:**
    - 유사한 내용이라도 개인화된 문구를 사용해 기계적인 응답처럼 보이지 않도록 합니다.

    ### 기본 톤과 스타일:
    - 항상 긍정적이고 정중하며, 고객의 입장에서 생각하는 태도를 보여줍니다.
    - 간결하고 깔끔하게 응대하며, 고객이 이해하기 쉬운 언어를 사용합니다.
    - 조사한 자료에 근거한 정확한 답변을 제공합니다.
    """
    user_prompt = f"고객의 리뷰에 답변하세요. 제목: {subject}, 내용: {body}"

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = chat_completion.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "죄송합니다. 현재 답변을 생성할 수 없습니다."

def login(driver, user_id, password):
    """Perform login with the given credentials."""
    print("Logging in...")
    driver.get(login_url)
    try:
        id_field = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "mall_id")))
        id_field.clear()
        id_field.send_keys(user_id)

        password_field = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "userpasswd")))
        password_field.clear()
        password_field.send_keys(password)

        login_button = driver.find_element(By.XPATH, "//button[contains(text(), '로그인')]")
        login_button.click()
        WebDriverWait(driver, 10).until(EC.title_contains("Dashboard"))  # Assume dashboard loads on successful login
        print("Login successful.")
    except (NoSuchElementException, TimeoutException, WebDriverException) as e:
        print(f"Error during login: {e}")
    finally:
        print("Login attempt finished.")

