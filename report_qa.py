import os
import pandas as pd
from openai import OpenAI
from docx import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI with your API key
client = OpenAI(api_key=OPENAI_API_KEY)

def fetch_data_from_excel(file_path):
    """Read Excel data and return the DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file {file_path} does not exist.")

    return pd.read_excel(file_path)

def generate_korean_analysis(draft_answer, final_answer):
    """Use GPT-4o to generate analysis of the differences."""
    prompt = (
        "아래의 초기 답변과 최종 답변을 비교하고 차이점을 분석하여 간략하게 설명해 주세요:\n"
        f"초기 답변: {draft_answer}\n"
        f"최종 답변: {final_answer}\n"
    )

    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "당신은 전문가적인 보고서를 작성하는 임무를 맡았습니다."},
            {"role": "user", "content": prompt}
        ]
    )

    analysis = chat_completion.choices[0].message.content.strip()
    
    return analysis


    FILE_PATH = 'data/questions_answers.xlsx'
    OUTPUT_FILE = '제품별_문의_종합_보고서.docx'

    main(FILE_PATH, OUTPUT_FILE)
