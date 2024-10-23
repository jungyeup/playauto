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

def create_docx_report(df, output_path):
    """Create a DOCX report with all inquiries organized by product."""
    document = Document()
    document.add_heading('제품별 문의 종합 보고서', level=1)

    for _, row in df.iterrows():
        document.add_heading(f"제품명: {row['상품명']}", level=2)
        document.add_paragraph(f"문의내용: {row['문의내용']}")
        document.add_paragraph(f"답변초안: {row['답변초안']}")
        document.add_paragraph(f"답변내용: {row['답변내용']}")

        # Generate analysis for the differences
        analysis = generate_korean_analysis(row['답변초안'], row['답변내용'])
        document.add_paragraph(f"분석 내용: {analysis}")

    # Save the document
    document.save(output_path)
    print(f"Report saved to {output_path}")

def main(file_path, output_file):
    try:
        # Load Excel data
        df = fetch_data_from_excel(file_path)

        # Create a single DOCX report with all data
        create_docx_report(df, output_file)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage
    FILE_PATH = 'data/questions_answers.xlsx'
    OUTPUT_FILE = '제품별_문의_종합_보고서.docx'

    main(FILE_PATH, OUTPUT_FILE)