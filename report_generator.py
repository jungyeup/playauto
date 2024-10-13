import os
import logging
from pandas import ExcelWriter
import pandas as pd
from datetime import datetime

# Ensure logging is configured
logging.basicConfig(level=logging.INFO)

class ReportGenerator:
    def __init__(self, folder_name='history'):
        self.folder_name = folder_name
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
            
    def generate_xlsx_report(self, data, file_path):
        """
        Generate an XLSX report for the given data and save it to file_path.
        """
        df_new = pd.DataFrame(data)
        
        # Try to load existing data if the file exists
        if os.path.exists(file_path):
            df_existing = pd.read_excel(file_path)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        
        df.to_excel(file_path, index=False)
        logging.info(f"Report saved as .xlsx at {file_path}")

    def generate_reports(self, data):
        """
        Generate daily reports in XLSX format.
        """
        current_date = datetime.now().strftime('%Y-%m-%d')
        xlsx_file_path = os.path.join(self.folder_name, f'Daily_Report_{current_date}.xlsx')

        logging.info("Generating XLSX report...")
        self.generate_xlsx_report(data, xlsx_file_path)