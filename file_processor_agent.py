# file_processor_agent.py - Intelligent File Processing Agent
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage
from config_manager import llm_manager, app_config
import os

# Import data models directly
from data_models import EnhancedTransaction, LLMAnalysisResult


class FileProcessorAgent:
    """Intelligent file processor for financial documents"""
    
    def __init__(self):
        self.llm = llm_manager.get_client()
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.txt']
    
    def process_uploaded_file(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], LLMAnalysisResult]:
        """Process a single uploaded file with optional masking preview"""
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Step 1: Basic file reading
            if file_extension in ['.csv', '.txt']:
                df = self._read_csv_with_detection(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == '.pdf':
                df = self._read_pdf_with_ocr(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            if df is None:
                return None, LLMAnalysisResult(
                    success=False,
                    error_message="Could not read file content"
                )
            
            # Step 2: Intelligent column detection
            if self.llm:
                processed_df, analysis_result = self._llm_enhanced_column_detection(df)
            else:
                processed_df, analysis_result = self._rule_based_column_detection(df)
            
            # Step 3: Data validation and cleaning
            if processed_df is not None:
                processed_df = self._clean_and_validate_data(processed_df)
                return processed_df, analysis_result
            
            return None, analysis_result
        
        except Exception as e:
            return None, LLMAnalysisResult(
                success=False,
                error_message=f"File processing failed: {str(e)}"
            )

    def _read_file_with_format_detection(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Read file with intelligent format detection"""
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        try:
            # Excel files
            if file_extension in ['xlsx', 'xls']:
                return pd.read_excel(uploaded_file)
            
            # CSV and text files - try different separators and encodings
            elif file_extension in ['csv', 'txt']:
                return self._read_csv_with_detection(uploaded_file)
            
            # Try to read as CSV anyway
            else:
                return self._read_csv_with_detection(uploaded_file)
        
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return None

    def _read_csv_with_detection(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Try multiple CSV reading strategies"""
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        delimiters = [',', '\t', ';', '|', ' ']
        
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    return pd.read_csv(
                        uploaded_file, 
                        sep=delimiter, 
                        engine='python',
                        on_bad_lines='skip',
                        encoding=encoding
                    )
                except:
                    continue
        
        # Fallback: Use first 10 lines to guess delimiter
        try:
            sample = pd.read_csv(uploaded_file, nrows=10, on_bad_lines='skip')
            delimiter = '\t' if '\t' in sample.columns[0] else ','
            return pd.read_csv(uploaded_file, sep=delimiter, on_bad_lines='skip')
        except Exception as e:
            st.error(f"CSV fallback failed: {e}")
            return None

    def _read_pdf_with_ocr(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Read PDF using OCR (for scanned bank statements)"""
        try:
            from pdfplumber import open as pdf_open
            from PIL import Image
            import pytesseract
            
            with pdf_open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        image = page.to_image(resolution=200).original
                        ocr_text = pytesseract.image_to_string(image)
                        text += ocr_text + "\n"
            
            # Save temporary text file for parsing
            temp_path = "temp_ocr_output.csv"
            with open(temp_path, "w") as f:
                f.write(text)
            
            # Now try reading as CSV
            df = pd.read_csv(temp_path)
            return df
            
        except Exception as e:
            st.error(f"PDF OCR failed: {e}")
            return None

    def _llm_enhanced_column_detection(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, LLMAnalysisResult]:
        """Use LLM to detect column mappings intelligently"""
        try:
            # Get sample data
            sample_data = df.head(5).to_dict(orient='records')
            
            prompt = f"""
            Analyze this bank statement data and identify which columns correspond to:
            - Date
            - Description
            - Amount
            - Optional: Debit/Credit indicator
            - Optional: Balance
            
            Sample data:
            {json.dumps(sample_data, indent=2)}
            
            Return JSON mapping of column names like:
            {{
                "date_col": "column_name_for_date",
                "description_col": "column_name_for_description",
                "amount_col": "column_name_for_amount",
                "debit_col": "optional_debit_column",
                "balance_col": "optional_balance_column"
            }}
            """
            
            messages = [
                SystemMessage(content="You are a financial data expert who helps identify bank statement columns"),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            raw_response = response.content
            
            # Extract JSON from response
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                mapping = json.loads(match.group())
            else:
                raise ValueError("No valid JSON found in LLM response")
            
            # Validate required fields
            required_fields = ['date_col', 'description_col', 'amount_col']
            if not all(field in mapping for field in required_fields):
                raise ValueError("Missing required fields in LLM response")
            
            # Reorder columns
            result_df = pd.DataFrame({
                'date': df[mapping['date_col']],
                'description': df[mapping['description_col']],
                'amount': df[mapping['amount_col']]
            })
            
            if 'balance_col' in mapping:
                result_df['balance'] = df[mapping['balance_col']]
            
            return result_df, LLMAnalysisResult(
                success=True,
                confidence='high',
                reasoning="LLM-assisted column detection"
            )
        
        except Exception as e:
            return self._rule_based_column_detection(df)

    def _rule_based_column_detection(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, LLMAnalysisResult]:
        """Fallback to rule-based column detection"""
        date_cols = [col for col in df.columns if any(kw in str(col).lower() for kw in ['date', 'posted', 'trans'])]
        description_cols = [col for col in df.columns if any(kw in str(col).lower() for kw in ['desc', 'description', 'memo', 'note'])]
        amount_cols = [col for col in df.columns if any(kw in str(col).lower() for kw in ['amount', 'value', 'total'])]
        
        if not all([date_cols, description_cols, amount_cols]):
            return None, LLMAnalysisResult(
                success=False,
                error_message="Missing required columns (date, description, amount)"
            )
        
        result_df = pd.DataFrame({
            'date': df[date_cols[0]],
            'description': df[description_cols[0]],
            'amount': df[amount_cols[0]]
        })
        
        return result_df, LLMAnalysisResult(
            success=True,
            confidence='medium',
            reasoning="Rule-based column detection"
        )

    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate transaction data"""
        # Ensure correct data types
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Clean dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Remove empty rows
        df = df.dropna(subset=['date', 'description', 'amount'])
        
        # Handle negative values
        df['amount'] = df['amount'].apply(lambda x: float(x) * (-1 if x < 0 else 1))
        
        return df

    def convert_to_enhanced_transactions(self, df: pd.DataFrame) -> List[EnhancedTransaction]:
        """Convert DataFrame to list of EnhancedTransaction objects"""
        transactions = []
        for _, row in df.iterrows():
            transaction = EnhancedTransaction(
                date=row['date'],
                description=row['description'],
                amount=float(row['amount']),
                category="other",
                subcategory="uncategorized"
            )
            transactions.append(transaction)
        return transactions

    def get_file_analysis_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary of file analysis"""
        if df is None or df.empty:
            return {'status': 'failed', 'message': 'No data processed'}
        
        total_transactions = len(df)
        date_range = f"{df['date'].min()} to {df['date'].max()}"
        
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        net_flow = total_income - total_expenses
        
        return {
            'status': 'success',
            'total_transactions': total_transactions,
            'date_range': date_range,
            'financial_summary': {
                'total_income': total_income,
                'total_expenses': total_expenses,
                'net_flow': net_flow
            },
            'data_quality': {
                'missing_dates': df['date'].isna().sum(),
                'missing_descriptions': df['description'].isna().sum(),
                'missing_amounts': df['amount'].isna().sum(),
                'zero_amounts': (df['amount'] == 0).sum()
            }
        }