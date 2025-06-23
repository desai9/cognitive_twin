# data_masker.py - PII Detection and Masking Module (Adapted for Indian Context)
import re
from typing import Dict, Any, Tuple
import pandas as pd

class PiiMasker:
    """
    Class for detecting and masking Personally Identifiable Information (PII)
    within text and Pandas DataFrames, with patterns adapted for Indian context.
    """
    
    def __init__(self):
        # Define comprehensive regex patterns for various PII types relevant to India.
        # These patterns are designed to be case-insensitive for broader detection.
        self.patterns = {
            # Indian Mobile Phone Numbers: 10 digits, optionally prefixed with +91 or 0
            "phone": r"\b(?:(?:\+91|0)?[6789]\d{9})\b",
            # Email addresses (universal)
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            # Basic address pattern (challenging to make comprehensive globally/locally)
            # This is a generic pattern and might require further refinement for specific Indian address formats.
            "address": r"\b\d+\s+[\w\s,\.#]+\b(?=\s*(st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|pl|place|ct|court|cir|circle|pin code)\b)",
            # Indian Bank Account Numbers: typically 8 to 18 digits (varies by bank)
            "bank_account": r"\b\d{8,18}\b",
            # Credit/Debit card numbers (universal 16-digit format)
            "credit_card": r"\b(?:(?:\d{4}[- ]?){3}\d{4}|\d{16})\b",
            # Indian PAN (Permanent Account Number): 5 Alphabets, 4 Digits, 1 Alphabet
            "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
            # Indian Aadhaar Number: 12 digits, often grouped as 4-4-4
            "aadhaar": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            # Indian Financial System Code (IFSC): 11 characters (first 4 alphabets, 5th is 0, next 6 are digits)
            "ifsc_code": r"\b[A-Z]{4}0\d{6}\b"
        }
        
        # Keywords that, when present, suggest the context might contain sensitive data.
        # Used by 'contains_sensitive_data' for a quick preliminary check, adapted for Indian terms.
        self.sensitive_keywords = [
            'pan', 'aadhaar', 'ifsc', 'upi', 'kyc', 'bank account', 'card number', 
            'credit card', 'debit card', 'account no', 'mobile no', 'phone no'
        ]

    def mask_text(self, text: str) -> str:
        """
        Masks all detected PII patterns within a given string with "[REDACTED]".
        
        Args:
            text (str): The input string to be masked.
            
        Returns:
            str: The text with all detected PII replaced by "[REDACTED]".
        """
        if not isinstance(text, str) or not text.strip():
            return text # Return original text if it's not a non-empty string
            
        masked_text = text
        
        # Iterate through each defined PII pattern and apply masking using regex substitution.
        # re.IGNORECASE ensures case-insensitive matching.
        for label, pattern in self.patterns.items():
            masked_text = re.sub(pattern, "[REDACTED]", masked_text, flags=re.IGNORECASE)
            
        return masked_text

    def mask_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes a Pandas DataFrame to create two versions:
        1. An original copy (for reference, though PII should be handled carefully).
        2. A masked copy where PII in string/object columns is redacted.
        
        This allows for side-by-side comparison and ensures sensitive data
        is not passed to downstream LLM components.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing transaction data.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the original DataFrame copy
                                               and the masked DataFrame copy. Returns empty
                                               DataFrames if the input is None or empty.
        """
        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        df_original = df.copy() # Create a copy of the original DataFrame
        df_masked = df.copy()   # Create another copy for masking operations
        
        # Iterate over columns that typically contain string data (object dtype)
        # and apply the mask_text function to each cell.
        for col in df.select_dtypes(include='object').columns:
            df_masked[col] = df[col].apply(lambda x: 
                self.mask_text(str(x)) if isinstance(x, str) else x) # Ensure x is a string before masking
                
        return df_original, df_masked

    def contains_sensitive_data(self, text: str) -> bool:
        """
        Checks if a given text string contains potential PII by looking for
        defined sensitive keywords (case-insensitive).
        
        This is a quicker, less precise check compared to full regex matching.
        
        Args:
            text (str): The input string to check.
            
        Returns:
            bool: True if any sensitive keyword is found, False otherwise.
        """
        if not isinstance(text, str):
            return False # Cannot check non-string data
            
        text_lower = text.lower() # Convert to lowercase once for efficiency
        return any(keyword in text_lower for keyword in self.sensitive_keywords)

    def get_pii_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Generates a summary count of detected PII types across all string/object
        columns in the DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            
        Returns:
            Dict[str, int]: A dictionary where keys are PII labels (e.g., "pan", "aadhaar")
                            and values are the number of occurrences detected.
        """
        summary = {}
        
        # Iterate through all string/object columns
        for col in df.select_dtypes(include='object').columns:
            # Iterate through each value in the column
            for value in df[col].dropna(): # Drop NA values to avoid errors with non-strings
                if isinstance(value, str):
                    # Check each PII pattern against the string value
                    for label, pattern in self.patterns.items():
                        if re.search(pattern, value, re.IGNORECASE):
                            # Increment count for the detected PII type
                            summary[label] = summary.get(label, 0) + 1
                
        return summary

