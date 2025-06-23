# trend_analyzer_agent.py - Financial Trend Detection Agent
import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any, Optional
import streamlit as st # Used for st.warning/info messages

from langchain.schema import HumanMessage, SystemMessage
from config_manager import llm_manager
from data_models import EnhancedTransaction, FinancialHealthMetrics # Import FinancialHealthMetrics

# Import HealthCalculatorEngine to reuse its metric calculation logic
from health_analyzer_agent import HealthCalculatorEngine


class HealthTrendAnalyzer:
    """
    Class for analyzing financial trends over time, leveraging
    HealthCalculatorEngine for consistent metric calculation across periods
    and LLMs for qualitative trend insights.
    """
    
    def __init__(self):
        self.llm = llm_manager.get_client()
        # Instantiate HealthCalculatorEngine to calculate consistent health metrics per period
        self.health_calculator = HealthCalculatorEngine()

    def analyze_trends(self, transactions: List[EnhancedTransaction]) -> Dict[str, Any]:
        """
        Analyzes financial trends across multiple periods (e.g., monthly).
        Calculates financial health metrics for each period and uses an LLM
        to provide a high-level assessment of trajectory, drivers, and recommendations.

        Args:
            transactions (List[EnhancedTransaction]): A list of categorized
                                                    EnhancedTransaction objects.

        Returns:
            Dict[str, Any]: A dictionary containing detailed period metrics and
                            an LLM-generated trend analysis.
        """
        try:
            if not transactions:
                return {
                    'status': 'no_data',
                    'error': 'No transactions provided for trend analysis.'
                }

            # Convert transactions to DataFrame for easier time-series operations
            df = pd.DataFrame([tx.to_dict() for tx in transactions])
            
            # Ensure 'date' column is in datetime format and sort by date
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date']).sort_values('date').copy() # Drop rows with invalid dates

            if df.empty:
                 return {
                    'status': 'no_valid_data',
                    'error': 'No valid transactions after date parsing for trend analysis.'
                }

            # Group transactions into monthly buckets
            # Use to_period('M') for robust monthly grouping
            df['month_key'] = df['date'].dt.to_period('M') 
            
            # Convert back to list of EnhancedTransactions for each month to pass to health_calculator
            monthly_transaction_lists = {}
            # Define the columns that an EnhancedTransaction expects
            enhanced_transaction_cols = ['date', 'description', 'amount', 'category', 'subcategory', 'spending_type', 'confidence', 'reasoning', 'balance', 'reference']

            for month_key, group in df.groupby('month_key'):
                # When converting back to EnhancedTransaction, select only relevant columns
                # This avoids passing 'month_key' or other temporary DataFrame columns
                filtered_transactions_data = group[enhanced_transaction_cols].to_dict(orient='records')
                monthly_transaction_lists[str(month_key)] = [
                    EnhancedTransaction.from_dict(row_data) for row_data in filtered_transactions_data
                ]

            period_metrics = []
            for month, tx_list_for_month in sorted(monthly_transaction_lists.items()):
                # Calculate full FinancialHealthMetrics for each month
                monthly_health_obj = self.health_calculator.calculate_financial_health(tx_list_for_month)
                
                period_metrics.append({
                    'period': month,
                    'metrics': monthly_health_obj.to_dict(), # Store the full metrics dictionary
                    'transaction_count': len(tx_list_for_month)
                })
            
            # Basic trend analysis using overall scores
            # Filter for periods where overall_score is available and not 0 (to avoid division by zero)
            valid_trend_scores = [
                float(p['metrics']['overall_score']) 
                for p in period_metrics 
                if p['metrics'].get('overall_score') is not None and p['metrics'].get('overall_score') != 0
            ]

            if len(valid_trend_scores) < 2:
                # If less than 2 valid data points, cannot determine a meaningful trend
                return {
                    'status': 'insufficient_data',
                    'period_metrics': period_metrics,
                    'trend_analysis': {
                        'trajectory': 'stable',
                        'change_percentage': 0.0,
                        'confidence': 'low',
                        'analysis': 'Not enough data points across periods to detect meaningful trends.',
                        'trend_drivers': [],
                        'warning_signs': [],
                        'positive_developments': [],
                        'recommendations': []
                    }
                }
            
            # Calculate simple numerical trend based on start and end scores
            start_score = valid_trend_scores[0]
            end_score = valid_trend_scores[-1]
            
            trend_direction_simple = "stable"
            change_percentage_simple = 0.0
            if start_score != 0: # Avoid division by zero
                change_percentage_simple = ((end_score - start_score) / start_score) * 100
                if end_score > start_score:
                    trend_direction_simple = "improving"
                elif end_score < start_score:
                    trend_direction_simple = "declining"
            
            # LLM-powered trend insights for richer, qualitative analysis
            if self.llm and llm_manager.is_available():
                st.info("Generating LLM-powered trend insights...")
                trend_insights = self._generate_trend_insights(period_metrics, trend_direction_simple, change_percentage_simple)
            else:
                st.warning("LLM not available for enhanced trend analysis. Providing basic trend assessment.")
                trend_insights = {
                    'trajectory_assessment': f"Basic trend analysis indicates the trajectory is {trend_direction_simple}.",
                    'trajectory': trend_direction_simple,
                    'change_percentage': round(change_percentage_simple, 1),
                    'confidence': 'medium',
                    'analysis': f"Overall trend: {trend_direction_simple}. Health score changed by {change_percentage_simple:+.1f}% from {period_metrics[0]['period']} to {period_metrics[-1]['period']}.",
                    'trend_drivers': [],
                    'warning_signs': [],
                    'positive_developments': [],
                    'recommendations': []
                }
            
            return {
                'status': 'success',
                'period_metrics': period_metrics,
                'trend_analysis': trend_insights
            }
        
        except Exception as e:
            # Catch and report any broad errors during trend analysis
            st.error(f"Error during overall trend analysis: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'period_metrics': [],
                'trend_analysis': {
                    'trajectory': 'unknown',
                    'change_percentage': 0.0,
                    'confidence': 'low',
                    'analysis': f"Trend analysis system error: {str(e)}",
                    'trend_drivers': [], 'warning_signs': [], 'positive_developments': [], 'recommendations': []
                }
            }

    # Removed _calculate_monthly_health as its functionality is now covered by HealthCalculatorEngine

    def _generate_trend_insights(self, period_metrics: List[Dict], simple_trajectory: str, simple_change_percentage: float) -> Dict:
        """
        Generates detailed trend insights using an LLM based on the calculated
        period-by-period financial health metrics.
        
        Args:
            period_metrics (List[Dict]): A list of dictionaries, where each dict
                                         contains 'period', 'metrics' (FinancialHealthMetrics.to_dict()),
                                         and 'transaction_count'.
            simple_trajectory (str): The basic trajectory (improving/stable/declining)
                                     from initial calculation.
            simple_change_percentage (float): The basic percentage change in score.

        Returns:
            Dict: A dictionary containing LLM-generated insights in a structured format.
        """
        if not self.llm or not llm_manager.is_available():
            st.warning("LLM not available for generating detailed trend insights.")
            return {
                'trajectory_assessment': f"Basic trend analysis indicates the trajectory is {simple_trajectory}.",
                'trajectory': simple_trajectory,
                'change_percentage': round(simple_change_percentage, 1),
                'confidence': 'medium',
                'analysis': f"Overall trend: {simple_trajectory}. Health score changed by {simple_change_percentage:+.1f}% from {period_metrics[0]['period']} to {period_metrics[-1]['period']} (if available).",
                'trend_drivers': [],
                'warning_signs': [],
                'positive_developments': [],
                'recommendations': []
            }

        try:
            # Prepare data for LLM, focusing on key metrics for each period
            llm_period_data = []
            for p in period_metrics:
                llm_period_data.append({
                    'period': p['period'],
                    'overall_score': p['metrics'].get('overall_score'),
                    'savings_ratio_score': p['metrics'].get('savings_ratio_score'),
                    'spending_stability_score': p['metrics'].get('spending_stability_score'),
                    'cashflow_score': p['metrics'].get('cashflow_score'),
                    # Accessing income/expenses from llm_insights if available, else default to 0
                    'total_income': p['metrics'].get('llm_insights', {}).get('FINANCIAL OVERVIEW', {}).get('Total Income', 0.0),
                    'total_expenses': p['metrics'].get('llm_insights', {}).get('FINANCIAL OVERVIEW', {}).get('Total Expenses', 0.0),
                })
            
            prompt = f"""
            Analyze the following time-series financial health metrics and provide a comprehensive trend analysis.
            The data shows financial health scores and related metrics for different periods.

            PERIOD FINANCIAL HEALTH METRICS:
            {json.dumps(llm_period_data, indent=2)}

            Based on this data, provide a detailed analysis focusing on:
            1.  **Overall trajectory**: Is the user's financial health generally "improving", "stable", or "declining"?
                Consider the overall scores and the direction of change. (e.g., "improving_gradually", "declining_sharply").
            2.  **Key drivers of change**: What specific metrics (e.g., spending stability, savings rate, cash flow) or underlying financial activities (e.g., increased income, reduced expenses in certain categories) appear to be driving the observed trend? Be specific.
            3.  **Warning signs or red flags**: Are there any negative trends or inconsistencies that warrant attention? (e.g., increasing debt, unstable spending, decreasing savings).
            4.  **Positive developments to reinforce**: What are the positive aspects of the financial trend that the user should continue? (e.g., consistent savings, stable income).
            5.  **Recommendations for improvement**: Concrete, actionable steps or advice to improve or maintain the financial health trend.

            Return the analysis in a structured JSON object. Ensure all fields are present as arrays, even if empty.

            ```json
            {{
                "trajectory_assessment": "The user's financial health is [improving|stable|declining|improving_gradually|etc.]...",
                "trend_drivers": ["Driver 1: explanation", "Driver 2: explanation"],
                "warning_signs": ["Warning 1: explanation", "Warning 2: explanation"],
                "positive_developments": ["Development 1: explanation", "Development 2: explanation"],
                "recommendations": ["Recommendation 1: step", "Recommendation 2: step"]
            }}
            ```
            """
            
            messages = [
                SystemMessage(content="You are a financial analyst expert in trend detection and behavioral finance. Provide clear, concise, and actionable insights based purely on the provided data."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            raw_response = response.content
            
            # Extract JSON from response, looking for markdown code block first
            match = re.search(r'```json\n(\{.*?\})\n```', raw_response, re.DOTALL)
            if not match:
                match = re.search(r'\{.*\}', raw_response, re.DOTALL) # Fallback if not in code block

            if match:
                llm_insights = json.loads(match.group(1)) # Use group(1) for content inside curly braces
                
                # Add simple trajectory and change percentage for consistency
                llm_insights['trajectory'] = llm_insights.get('trajectory', simple_trajectory)
                llm_insights['change_percentage'] = llm_insights.get('change_percentage', simple_change_percentage)
                llm_insights['confidence'] = llm_insights.get('confidence', 'high') # Assume high confidence from LLM
                return llm_insights
            else:
                raise ValueError(f"LLM did not return valid JSON for trend insights. Raw response: {raw_response[:500]}...")
        
        except Exception as e:
            # Fallback for LLM failure, providing a basic summary
            st.warning(f"Error generating LLM trend insights: {str(e)}. Providing basic trend summary.")
            return {
                'trajectory_assessment': f"Basic trend analysis indicates the trajectory is {simple_trajectory}.",
                'trajectory': simple_trajectory, # Keep the simple trajectory
                'change_percentage': round(simple_change_percentage, 1),
                'confidence': 'low',
                'trend_drivers': [f"Change is {round(simple_change_percentage, 1)}%"],
                'warning_signs': [],
                'positive_developments': [],
                'recommendations': []
            }

