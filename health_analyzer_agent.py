# health_analyzer_agent.py - Financial Health Analysis Engine
import json
import re
from typing import List, Dict, Any, Optional
import streamlit as st # Keep streamlit import for logging/warnings within this agent
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from langchain.schema import HumanMessage, SystemMessage
from config_manager import llm_manager
from data_models import EnhancedTransaction, FinancialHealthMetrics


class HealthCalculatorEngine:
    """
    Core financial health metrics calculator.
    Calculates various scores based on transaction data (e.g., savings rate,
    emergency fund coverage, spending stability, cash flow trend).
    """

    def calculate_financial_health(self, transactions: List[EnhancedTransaction]) -> FinancialHealthMetrics:
        """
        Calculate core financial health metrics from a list of EnhancedTransaction objects.

        Args:
            transactions (List[EnhancedTransaction]): A list of financial transactions.

        Returns:
            FinancialHealthMetrics: An object containing calculated scores and risk level.
                                    Returns basic metrics even if issues occur.
        """
        # Initialize all scores to 0.0 to prevent NameError if calculation functions return early
        savings_rate_score_normalized = 0.0
        emergency_fund_coverage_normalized = 0.0
        spending_stability_score_normalized = 0.0
        cashflow_trend_score_normalized = 0.0

        if not transactions:
            # Return a default FinancialHealthMetrics if no transactions are provided
            return FinancialHealthMetrics(
                overall_score=0.0,
                risk_level="No Data",
                savings_ratio_score=0.0,
                emergency_fund_score=0.0,
                spending_stability_score=0.0,
                cashflow_score=0.0,
                analysis_quality='basic',
                llm_insights={"overall_assessment": "No transaction data available for analysis."}
            )

        try:
            # Convert EnhancedTransaction objects to DataFrame for easier calculations.
            # Ensure 'date', 'amount', and 'balance' (if present) are usable.
            df = pd.DataFrame([tx.to_dict() for tx in transactions])
            
            # Ensure 'amount' is numeric and 'date' is datetime for calculations
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Drop rows with invalid dates or zero amounts, as they won't contribute to health metrics
            df = df.dropna(subset=['date', 'amount'])
            df = df[df['amount'] != 0].copy()

            if df.empty:
                 # Return a default FinancialHealthMetrics if no valid transactions after cleaning
                return FinancialHealthMetrics(
                    overall_score=0.0,
                    risk_level="No Valid Data",
                    savings_ratio_score=0.0,
                    emergency_fund_score=0.0,
                    spending_stability_score=0.0,
                    cashflow_score=0.0,
                    analysis_quality='basic',
                    llm_insights={"overall_assessment": "No valid transaction data after initial cleaning."}
                )

            # Component weights for overall score calculation
            component_weights = {
                'savings_ratio': 0.30,
                'emergency_fund': 0.20,
                'spending_stability': 0.25,
                'cashflow': 0.25
            }

            # Calculate individual component scores (0.0 to 1.0 scale)
            # These variables are now initialized above, so no NameError will occur
            savings_rate_score_normalized = self._calculate_savings_rate(df)
            emergency_fund_coverage_normalized = self._calculate_emergency_fund_coverage(df)
            spending_stability_score_normalized = self._calculate_spending_stability(df)
            cashflow_trend_score_normalized = self._calculate_cashflow_trend(df)

            # Calculate overall score out of 100
            overall_score = (
                savings_rate_score_normalized * component_weights['savings_ratio'] +
                emergency_fund_coverage_normalized * component_weights['emergency_fund'] +
                spending_stability_score_normalized * component_weights['spending_stability'] +
                cashflow_trend_score_normalized * component_weights['cashflow']
            ) * 100

            # Determine risk level based on overall score
            if overall_score >= 80:
                risk_level = "Low Risk"
            elif overall_score >= 60:
                risk_level = "Moderate Risk"
            elif overall_score >= 40:
                risk_level = "High Risk"
            else:
                risk_level = "Critical Risk"

            # Return FinancialHealthMetrics object
            return FinancialHealthMetrics(
                overall_score=round(overall_score, 1),
                risk_level=risk_level,
                savings_ratio_score=round(savings_rate_score_normalized * 100, 1),
                emergency_fund_score=round(emergency_fund_coverage_normalized * 100, 1), # Scale to 100 for score
                spending_stability_score=round(spending_stability_score_normalized * 100, 1),
                cashflow_score=round(cashflow_trend_score_normalized * 100, 1),
                analysis_quality='basic' # Default to basic, LLMHealthAnalyzer will upgrade this
            )

        except Exception as e:
            # If any calculation fails, return a default/error state for resilience
            st.error(f"Error calculating core financial health metrics: {str(e)}")
            return FinancialHealthMetrics(
                overall_score=0.0,
                risk_level="Calculation Error",
                savings_ratio_score=0.0,
                emergency_fund_score=0.0,
                spending_stability_score=0.0,
                cashflow_score=0.0,
                analysis_quality='basic',
                llm_insights={"overall_assessment": f"Failed to calculate core metrics: {str(e)}"}
            )


    def _calculate_savings_rate(self, df: pd.DataFrame) -> float:
        """
        Calculates the savings rate based on total income and expenses.
        Returns a value between 0.0 and 1.0.
        """
        income = df[df['amount'] > 0]['amount'].sum()
        expenses = abs(df[df['amount'] < 0]['amount'].sum())
        
        if income <= 0: # Avoid division by zero or negative income scenarios
            return 0.0
        
        net_flow = income - expenses
        savings_rate = net_flow / income
        
        return max(0.0, min(1.0, savings_rate)) # Clamp between 0 and 1

    def _calculate_emergency_fund_coverage(self, df: pd.DataFrame) -> float:
        """
        Calculates emergency fund coverage in months, based on average monthly expenses.
        Attempts to use latest balance or average income if balance is not available.
        Returns a normalized score between 0.0 and 1.0 (where 1.0 indicates 6+ months).
        """
        df_expenses = df[df['amount'] < 0]
        if df_expenses.empty:
            return 1.0 # If no expenses, assume full coverage (or no need)

        # Calculate average monthly expenses
        df_expenses['month_key'] = df_expenses['date'].dt.to_period('M')
        monthly_expense_sums = df_expenses.groupby('month_key')['amount'].sum().abs()
        
        if monthly_expense_sums.empty or monthly_expense_sums.mean() == 0:
            return 1.0 # If no meaningful monthly expenses, full coverage

        avg_monthly_expenses = monthly_expense_sums.mean()

        current_balance = 0.0
        # Prioritize 'balance' column if available and has valid data
        if 'balance' in df.columns and not df['balance'].isna().all():
            latest_balance_series = df['balance'].dropna()
            if not latest_balance_series.empty:
                current_balance = latest_balance_series.iloc[-1]
            else:
                st.warning("Balance column found but contains no valid data.")
        
        # If no valid balance or balance is 0, use average income as a proxy for savings capacity
        # This is a heuristic if true emergency fund balance is not directly available.
        if current_balance <= 0:
            income = df[df['amount'] > 0]['amount'].sum()
            expenses = abs(df[df['amount'] < 0]['amount'].sum())
            net_flow = income - expenses
            if net_flow > 0: # If there's positive net flow, use it as a proxy for 'saved' amount
                current_balance = net_flow # Using net flow over the period as a proxy for saved amount

        if current_balance <= 0 or avg_monthly_expenses == 0:
            return 0.0 # No fund or no expenses means 0 coverage

        months_coverage = current_balance / avg_monthly_expenses
        
        # Normalize to a 0-1 scale, where 1.0 represents 6 or more months of coverage
        # max(0, min(6, months_coverage)) / 6.0
        return max(0.0, min(1.0, months_coverage / 6.0))


    def _calculate_spending_stability(self, df: pd.DataFrame) -> float:
        """
        Calculates spending stability using the coefficient of variation (CV) of monthly expenses.
        Returns a normalized score between 0.0 and 1.0 (higher score means more stable spending).
        """
        df_expenses = df[df['amount'] < 0].copy()
        
        if df_expenses.empty:
            return 1.0 # Perfectly stable if no expenses

        df_expenses['month_key'] = df_expenses['date'].dt.to_period('M')
        monthly_expenses = df_expenses.groupby('month_key')['amount'].sum().abs()

        if len(monthly_expenses) < 2:
            return 0.7 # Minimum data points for meaningful std dev, return a moderate score

        std_dev = monthly_expenses.std()
        mean = monthly_expenses.mean()

        if mean == 0:
            return 1.0 # Perfectly stable if mean expenses are zero (no spending)

        cv = std_dev / mean # Coefficient of Variation
        
        # Invert CV to get a stability score: lower CV means higher stability
        # Scaling: A CV of 0 means perfect stability (score 1.0). A higher CV means lower score.
        # We cap CV at 2 for normalization purpose (e.g., CV > 2 implies very unstable)
        normalized_cv = min(1.0, cv / 2.0) # Map CV to 0-1 range (0=perfect, 1=highly variable)
        stability_score = 1.0 - normalized_cv # Invert to get stability score (1=stable, 0=unstable)
        
        return max(0.0, min(1.0, stability_score)) # Clamp between 0 and 1

    def _calculate_cashflow_trend(self, df: pd.DataFrame) -> float:
        """
        Calculates the trend of net cash flow over the most recent 6 months using linear regression.
        Returns a normalized score between 0.0 and 1.0 (1.0 for strong positive trend, 0.0 for strong negative).
        """
        # Ensure 'amount' is numeric and 'date' is datetime for consistency
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Filter for recent data (last 6 months relative to the latest transaction date in the data)
        if df['date'].empty:
            return 0.5 # Neutral if no dates

        latest_transaction_date = df['date'].max()
        six_months_ago_from_latest = latest_transaction_date - pd.DateOffset(months=6)
        recent_data = df[df['date'] >= six_months_ago_from_latest].copy()
        
        if recent_data.empty:
             return 0.5 # Neutral if no recent data

        recent_data['month_key'] = recent_data['date'].dt.to_period('M')
        # Calculate monthly net flow (income - expenses)
        monthly_net = recent_data.groupby('month_key')['amount'].sum().reset_index()

        if len(monthly_net) < 2:
            return 0.5 # Not enough data points to determine a trend, return neutral score

        # Assign a numeric index to months for linear regression
        # Sort by month_key to ensure correct chronological order for month_num
        monthly_net = monthly_net.sort_values('month_key').reset_index(drop=True)
        monthly_net['month_num'] = (monthly_net['month_key'].apply(lambda x: x.year * 12 + x.month) - 
                                     monthly_net['month_key'].apply(lambda x: x.year * 12 + x.month).min())

        x = monthly_net['month_num'].values.reshape(-1, 1) # Independent variable (time)
        y = monthly_net['amount'].values # Dependent variable (net cash flow)

        # Calculate slope of the linear regression line
        # Check for sufficient unique values for regression
        if len(np.unique(x)) < 2: # Need at least two distinct month_num values
            return 0.5

        try:
            slope = np.polyfit(x.flatten(), y, 1)[0]
        except np.linalg.LinAlgError:
            # Handle singular matrix or other numpy errors during polyfit
            return 0.5 # Return neutral if regression fails

        # Normalize the slope to a 0-1 range.
        y_std = y.std()
        if y_std == 0: # No variability in net flow, assume stable
            return 0.5
        
        # Scale slope relative to the variability of net flows
        # Arbitrary scaling factor, adjust as needed. Here, 0.1 means 10% of std dev is significant.
        normalized_slope_component = slope / (y_std * 5) # Scale factor 5, adjust based on desired sensitivity

        # Use tanh to map potentially large positive/negative slopes to -1 to 1 range,
        # then shift and scale to 0 to 1.
        # tanh(x) approaches 1 for large positive x, -1 for large negative x.
        # (tanh(x) + 1) / 2 maps from [-1, 1] to [0, 1].
        cashflow_score = (np.tanh(normalized_slope_component) + 1) / 2

        return max(0.0, min(1.0, cashflow_score)) # Clamp between 0 and 1


class LLMHealthAnalyzer(HealthCalculatorEngine):
    """
    LLM-enhanced financial health analysis.
    Extends HealthCalculatorEngine to provide qualitative insights using an LLM.
    """

    def __init__(self):
        super().__init__() # Initialize the base HealthCalculatorEngine
        self.llm = llm_manager.get_client() # Get the LLM client from config_manager

    def analyze_financial_health(self, transactions: List[EnhancedTransaction]) -> FinancialHealthMetrics:
        """
        Calculates basic financial health metrics and then enhances them with LLM-generated insights.

        Args:
            transactions (List[EnhancedTransaction]): A list of financial transactions.

        Returns:
            FinancialHealthMetrics: An object containing both calculated scores and LLM insights.
        """
        # First, calculate the base financial health metrics using the parent class method
        base_metrics_obj = super().calculate_financial_health(transactions)
        base_metrics_dict = base_metrics_obj.to_dict() # Convert to dict for updating

        # If LLM is not available, return the basic metrics directly
        if not self.llm or not llm_manager.is_available():
            st.info("LLM not available for enhanced financial health analysis. Providing basic metrics.")
            return base_metrics_obj # Return the object directly

        try:
            # Prepare the prompt for the LLM using the transactions and base metrics
            prompt = self._create_insights_prompt(transactions, base_metrics_dict)

            messages = [
                SystemMessage(content="You are a certified financial planner and behavioral finance coach. Provide empathetic, practical, and structured advice."),
                HumanMessage(content=prompt)
            ]

            # Invoke the LLM to get financial insights
            response = self.llm.invoke(messages)
            raw_response = response.content

            # Extract JSON from the LLM's response. It might be within a markdown code block.
            match = re.search(r'```json\n(\{.*?\})\n```', raw_response, re.DOTALL)
            if not match:
                match = re.search(r'\{.*\}', raw_response, re.DOTALL) # Fallback if not in code block

            if match:
                insights = json.loads(match.group(1)) # Use group(1) to get content inside curly braces
            else:
                raise ValueError(f"No valid JSON insights found in LLM response. Raw response: {raw_response[:500]}...")

            # Update the base metrics dictionary with LLM insights
            full_metrics_dict = base_metrics_dict.copy()
            full_metrics_dict.update({
                'llm_insights': insights,
                'analysis_quality': 'enhanced' # Mark as enhanced analysis
            })

            # Create and return a FinancialHealthMetrics object with enhanced insights
            return FinancialHealthMetrics(**full_metrics_dict)

        except Exception as e:
            # If LLM analysis fails, log the warning and return the basic metrics
            st.warning(f"LLM enhanced analysis failed: {e}. Falling back to basic metrics.")
            return base_metrics_obj # Return the original base_metrics_obj


    def _create_insights_prompt(self, transactions: List[EnhancedTransaction], base_metrics: Dict) -> str:
        """
        Generates a detailed prompt for the LLM, including financial overview,
        category breakdown, and calculated health metrics, to guide its analysis.
        """
        # Create a DataFrame from transactions for easier aggregation
        df = pd.DataFrame([tx.to_dict() for tx in transactions])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0) # Ensure numeric
        
        income = df[df['amount'] > 0]['amount'].sum()
        expenses = abs(df[df['amount'] < 0]['amount'].sum())
        net_flow = income - expenses

        # Calculate category breakdown
        category_breakdown = {}
        for tx in transactions: # Iterate through EnhancedTransaction objects to use categorized data
            # Ensure category is not 'other' or 'uncategorized' if better ones are available
            category_key = tx.category if tx.category and tx.category != "other" else "uncategorized_or_other"
            category_breakdown[category_key] = category_breakdown.get(category_key, 0) + abs(tx.amount)
        
        # Sort category breakdown for consistent prompt input
        sorted_category_breakdown = dict(sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True))

        # Handle division by zero for net flow percentage in prompt
        net_flow_percentage_str = "N/A"
        if income > 0:
            net_flow_percentage_str = f"{'+ ' if net_flow >= 0 else ''}{net_flow/income*100:.1f}%"

        # Emergency Fund Coverage formatting for prompt
        emergency_fund_months = round(base_metrics.get('emergency_fund_score', 0) / (100/6), 1) # Convert score back to months out of 6

        return f"""
        Analyze the user's financial situation based on the provided data and metrics.
        Your goal is to provide empathetic, practical, and structured advice.

        FINANCIAL OVERVIEW (from raw data):
        - Total Income: ${income:,.2f}
        - Total Expenses: ${expenses:,.2f}
        - Net Flow: ${net_flow:,.2f} ({net_flow_percentage_str} of income)

        SPENDING BREAKDOWN (by Category):
        {json.dumps({k: f"${v:,.2f}" for k, v in sorted_category_breakdown.items()}, indent=2)}

        CALCULATED HEALTH METRICS (on a 0-100 scale unless specified):
        - Overall Score: {base_metrics.get('overall_score', 0)}/100 (Risk Level: {base_metrics.get('risk_level', 'Unknown')})
        - Savings Ratio Score: {base_metrics.get('savings_ratio_score', 0)}%
        - Emergency Fund Coverage: {emergency_fund_months}/6 months (score: {base_metrics.get('emergency_fund_score', 0)}%)
        - Spending Stability Score: {base_metrics.get('spending_stability_score', 0)}%
        - Cashflow Trend Score: {base_metrics.get('cashflow_score', 0)}%

        INSTRUCTIONS FOR YOUR RESPONSE:
        Provide the following in a single JSON object. Ensure all fields are present, even if empty arrays or strings.
        1.  **overall_assessment**: A concise 2-3 sentence summary of the user's financial health.
        2.  **strengths**: An array of up to 5 key financial strengths.
        3.  **weaknesses**: An array of up to 5 key financial weaknesses.
        4.  **risk_factors**: An array of up to 5 potential financial risk factors.
        5.  **behavioral_insights**: An array of up to 5 observations about spending/saving behavior.
        6.  **trend_analysis**: A 2-3 sentence summary of any noticeable financial trends.
        7.  **key_metrics_interpretation**: A 2-3 sentence interpretation of the main health metrics.

        Return structured JSON in a markdown code block:
        ```json
        {{
            "overall_assessment": "...",
            "strengths": ["...", "..."],
            "weaknesses": ["...", "..."],
            "risk_factors": ["...", "..."],
            "behavioral_insights": ["...", "..."],
            "trend_analysis": "...",
            "key_metrics_interpretation": "..."
        }}
        ```
        """

