# data_models.py - Core Data Models and Schemas
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

@dataclass
class EnhancedTransaction:
    """
    Represents an enhanced financial transaction with categorization and LLM analysis metadata.
    """
    date: str
    description: str
    amount: float
    category: str = "other"  # e.g., food_dining, transportation, income, fixed_expenses
    subcategory: str = "uncategorized" # e.g., groceries, fuel, rent_mortgage, salary
    spending_type: str = "regular"  # e.g., regular, impulse, seasonal, major_purchase
    confidence: float = 0.0  # Confidence score (0.0 to 1.0) for categorization/spending type
    reasoning: str = ""  # Explanation for categorization or spending type from LLM
    balance: Optional[float] = None # Account balance after the transaction, if available
    reference: Optional[str] = None # Any external reference ID for the transaction

    def to_dict(self) -> Dict[str, Any]:
        """Converts the EnhancedTransaction object to a dictionary."""
        return {
            'date': self.date,
            'description': self.description,
            'amount': self.amount,
            'category': self.category,
            'subcategory': self.subcategory,
            'spending_type': self.spending_type,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'balance': self.balance,
            'reference': self.reference
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedTransaction':
        """Creates an EnhancedTransaction object from a dictionary."""
        return cls(**data)


@dataclass
class FinancialHealthMetrics:
    """
    Represents various scores and insights related to a user's financial health.
    Includes basic metrics and LLM-generated insights.
    """
    overall_score: float
    risk_level: str # e.g., "Low Risk", "Moderate Risk", "High Risk", "Critical Risk"
    savings_ratio_score: float # Score related to savings rate
    emergency_fund_score: float # Score related to emergency fund coverage
    spending_stability_score: float # Score indicating consistency of spending
    cashflow_score: float # Score indicating the trend of cash flow (positive/negative)
    analysis_quality: str = "basic"  # Indicates the depth of analysis: "basic", "enhanced" (with LLM)
    llm_insights: Dict[str, Any] = field(default_factory=dict) # LLM-generated insights (strengths, weaknesses, etc.)
    # llm_recommendations: Dict = field(default_factory=dict) # Removed as it's redundant with llm_insights
    trend_analysis: Dict[str, Any] = field(default_factory=dict) # LLM-generated trend analysis (trajectory, drivers, etc.)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the FinancialHealthMetrics object to a dictionary."""
        return {
            'overall_score': self.overall_score,
            'risk_level': self.risk_level,
            'savings_ratio_score': self.savings_ratio_score,
            'emergency_fund_score': self.emergency_fund_score,
            'spending_stability_score': self.spending_stability_score,
            'cashflow_score': self.cashflow_score,
            'llm_insights': self.llm_insights,
            # 'llm_recommendations': self.llm_recommendations, # Removed
            'trend_analysis': self.trend_analysis,
            'analysis_quality': self.analysis_quality
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialHealthMetrics':
        """Creates a FinancialHealthMetrics object from a dictionary."""
        return cls(**data)


@dataclass
class ConversationEntry:
    """
    Represents a single entry in the conversation history with the AI advisor,
    including user query, AI response, and metadata about the interaction.
    """
    user_query: str
    ai_response: str
    emotion: str = "neutral" # Detected emotion in user's query
    intent: str = "general_inquiry" # Detected intent of user's query
    topics: List[str] = field(default_factory=list) # List of financial topics identified
    confidence: str = "medium" # Confidence level of AI's understanding/response
    follow_up_suggestions: List[str] = field(default_factory=list) # Suggested follow-up questions for the user
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat()) # Timestamp of the conversation entry

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ConversationEntry object to a dictionary."""
        return {
            'user_query': self.user_query,
            'ai_response': self.ai_response,
            'emotion': self.emotion,
            'intent': self.intent,
            'topics': self.topics,
            'confidence': self.confidence,
            'follow_up_suggestions': self.follow_up_suggestions,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        """Creates a ConversationEntry object from a dictionary."""
        return cls(**data)


@dataclass
class LLMAnalysisResult:
    """
    Standardized structure for capturing the result of any LLM-based operation,
    including success status, extracted data, confidence, and error messages.
    """
    success: bool
    data: Dict[str, Any] = field(default_factory=dict) # Data extracted or generated by the LLM
    confidence: str = "medium" # Confidence level of the LLM's analysis
    reasoning: str = "" # Explanation for the LLM's output
    error_message: str = "" # Error message if the LLM operation failed
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat()) # Timestamp of the analysis result

    def to_dict(self) -> Dict[str, Any]:
        """Converts the LLMAnalysisResult object to a dictionary."""
        return {
            'success': self.success,
            'data': self.data,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMAnalysisResult':
        """Creates an LLMAnalysisResult object from a dictionary."""
        return cls(**data)


@dataclass
class UserProfile:
    """
    Represents user-specific preferences, financial goals, and other contextual information.
    """
    user_id: str = "" # Unique identifier for the user
    financial_goals: List[str] = field(default_factory=list) # e.g., "save for down payment", "pay off debt"
    risk_tolerance: str = "medium" # e.g., "low", "medium", "high"
    income_bracket: str = "unknown" # e.g., "low", "middle", "high"
    age_group: str = "unknown" # e.g., "18-24", "25-34", "35-44"
    conversation_preferences: Dict[str, Any] = field(default_factory=dict) # Preferences for AI interaction

    def to_dict(self) -> Dict[str, Any]:
        """Converts the UserProfile object to a dictionary."""
        return {
            'user_id': self.user_id,
            'financial_goals': self.financial_goals,
            'risk_tolerance': self.risk_tolerance,
            'income_bracket': self.income_bracket,
            'age_group': self.age_group,
            'conversation_preferences': self.conversation_preferences
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Creates a UserProfile object from a dictionary."""
        return cls(**data)


@dataclass
class FinancialDataSummary:
    """
    Aggregates financial data, metrics, and patterns for comprehensive LLM analysis.
    This acts as a consolidated view for passing context to LLMs.
    """
    transactions: List[EnhancedTransaction]
    metrics: FinancialHealthMetrics
    temporal_patterns: Dict[str, Any] # e.g., monthly spending, income variations
    spending_breakdown: Dict[str, Any] # e.g., by category, by spending behavior
    summary_quality: str = "basic" # Quality/depth of this summary: "basic", "enhanced"

    def to_dict(self) -> Dict[str, Any]:
        """Converts the FinancialDataSummary object to a dictionary."""
        return {
            'transactions': [tx.to_dict() for tx in self.transactions],
            'metrics': self.metrics.to_dict(),
            'temporal_patterns': self.temporal_patterns,
            'spending_breakdown': self.spending_breakdown,
            'summary_quality': self.summary_quality
        }

    def get_spending_by_category(self) -> Dict[str, float]:
        """Calculates and returns total spending by category."""
        categories = {}
        for tx in self.transactions:
            if tx.amount < 0: # Only consider expenses
                categories[tx.category] = categories.get(tx.category, 0) + abs(tx.amount)
        return dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))

    def get_spending_trend(self) -> List[float]:
        """
        Calculates and returns a list of monthly expenses, showing spending trends over time.
        Note: This currently returns just the values, not the month labels.
        """
        monthly_expenses = {}
        for tx in self.transactions:
            if tx.amount < 0: # Only consider expenses
                month_key = tx.date[:7]  # Extracts YYYY-MM
                monthly_expenses[month_key] = monthly_expenses.get(month_key, 0) + abs(tx.amount)
        return list(monthly_expenses.values())

    def to_llm_context(self) -> Dict[str, Any]:
        """
        Converts the FinancialDataSummary into a structured JSON-like dictionary
        suitable for consumption by an LLM as context.
        """
        return {
            'financial_overview': {
                'total_transactions': len(self.transactions),
                'date_range': self._get_date_range(),
                'total_income': self._get_total_income(),
                'total_expenses': self._get_total_expenses(),
                'net_flow': self._get_net_flow(),
                'savings_rate': self._get_savings_rate()
            },
            'health_metrics': {
                'overall_score': self.metrics.overall_score,
                'risk_level': self.metrics.risk_level,
                'cashflow_score': self.metrics.cashflow_score,
                'savings_ratio_score': self.metrics.savings_ratio_score,
                'spending_stability_score': self.metrics.spending_stability_score,
                'emergency_fund_score': self.metrics.emergency_fund_score
            },
            'spending_breakdown': {
                'by_category': self.get_spending_by_category(),
                'by_behavior': self._get_spending_by_behavior()
            },
            'temporal_patterns': self.temporal_patterns # Assuming temporal_patterns is already structured
        }

    def _get_date_range(self) -> str:
        """Helper to get the date range of transactions."""
        dates = [datetime.fromisoformat(tx.date) for tx in self.transactions]
        if not dates: # Handle empty transactions list
            return "N/A"
        return f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"

    def _get_total_income(self) -> float:
        """Helper to calculate total income from transactions."""
        return sum(tx.amount for tx in self.transactions if tx.amount > 0)

    def _get_total_expenses(self) -> float:
        """Helper to calculate total expenses from transactions."""
        return sum(abs(tx.amount) for tx in self.transactions if tx.amount < 0)

    def _get_net_flow(self) -> float:
        """Helper to calculate net flow (income - expenses)."""
        return self._get_total_income() - self._get_total_expenses()

    def _get_savings_rate(self) -> float:
        """Helper to calculate savings rate."""
        income = self._get_total_income()
        if income == 0:
            return 0.0
        # Ensure savings rate is between 0 and 1
        return max(0.0, min(1.0, self._get_net_flow() / income))

    def _get_spending_by_behavior(self) -> Dict[str, float]:
        """Helper to calculate spending breakdown by behavior type."""
        behaviors = {}
        for tx in self.transactions:
            if tx.amount < 0: # Only consider expenses
                behavior = tx.spending_type
                behaviors[behavior] = behaviors.get(behavior, 0) + abs(tx.amount)
        return dict(sorted(behaviors.items(), key=lambda x: x[1], reverse=True))

