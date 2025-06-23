# ==============================================================================
# test_data_generator.py - Comprehensive Test Data Generator
# ==============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
from typing import Dict, List, Tuple
import os


class FinancialTestDataGenerator:
    """Generate comprehensive test data for financial intelligence platform"""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible data"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Define realistic financial data parameters
        self.monthly_income_range = (3500, 8500)
        self.base_expenses = {
            'rent': (1200, 2500),
            'groceries': (300, 800),
            'utilities': (80, 200),
            'transportation': (200, 600),
            'entertainment': (100, 500),
            'dining': (150, 400),
            'shopping': (100, 600),
            'healthcare': (50, 300),
            'insurance': (150, 400),
            'subscriptions': (50, 200)
        }
        
        # Merchant names for realistic transactions
        self.merchants = {
            'groceries': [
                'WHOLE FOODS MARKET', 'KROGER #123', 'WALMART SUPERCENTER', 
                'SAFEWAY STORE', 'TRADER JOES', 'COSTCO WHOLESALE',
                'TARGET STORES', 'ALDI FOODS', 'PUBLIX SUPER MARKET'
            ],
            'dining': [
                'MCDONALDS', 'STARBUCKS COFFEE', 'CHIPOTLE MEXICAN GRILL',
                'SUBWAY', 'DOMINOS PIZZA', 'OLIVE GARDEN', 'PANERA BREAD',
                'CHICK-FIL-A', 'TACO BELL', 'APPLEBEES GRILL'
            ],
            'entertainment': [
                'NETFLIX.COM', 'SPOTIFY PREMIUM', 'AMAZON PRIME VIDEO',
                'AMC THEATRES', 'REGAL CINEMAS', 'DISNEY+', 'HULU',
                'XBOX LIVE', 'STEAM GAMES', 'APPLE ITUNES'
            ],
            'transportation': [
                'SHELL OIL', 'EXXON MOBIL', 'BP GAS STATION', 'CHEVRON',
                'UBER TRIP', 'LYFT RIDE', 'METRO TRANSIT', 'PARKING METER'
            ],
            'shopping': [
                'AMAZON.COM', 'AMAZON MARKETPLACE', 'BEST BUY', 'HOME DEPOT',
                'LOWES STORES', 'MACYS', 'KOHLS DEPT STORE', 'NIKE STORE',
                'ADIDAS OUTLET', 'BARNES & NOBLE'
            ],
            'healthcare': [
                'CVS PHARMACY', 'WALGREENS', 'RITE AID PHARMACY',
                'DR. SMITH FAMILY PRACTICE', 'DENTAL ASSOCIATES',
                'VISION CENTER', 'LAB CORP'
            ],
            'utilities': [
                'ELECTRIC COMPANY', 'GAS UTILITY SERVICE', 'WATER DEPT',
                'COMCAST CABLE', 'VERIZON WIRELESS', 'AT&T MOBILITY',
                'INTERNET SERVICE PROVIDER'
            ],
            'income': [
                'PAYROLL DEPOSIT - ACME CORP', 'SALARY - TECH SOLUTIONS INC',
                'FREELANCE PAYMENT', 'CONSULTING FEE', 'BONUS PAYMENT',
                'DIVIDEND PAYMENT', 'INTEREST PAYMENT'
            ]
        }
        
        # Financial scenarios for different user profiles
        self.user_profiles = {
            'struggling_student': {
                'income_multiplier': 0.4,
                'expense_volatility': 0.8,
                'savings_rate': -0.1,
                'impulse_spending': 0.7
            },
            'young_professional': {
                'income_multiplier': 0.8,
                'expense_volatility': 0.6,
                'savings_rate': 0.15,
                'impulse_spending': 0.5
            },
            'family_household': {
                'income_multiplier': 1.2,
                'expense_volatility': 0.4,
                'savings_rate': 0.1,
                'impulse_spending': 0.3
            },
            'high_earner': {
                'income_multiplier': 2.0,
                'expense_volatility': 0.3,
                'savings_rate': 0.25,
                'impulse_spending': 0.4
            },
            'retiree': {
                'income_multiplier': 0.6,
                'expense_volatility': 0.2,
                'savings_rate': -0.05,
                'impulse_spending': 0.2
            }
        }
    
    def generate_comprehensive_dataset(self, months: int = 12) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive test dataset with multiple scenarios"""
        datasets = {}
        
        # Generate data for each user profile
        for profile_name, profile_params in self.user_profiles.items():
            print(f"Generating data for {profile_name}...")
            
            # Generate transaction data
            transactions = self._generate_transactions(months, profile_params)
            
            # Create DataFrame
            df = pd.DataFrame(transactions)
            datasets[profile_name] = df
            
            # Save to CSV
            filename = f"test_data_{profile_name}_{months}months.csv"
            df.to_csv(filename, index=False)
            print(f"  Saved {len(df)} transactions to {filename}")
        
        # Generate edge case scenarios
        edge_cases = self._generate_edge_cases()
        for case_name, case_data in edge_cases.items():
            df = pd.DataFrame(case_data)
            datasets[f"edge_case_{case_name}"] = df
            filename = f"test_data_edge_case_{case_name}.csv"
            df.to_csv(filename, index=False)
            print(f"  Saved edge case '{case_name}' to {filename}")
        
        # Generate file format variations
        self._generate_format_variations(datasets['young_professional'])
        
        # Generate analysis summary
        self._generate_test_summary(datasets)
        
        return datasets
    
    def _generate_transactions(self, months: int, profile: Dict) -> List[Dict]:
        """Generate realistic transactions for a specific profile"""
        transactions = []
        start_date = datetime.now() - timedelta(days=months * 30)
        
        # Calculate profile-adjusted parameters
        base_income = random.uniform(*self.monthly_income_range) * profile['income_multiplier']
        
        for month in range(months):
            month_start = start_date + timedelta(days=month * 30)
            
            # Generate monthly income (1-3 payments per month)
            income_payments = random.randint(1, 3)
            monthly_income = base_income * random.uniform(0.95, 1.05)
            
            for payment in range(income_payments):
                payment_date = month_start + timedelta(days=random.randint(1, 28))
                payment_amount = monthly_income / income_payments
                
                transactions.append({
                    'date': payment_date.strftime('%Y-%m-%d'),
                    'description': random.choice(self.merchants['income']),
                    'amount': round(payment_amount, 2),
                    'category': 'income',
                    'actual_category': 'income'  # For testing categorization accuracy
                })
            
            # Generate monthly expenses
            monthly_transactions = self._generate_monthly_expenses(
                month_start, profile, monthly_income
            )
            transactions.extend(monthly_transactions)
        
        # Add some random one-time transactions
        special_transactions = self._generate_special_transactions(start_date, months, profile)
        transactions.extend(special_transactions)
        
        # Sort by date
        transactions.sort(key=lambda x: x['date'])
        
        return transactions
    
    def _generate_monthly_expenses(self, month_start: datetime, profile: Dict, monthly_income: float) -> List[Dict]:
        """Generate monthly recurring and variable expenses"""
        transactions = []
        
        for expense_type, (min_amt, max_amt) in self.base_expenses.items():
            # Adjust expense based on income and profile
            income_factor = monthly_income / 5000  # Normalize to $5k baseline
            base_amount = random.uniform(min_amt, max_amt) * income_factor
            
            # Add volatility based on profile
            volatility = profile['expense_volatility']
            actual_amount = base_amount * random.uniform(1 - volatility * 0.3, 1 + volatility * 0.3)
            
            # Determine number of transactions for this category this month
            if expense_type in ['rent', 'insurance', 'utilities']:
                # Fixed monthly expenses
                transaction_count = 1
            elif expense_type in ['groceries', 'transportation']:
                # Weekly-ish expenses
                transaction_count = random.randint(3, 6)
            else:
                # Variable frequency
                transaction_count = random.randint(1, 8)
            
            # Generate transactions for this expense type
            category_merchants = self.merchants.get(expense_type, [f"{expense_type.upper()} MERCHANT"])
            
            for i in range(transaction_count):
                transaction_date = month_start + timedelta(days=random.randint(1, 28))
                transaction_amount = actual_amount / transaction_count
                
                # Add some randomness to individual transaction amounts
                transaction_amount *= random.uniform(0.7, 1.3)
                
                # Impulse spending modifier
                if random.random() < profile['impulse_spending'] * 0.3:
                    transaction_amount *= random.uniform(1.2, 2.0)
                
                transactions.append({
                    'date': transaction_date.strftime('%Y-%m-%d'),
                    'description': random.choice(category_merchants),
                    'amount': -round(transaction_amount, 2),  # Negative for expenses
                    'category': 'unknown',  # For testing categorization
                    'actual_category': expense_type
                })
        
        return transactions
    
    def _generate_special_transactions(self, start_date: datetime, months: int, profile: Dict) -> List[Dict]:
        """Generate special one-time transactions"""
        transactions = []
        
        # Large purchases
        large_purchases = [
            ('BEST BUY - LAPTOP PURCHASE', -1200, 'shopping'),
            ('HONDA DEALERSHIP - CAR REPAIR', -800, 'transportation'),
            ('EMERGENCY ROOM VISIT', -2500, 'healthcare'),
            ('VACATION RENTAL - AIRBNB', -600, 'entertainment'),
            ('FURNITURE STORE', -900, 'shopping')
        ]
        
        # Add 1-3 large purchases over the period
        for _ in range(random.randint(1, 3)):
            purchase = random.choice(large_purchases)
            purchase_date = start_date + timedelta(days=random.randint(30, months * 30 - 30))
            
            transactions.append({
                'date': purchase_date.strftime('%Y-%m-%d'),
                'description': purchase[0],
                'amount': purchase[1] * random.uniform(0.8, 1.2),
                'category': 'unknown',
                'actual_category': purchase[2]
            })
        
        # Seasonal transactions
        seasonal_transactions = [
            ('AMAZON.COM - HOLIDAY GIFTS', -300, 'shopping', [11, 12]),  # Nov-Dec
            ('TAX REFUND', 1200, 'income', [2, 3, 4]),  # Feb-Apr
            ('VACATION EXPENSES', -800, 'entertainment', [6, 7, 8]),  # Summer
            ('BACK TO SCHOOL SHOPPING', -200, 'shopping', [8, 9])  # Aug-Sep
        ]
        
        for transaction in seasonal_transactions:
            if random.random() < 0.7:  # 70% chance of seasonal transaction
                # Find appropriate month
                current_month = start_date.month
                target_months = transaction[3]
                
                for month_offset in range(months):
                    check_month = (current_month + month_offset - 1) % 12 + 1
                    if check_month in target_months:
                        transaction_date = start_date + timedelta(days=month_offset * 30 + random.randint(1, 28))
                        
                        transactions.append({
                            'date': transaction_date.strftime('%Y-%m-%d'),
                            'description': transaction[0],
                            'amount': transaction[1] * random.uniform(0.7, 1.3),
                            'category': 'unknown',
                            'actual_category': transaction[2]
                        })
                        break
        
        return transactions
    
    def _generate_edge_cases(self) -> Dict[str, List[Dict]]:
        """Generate edge case scenarios for testing robustness"""
        edge_cases = {}
        
        # 1. Empty file
        edge_cases['empty_file'] = []
        
        # 2. Single transaction
        edge_cases['single_transaction'] = [{
            'date': '2024-01-15',
            'description': 'SINGLE TEST TRANSACTION',
            'amount': -25.50,
            'category': 'unknown',
            'actual_category': 'other'
        }]
        
        # 3. All positive amounts (income only)
        edge_cases['income_only'] = [
            {
                'date': f'2024-01-{day:02d}',
                'description': f'INCOME SOURCE {i}',
                'amount': random.uniform(1000, 3000),
                'category': 'unknown',
                'actual_category': 'income'
            }
            for i, day in enumerate(range(1, 11), 1)
        ]
        
        # 4. All negative amounts (expenses only)
        edge_cases['expenses_only'] = [
            {
                'date': f'2024-01-{day:02d}',
                'description': f'EXPENSE {i}',
                'amount': -random.uniform(10, 200),
                'category': 'unknown',
                'actual_category': 'other'
            }
            for i, day in enumerate(range(1, 21), 1)
        ]
        
        # 5. Very large amounts
        edge_cases['large_amounts'] = [
            {
                'date': '2024-01-05',
                'description': 'HOUSE DOWN PAYMENT',
                'amount': -50000.00,
                'category': 'unknown',
                'actual_category': 'other'
            },
            {
                'date': '2024-01-10',
                'description': 'INVESTMENT RETURN',
                'amount': 25000.00,
                'category': 'unknown',
                'actual_category': 'income'
            },
            {
                'date': '2024-01-15',
                'description': 'CAR PURCHASE',
                'amount': -30000.00,
                'category': 'unknown',
                'actual_category': 'transportation'
            }
        ]
        
        # 6. Very small amounts (micro-transactions)
        edge_cases['micro_transactions'] = [
            {
                'date': f'2024-01-{day:02d}',
                'description': f'MICRO PAYMENT {i}',
                'amount': -random.uniform(0.01, 0.99),
                'category': 'unknown',
                'actual_category': 'other'
            }
            for i, day in enumerate(range(1, 31), 1)
        ]
        
        # 7. Special characters and encoding issues
        edge_cases['special_characters'] = [
            {
                'date': '2024-01-01',
                'description': 'CAFÉ MÜLLER & CO.',
                'amount': -15.50,
                'category': 'unknown',
                'actual_category': 'dining'
            },
            {
                'date': '2024-01-02',
                'description': 'JOSÉ\'S RESTAURANT',
                'amount': -32.75,
                'category': 'unknown',
                'actual_category': 'dining'
            },
            {
                'date': '2024-01-03',
                'description': 'STORE #123 (PARKING)',
                'amount': -5.00,
                'category': 'unknown',
                'actual_category': 'transportation'
            },
            {
                'date': '2024-01-04',
                'description': 'TRANSFER: SAVINGS→CHECKING',
                'amount': 1000.00,
                'category': 'unknown',
                'actual_category': 'financial'
            }
        ]
        
        # 8. Duplicate transactions
        base_transaction = {
            'date': '2024-01-15',
            'description': 'STARBUCKS COFFEE',
            'amount': -4.25,
            'category': 'unknown',
            'actual_category': 'dining'
        }
        edge_cases['duplicates'] = [base_transaction.copy() for _ in range(5)]
        
        # 9. Zero amounts
        edge_cases['zero_amounts'] = [
            {
                'date': '2024-01-01',
                'description': 'ZERO AMOUNT TRANSACTION',
                'amount': 0.00,
                'category': 'unknown',
                'actual_category': 'other'
            },
            {
                'date': '2024-01-02',
                'description': 'ANOTHER ZERO TRANSACTION',
                'amount': 0.00,
                'category': 'unknown',
                'actual_category': 'other'
            }
        ]
        
        # 10. Date edge cases
        edge_cases['date_variations'] = [
            {
                'date': '2024-02-29',  # Leap year
                'description': 'LEAP YEAR TRANSACTION',
                'amount': -20.00,
                'category': 'unknown',
                'actual_category': 'other'
            },
            {
                'date': '2024-12-31',  # Year end
                'description': 'YEAR END TRANSACTION',
                'amount': -15.00,
                'category': 'unknown',
                'actual_category': 'other'
            },
            {
                'date': '2024-01-01',  # Year start
                'description': 'NEW YEAR TRANSACTION',
                'amount': -10.00,
                'category': 'unknown',
                'actual_category': 'other'
            }
        ]
        
        return edge_cases
    
    def _generate_format_variations(self, base_data: pd.DataFrame):
        """Generate different file format variations for testing file processor"""
        
        # 1. Different column names
        variations = {
            'bank_a_format': {
                'Date': 'date',
                'Transaction Description': 'description', 
                'Amount': 'amount'
            },
            'bank_b_format': {
                'Transaction Date': 'date',
                'Description': 'description',
                'Debit': 'debit',
                'Credit': 'credit'
            },
            'bank_c_format': {
                'posting_date': 'date',
                'memo': 'description',
                'transaction_amount': 'amount'
            },
            'export_format': {
                'DATE': 'date',
                'DESCRIPTION': 'description',
                'AMOUNT': 'amount',
                'BALANCE': 'balance'
            }
        }
        
        for format_name, column_mapping in variations.items():
            df_variant = base_data.copy()
            
            # Rename columns
            reverse_mapping = {v: k for k, v in column_mapping.items()}
            df_variant = df_variant.rename(columns=reverse_mapping)
            
            # Handle debit/credit format
            if 'debit' in column_mapping and 'credit' in column_mapping:
                df_variant['Debit'] = df_variant['amount'].apply(lambda x: abs(x) if x < 0 else '')
                df_variant['Credit'] = df_variant['amount'].apply(lambda x: x if x > 0 else '')
                df_variant = df_variant.drop('amount', axis=1)
            
            # Add balance column if specified
            if 'balance' in column_mapping.values():
                df_variant['BALANCE'] = df_variant['amount'].cumsum()
            
            # Save as CSV
            filename = f"test_data_format_{format_name}.csv"
            df_variant.to_csv(filename, index=False)
            print(f"  Generated format variation: {filename}")
            
            # Save as Excel
            excel_filename = f"test_data_format_{format_name}.xlsx"
            df_variant.to_excel(excel_filename, index=False)
            print(f"  Generated Excel format: {excel_filename}")
    
    def _generate_test_summary(self, datasets: Dict[str, pd.DataFrame]):
        """Generate comprehensive test summary and expected results"""
        summary = {
            'generation_date': datetime.now().isoformat(),
            'datasets': {},
            'test_scenarios': {},
            'expected_categorization_accuracy': {},
            'expected_health_scores': {}
        }
        
        for name, df in datasets.items():
            if df.empty:
                continue
                
            # Dataset statistics
            total_income = df[df['amount'] > 0]['amount'].sum()
            total_expenses = df[df['amount'] < 0]['amount'].abs().sum()
            transaction_count = len(df)
            date_range = f"{df['date'].min()} to {df['date'].max()}"
            
            summary['datasets'][name] = {
                'transaction_count': transaction_count,
                'total_income': round(total_income, 2),
                'total_expenses': round(total_expenses, 2),
                'net_flow': round(total_income - total_expenses, 2),
                'date_range': date_range,
                'categories_present': df['actual_category'].unique().tolist() if 'actual_category' in df.columns else []
            }
            
            # Expected categorization accuracy
            if 'actual_category' in df.columns:
                category_counts = df['actual_category'].value_counts()
                summary['expected_categorization_accuracy'][name] = {
                    'total_transactions': transaction_count,
                    'category_distribution': category_counts.to_dict(),
                    'expected_accuracy': {
                        'income': 0.95,  # High accuracy expected
                        'groceries': 0.85,
                        'dining': 0.80,
                        'transportation': 0.85,
                        'utilities': 0.90,
                        'entertainment': 0.75,
                        'shopping': 0.70,
                        'healthcare': 0.85,
                        'other': 0.60
                    }
                }
            
            # Expected health score ranges
            if not name.startswith('edge_case'):
                savings_rate = (total_income - total_expenses) / total_income if total_income > 0 else 0
                
                expected_score_range = {
                    'struggling_student': (25, 45),
                    'young_professional': (55, 75), 
                    'family_household': (45, 65),
                    'high_earner': (70, 90),
                    'retiree': (40, 60)
                }
                
                summary['expected_health_scores'][name] = {
                    'savings_rate': round(savings_rate, 3),
                    'expected_score_range': expected_score_range.get(name, (30, 70)),
                    'risk_factors': self._identify_risk_factors(df, name)
                }
        
        # Test scenarios for each agent
        summary['test_scenarios'] = {
            'file_processor': {
                'column_detection': [
                    'Standard format (Date, Description, Amount)',
                    'Debit/Credit format',
                    'Different column names',
                    'Mixed case headers',
                    'Excel formats'
                ],
                'edge_cases': [
                    'Empty files',
                    'Single transaction',
                    'Special characters',
                    'Large datasets'
                ]
            },
            'categorizer': {
                'accuracy_tests': [
                    'Common merchant names',
                    'Ambiguous descriptions', 
                    'Special characters',
                    'Amount-based classification'
                ],
                'confidence_scoring': [
                    'High confidence: Clear merchant names',
                    'Medium confidence: Partial matches',
                    'Low confidence: Ambiguous descriptions'
                ]
            },
            'health_analyzer': {
                'score_validation': [
                    'High earner → High scores',
                    'Struggling student → Low scores', 
                    'Balanced family → Medium scores'
                ],
                'insight_generation': [
                    'Spending pattern analysis',
                    'Risk factor identification',
                    'Improvement suggestions'
                ]
            },
            'conversational': {
                'context_awareness': [
                    'Reference specific transactions',
                    'Remember conversation history',
                    'Adapt to user emotional state'
                ],
                'financial_expertise': [
                    'Provide relevant advice',
                    'Suggest appropriate actions',
                    'Educational responses'
                ]
            }
        }
        
        # Save summary
        with open('test_data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nGenerated comprehensive test summary: test_data_summary.json")
        print(f"Total datasets created: {len(datasets)}")
        print(f"Total transactions across all datasets: {sum(len(df) for df in datasets.values())}")
    
    def _identify_risk_factors(self, df: pd.DataFrame, profile_name: str) -> List[str]:
        """Identify expected risk factors for each profile"""
        risk_factors = []
        
        if df.empty:
            return risk_factors
        
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expenses = df[df['amount'] < 0]['amount'].abs().sum()
        
        if total_expenses > total_income:
            risk_factors.append("Spending exceeds income")
        
        if profile_name == 'struggling_student':
            risk_factors.extend([
                "Low income relative to expenses",
                "High spending volatility",
                "Limited emergency fund"
            ])
        elif profile_name == 'young_professional':
            risk_factors.extend([
                "Potential lifestyle inflation",
                "Limited emergency fund building"
            ])
        elif profile_name == 'family_household':
            risk_factors.extend([
                "High expense obligations",
                "Limited savings flexibility"
            ])
        elif profile_name == 'retiree':
            risk_factors.extend([
                "Fixed income dependency",
                "Healthcare cost uncertainty"
            ])
        
        return risk_factors


def generate_all_test_data():
    """Generate complete test data suite"""
    print("🧪 Generating Comprehensive Test Data Suite")
    print("=" * 50)
    
    generator = FinancialTestDataGenerator()
    
    # Generate main datasets
    datasets = generator.generate_comprehensive_dataset(months=12)
    
    # Generate additional stress test data
    print("\n📊 Generating stress test scenarios...")
    
    # Very large dataset
    large_dataset_transactions = []
    for i in range(10000):  # 10k transactions
        date = datetime.now() - timedelta(days=random.randint(1, 365))
        amount = random.uniform(-500, 2000) if random.random() > 0.8 else random.uniform(-50, 50)
        
        large_dataset_transactions.append({
            'date': date.strftime('%Y-%m-%d'),
            'description': f'TRANSACTION {i:05d}',
            'amount': round(amount, 2),
            'category': 'unknown',
            'actual_category': random.choice(['groceries', 'dining', 'shopping', 'transportation'])
        })
    
    large_df = pd.DataFrame(large_dataset_transactions)
    large_df.to_csv('test_data_stress_large_dataset.csv', index=False)
    print(f"  Generated large dataset: {len(large_df)} transactions")
    
    # Performance test data
    performance_transactions = []
    for i in range(1000):
        date = datetime.now() - timedelta(days=i)
        performance_transactions.append({
            'date': date.strftime('%Y-%m-%d'),
            'description': f'PERFORMANCE TEST {i}',
            'amount': round(random.uniform(-100, 500), 2),
            'category': 'unknown',
            'actual_category': 'other'
        })
    
    perf_df = pd.DataFrame(performance_transactions)
    perf_df.to_csv('test_data_performance_test.csv', index=False)
    print(f"  Generated performance test data: {len(perf_df)} transactions")
    
    print("\n✅ Test data generation complete!")
    print("\nGenerated files:")
    
    # List all generated files
    test_files = [f for f in os.listdir('.') if f.startswith('test_data_')]
    for file in sorted(test_files):
        print(f"  📄 {file}")
    
    print(f"\n📋 Test Summary:")
    print(f"  • {len(datasets)} user profile datasets")
    print(f"  • Multiple file format variations")
    print(f"  • Edge case scenarios")
    print(f"  • Stress test data")
    print(f"  • Performance test data")
    print(f"  • Comprehensive test summary (JSON)")
    
    return datasets


if __name__ == "__main__":
    generate_all_test_data()