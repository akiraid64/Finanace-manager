"""
Models for financial data analysis and prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime, timedelta
import joblib
import os

from src.python.utils import load_transactions, DATA_DIR

# Define model save paths
MODEL_DIR = DATA_DIR / 'models'
EXPENSE_MODEL_PATH = MODEL_DIR / 'expense_predictor.joblib'
INCOME_MODEL_PATH = MODEL_DIR / 'income_predictor.joblib'
SAVINGS_MODEL_PATH = MODEL_DIR / 'savings_predictor.joblib'

class ExpensePredictionModel:
    """
    Advanced model to predict future expenses based on historical data
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.categorical_features = ['category']
        self.numeric_features = ['day_of_month', 'day_of_week', 'month', 'year', 'amount_trend']
        self.metrics = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Try to load a pre-trained model if it exists
        if os.path.exists(EXPENSE_MODEL_PATH):
            try:
                self._load_model()
                self.is_trained = True
            except Exception as e:
                print(f"Could not load saved model: {e}")
    
    def _prepare_data(self, df):
        """
        Prepare transaction data for modeling with advanced features
        
        Args:
            df (pandas.DataFrame): Transaction data
            
        Returns:
            tuple: (X features, y target values)
        """
        if df.empty:
            return None, None
        
        # Filter only expenses
        expenses_df = df[df['type'] == 'expense'].copy()
        
        if expenses_df.empty or len(expenses_df) < 5:
            return None, None
        
        # Convert date to datetime if not already
        expenses_df['date'] = pd.to_datetime(expenses_df['date'])
        
        # Sort by date
        expenses_df = expenses_df.sort_values('date')
        
        # Create temporal features
        expenses_df['day_of_month'] = expenses_df['date'].dt.day
        expenses_df['day_of_week'] = expenses_df['date'].dt.dayofweek
        expenses_df['month'] = expenses_df['date'].dt.month
        expenses_df['year'] = expenses_df['date'].dt.year
        expenses_df['quarter'] = expenses_df['date'].dt.quarter
        expenses_df['is_weekend'] = expenses_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Calculate rolling statistics and trends by category
        expense_by_category = expenses_df.groupby(['category', 'year', 'month'])['amount'].sum().reset_index()
        expense_by_category['year_month'] = expense_by_category['year'].astype(str) + '-' + expense_by_category['month'].astype(str)
        
        # Create a trend feature that shows average spending over the last 3 months
        expense_trends = {}
        for category in expenses_df['category'].unique():
            cat_data = expense_by_category[expense_by_category['category'] == category]
            if len(cat_data) >= 3:
                cat_data = cat_data.sort_values('year_month')
                expense_trends[category] = cat_data['amount'].rolling(window=3).mean().iloc[-1]
            else:
                expense_trends[category] = cat_data['amount'].mean() if not cat_data.empty else 0
        
        # Add the trend as a feature
        expenses_df['amount_trend'] = expenses_df['category'].map(expense_trends)
        expenses_df['amount_trend'] = expenses_df['amount_trend'].fillna(0)
        
        # One-hot encode the category
        X = expenses_df[['category', 'day_of_month', 'day_of_week', 'month', 'year', 'quarter', 
                        'is_weekend', 'amount_trend']]
        y = expenses_df['amount']
        
        return X, y
    
    def train(self):
        """
        Train the expense prediction model with advanced features and model selection
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        transactions_df = load_transactions()
        
        # Ensure date is datetime
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        
        X, y = self._prepare_data(transactions_df)
        
        if X is None or len(X) < 10:  # Need reasonable data to train
            return False
        
        # Split data for training and validation
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Define preprocessing for numeric and categorical features
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ]
            )
            
            # Try different models to find the best one
            models = {
                'rf': RandomForestRegressor(random_state=42),
                'gbm': GradientBoostingRegressor(random_state=42),
                'ridge': Ridge(random_state=42)
            }
            
            best_score = -float('inf')
            best_model_name = None
            
            for name, model in models.items():
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_test, y_test)
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
            
            # Train final model with best architecture
            final_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', models[best_model_name])
            ])
            
            final_pipeline.fit(X, y)
            
            # Save metrics
            y_pred = final_pipeline.predict(X_test)
            self.metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'model_type': best_model_name
            }
            
            # Save the model for future use
            self.model = final_pipeline
            self._save_model()
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def _save_model(self):
        """Save the trained model to disk"""
        try:
            joblib.dump(self.model, EXPENSE_MODEL_PATH)
            print(f"Model saved to {EXPENSE_MODEL_PATH}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load a previously trained model from disk"""
        self.model = joblib.load(EXPENSE_MODEL_PATH)
    
    def predict_next_month(self):
        """
        Predict expenses for the next month with confidence intervals
        
        Returns:
            dict: Dictionary with categories as keys and prediction details as values
        """
        if not self.is_trained:
            success = self.train()
            if not success:
                return {}
        
        transactions_df = load_transactions()
        
        # If we don't have data, return empty prediction
        if transactions_df.empty:
            return {}
        
        # Get unique expense categories
        categories = transactions_df[transactions_df['type'] == 'expense']['category'].unique()
        
        # Prepare data for next month prediction
        next_month = datetime.now() + timedelta(days=30)
        next_month_end = next_month.replace(day=28)  # Safe for all months
        
        predictions = {}
        
        for category in categories:
            # Calculate trend for this category
            cat_expenses = transactions_df[(transactions_df['type'] == 'expense') & 
                                          (transactions_df['category'] == category)]
            
            # Skip if no data for this category
            if cat_expenses.empty:
                continue
                
            # Get average monthly spending
            cat_expenses['date'] = pd.to_datetime(cat_expenses['date'])
            cat_expenses['year_month'] = cat_expenses['date'].dt.strftime('%Y-%m')
            monthly_avg = cat_expenses.groupby('year_month')['amount'].sum().mean()
            
            # Create feature row for prediction
            features = pd.DataFrame([{
                'category': category,
                'day_of_month': 15,  # Middle of month
                'day_of_week': next_month.weekday(),
                'month': next_month.month,
                'year': next_month.year,
                'quarter': (next_month.month - 1) // 3 + 1,
                'is_weekend': 1 if next_month.weekday() >= 5 else 0,
                'amount_trend': monthly_avg if not pd.isna(monthly_avg) else 0
            }])
            
            try:
                # Make prediction
                predicted_amount = max(0, self.model.predict(features)[0])
                
                # For confidence interval, we'll use a simple heuristic
                # based on historical variance
                historical_std = cat_expenses['amount'].std()
                if pd.isna(historical_std):
                    historical_std = 0
                    
                lower_bound = max(0, predicted_amount - 1.96 * historical_std)
                upper_bound = predicted_amount + 1.96 * historical_std
                
                # Store prediction with confidence interval
                predictions[category] = {
                    'amount': predicted_amount,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'historical_avg': monthly_avg if not pd.isna(monthly_avg) else 0
                }
            except Exception as e:
                print(f"Error predicting for category {category}: {e}")
        
        return predictions

class SavingsPredictionModel:
    """
    Model to predict future savings potential based on income, expenses, and trends
    """
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Try to load a pre-trained model if it exists
        if os.path.exists(SAVINGS_MODEL_PATH):
            try:
                self._load_model()
                self.is_trained = True
            except Exception as e:
                print(f"Could not load saved savings model: {e}")
                
    def _prepare_data(self, df):
        """
        Prepare transaction data for savings prediction modeling
        
        Args:
            df (pandas.DataFrame): Transaction data
            
        Returns:
            tuple: (X features, y target values)
        """
        if df.empty or len(df) < 10:
            return None, None
            
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by month and calculate monthly metrics
        df['year_month'] = df['date'].dt.strftime('%Y-%m')
        
        # Calculate monthly income, expenses, and savings
        monthly_data = []
        
        for month, month_df in df.groupby('year_month'):
            month_income = month_df[month_df['type'] == 'income']['amount'].sum()
            month_expenses = month_df[month_df['type'] == 'expense']['amount'].sum()
            month_savings = month_income - month_expenses
            
            # Count transactions
            num_income_transactions = len(month_df[month_df['type'] == 'income'])
            num_expense_transactions = len(month_df[month_df['type'] == 'expense'])
            
            # Get the month and year
            if not month_df.empty:
                date = pd.to_datetime(month_df['date'].iloc[0])
                month = date.month
                year = date.year
            else:
                continue
                
            monthly_data.append({
                'year': year,
                'month': month,
                'income': month_income,
                'expenses': month_expenses,
                'savings': month_savings,
                'num_income_transactions': num_income_transactions,
                'num_expense_transactions': num_expense_transactions,
                'expense_to_income_ratio': month_expenses / month_income if month_income > 0 else 1,
                'month_savings_rate': month_savings / month_income if month_income > 0 else 0
            })
            
        if not monthly_data:
            return None, None
            
        monthly_df = pd.DataFrame(monthly_data)
        
        # Create lagged features for time series prediction
        monthly_df['prev_month_income'] = monthly_df['income'].shift(1)
        monthly_df['prev_month_expenses'] = monthly_df['expenses'].shift(1)
        monthly_df['prev_month_savings'] = monthly_df['savings'].shift(1)
        monthly_df['prev_month_savings_rate'] = monthly_df['month_savings_rate'].shift(1)
        
        # Calculate 3-month rolling averages
        monthly_df['income_3m_avg'] = monthly_df['income'].rolling(window=3, min_periods=1).mean()
        monthly_df['expenses_3m_avg'] = monthly_df['expenses'].rolling(window=3, min_periods=1).mean()
        monthly_df['savings_3m_avg'] = monthly_df['savings'].rolling(window=3, min_periods=1).mean()
        
        # Drop rows with NaN (first row due to lagging)
        monthly_df = monthly_df.dropna()
        
        if monthly_df.empty:
            return None, None
            
        # Define features and target
        features = [
            'year', 'month', 'income', 'expenses', 
            'num_income_transactions', 'num_expense_transactions',
            'expense_to_income_ratio', 'prev_month_income', 
            'prev_month_expenses', 'prev_month_savings',
            'prev_month_savings_rate', 'income_3m_avg', 
            'expenses_3m_avg', 'savings_3m_avg'
        ]
        
        X = monthly_df[features]
        y = monthly_df['savings']
        
        return X, y
        
    def train(self):
        """
        Train the savings prediction model
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        transactions_df = load_transactions()
        
        X, y = self._prepare_data(transactions_df)
        
        if X is None or len(X) < 5:  # Need at least a few months of data
            return False
            
        # Create preprocessing pipeline
        preprocessor = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Define the full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])
        
        try:
            # Train the model
            pipeline.fit(X, y)
            self.model = pipeline
            self._save_model()
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training savings model: {e}")
            return False
            
    def _save_model(self):
        """Save the trained model to disk"""
        try:
            joblib.dump(self.model, SAVINGS_MODEL_PATH)
        except Exception as e:
            print(f"Error saving savings model: {e}")
            
    def _load_model(self):
        """Load a previously trained model from disk"""
        self.model = joblib.load(SAVINGS_MODEL_PATH)
        
    def predict_future_savings(self, months=6):
        """
        Predict savings for future months
        
        Args:
            months (int): Number of months to predict into the future
            
        Returns:
            dict: Dictionary with predicted savings and related metrics
        """
        if not self.is_trained:
            success = self.train()
            if not success:
                return {}
                
        transactions_df = load_transactions()
        
        if transactions_df.empty:
            return {}
            
        # Prepare historical data
        X, _ = self._prepare_data(transactions_df)
        
        if X is None or X.empty:
            return {}
            
        # Get the most recent data point to start projections
        latest_data = X.iloc[-1].to_dict()
        
        # Initialize prediction results
        predictions = {
            'months': [],
            'predicted_income': [],
            'predicted_expenses': [],
            'predicted_savings': [],
            'cumulative_savings': []
        }
        
        # Get the latest month and year
        current_year = latest_data['year']
        current_month = latest_data['month']
        
        # Simple growth rates based on historical data
        income_growth = 1.005  # 0.5% monthly income growth
        expense_growth = 1.003  # 0.3% monthly expense growth
        
        cumulative_savings = 0
        
        for i in range(1, months + 1):
            # Increment month and year
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
                
            # Project next month's income and expenses
            next_income = latest_data['income'] * (income_growth ** i)
            next_expenses = latest_data['expenses'] * (expense_growth ** i)
            
            # Create feature set for prediction
            features = latest_data.copy()
            features['year'] = current_year
            features['month'] = current_month
            features['income'] = next_income
            features['expenses'] = next_expenses
            features['prev_month_income'] = latest_data['income']
            features['prev_month_expenses'] = latest_data['expenses']
            features['prev_month_savings'] = latest_data['income'] - latest_data['expenses']
            features['prev_month_savings_rate'] = (features['prev_month_savings'] / features['prev_month_income']) if features['prev_month_income'] > 0 else 0
            
            # Update rolling averages
            features['income_3m_avg'] = (features['income_3m_avg'] * 2 + next_income) / 3
            features['expenses_3m_avg'] = (features['expenses_3m_avg'] * 2 + next_expenses) / 3
            
            # Predict savings
            try:
                df = pd.DataFrame([features])
                predicted_savings = max(0, self.model.predict(df)[0])
                cumulative_savings += predicted_savings
                
                # Store predictions
                predictions['months'].append(f"{current_year}-{current_month:02d}")
                predictions['predicted_income'].append(round(next_income, 2))
                predictions['predicted_expenses'].append(round(next_expenses, 2))
                predictions['predicted_savings'].append(round(predicted_savings, 2))
                predictions['cumulative_savings'].append(round(cumulative_savings, 2))
                
                # Update latest data for next iteration
                latest_data = features.copy()
                latest_data['savings'] = predicted_savings
                
            except Exception as e:
                print(f"Error predicting savings for month {current_month}: {e}")
                break
                
        return predictions

class FinancialGoalModel:
    """
    Model to analyze financial goals and predict time to reach them
    """
    
    def __init__(self):
        self.expense_model = ExpensePredictionModel()
        self.savings_model = SavingsPredictionModel()
        
    def analyze_savings_goal(self, target_amount, current_savings=0):
        """
        Analyze how long it will take to reach a savings goal
        
        Args:
            target_amount (float): Target savings amount
            current_savings (float): Current savings amount
            
        Returns:
            dict: Dictionary with projection details
        """
        # Make sure savings model is trained
        if not self.savings_model.is_trained:
            if not self.savings_model.train():
                return {
                    'feasible': False,
                    'message': 'Not enough historical data to make a prediction'
                }
        
        # Get savings projections for next 5 years (60 months)
        projections = self.savings_model.predict_future_savings(months=60)
        
        if not projections or 'cumulative_savings' not in projections:
            return {
                'feasible': False,
                'message': 'Could not generate savings projections'
            }
            
        remaining_amount = target_amount - current_savings
        
        # Check if goal is already achieved
        if remaining_amount <= 0:
            return {
                'feasible': True,
                'time_to_goal': 0,
                'time_unit': 'months',
                'message': 'Goal already achieved!',
                'current_savings': current_savings,
                'target_amount': target_amount
            }
            
        # Find when cumulative savings exceeds the target
        months_to_goal = 0
        for i, savings in enumerate(projections['cumulative_savings']):
            if savings >= remaining_amount:
                months_to_goal = i + 1
                break
                
        if months_to_goal > 0:
            # Goal is achievable within projected timeframe
            return {
                'feasible': True,
                'time_to_goal': months_to_goal,
                'time_unit': 'months',
                'message': f'You can reach your goal in {months_to_goal} months',
                'current_savings': current_savings,
                'target_amount': target_amount,
                'monthly_savings_needed': remaining_amount / months_to_goal,
                'projected_date': projections['months'][months_to_goal - 1] if months_to_goal <= len(projections['months']) else 'Beyond 5 years'
            }
        else:
            # Calculate average monthly savings from projections
            avg_monthly_savings = sum(projections['predicted_savings']) / len(projections['predicted_savings'])
            
            # Estimate months needed based on average savings rate
            if avg_monthly_savings > 0:
                est_months = round(remaining_amount / avg_monthly_savings)
                
                return {
                    'feasible': True,
                    'time_to_goal': est_months,
                    'time_unit': 'months',
                    'message': f'Based on your saving trends, you can reach your goal in approximately {est_months} months',
                    'current_savings': current_savings,
                    'target_amount': target_amount,
                    'monthly_savings_needed': avg_monthly_savings,
                    'is_estimate': True
                }
            else:
                return {
                    'feasible': False,
                    'message': 'At your current saving rate, this goal is not achievable. Try increasing income or reducing expenses.',
                    'current_savings': current_savings,
                    'target_amount': target_amount
                }
    
    def recommend_budget_adjustments(self):
        """
        Recommend budget adjustments to improve financial health
        
        Returns:
            dict: Dictionary with recommendations
        """
        transactions_df = load_transactions()
        
        if transactions_df.empty:
            return {
                'status': 'insufficient_data',
                'message': 'Need more transaction data to provide recommendations'
            }
            
        # Calculate key financial metrics
        income = transactions_df[transactions_df['type'] == 'income']['amount'].sum()
        expenses = transactions_df[transactions_df['type'] == 'expense']['amount'].sum()
        savings_rate = (income - expenses) / income if income > 0 else 0
        
        # Get expense breakdown by category
        expenses_by_category = transactions_df[transactions_df['type'] == 'expense'].groupby('category')['amount'].sum()
        total_expenses = expenses_by_category.sum()
        
        # Calculate category percentages
        category_percentages = {}
        for category, amount in expenses_by_category.items():
            category_percentages[category] = (amount / total_expenses) * 100 if total_expenses > 0 else 0
            
        # Define recommended percentages for common categories
        recommended_percentages = {
            'Housing': 30,
            'Food': 15,
            'Transportation': 10,
            'Utilities': 10,
            'Healthcare': 10,
            'Entertainment': 5,
            'Shopping': 5,
            'Debt Payment': 10,
            'Other': 5
        }
        
        # Identify categories that exceed recommended percentages
        recommendations = []
        
        for category, percentage in category_percentages.items():
            # Find the closest recommended category
            closest_category = min(recommended_percentages.keys(), 
                                 key=lambda x: abs(recommended_percentages[x] - percentage))
                                 
            recommended_pct = recommended_percentages[closest_category]
            
            if percentage > recommended_pct * 1.2:  # If more than 20% over recommended
                recommendations.append({
                    'category': category,
                    'current_percentage': round(percentage, 1),
                    'recommended_percentage': recommended_pct,
                    'potential_savings': round((percentage - recommended_pct) * total_expenses / 100, 2),
                    'message': f'Consider reducing {category} expenses from {round(percentage, 1)}% to closer to {recommended_pct}% of your budget'
                })
                
        # Overall savings rate recommendation
        if savings_rate < 0.2:  # Less than 20% savings rate
            target_savings_rate = 0.2
            recommendations.append({
                'type': 'savings_rate',
                'current_rate': round(savings_rate * 100, 1),
                'recommended_rate': 20,
                'message': f'Your current savings rate is {round(savings_rate * 100, 1)}%. Aim for at least 20% to improve financial health.'
            })
            
        return {
            'status': 'success',
            'total_income': round(income, 2),
            'total_expenses': round(total_expenses, 2),
            'current_savings_rate': round(savings_rate * 100, 1),
            'recommendations': recommendations
        }
