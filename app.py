"""
Flask web application for Finance Manager with ML-based predictions
"""

import os
import logging
import importlib.util
import base64
import json
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "finance_manager_secret_key")

# Import utility functions
from src.python.utils import (
    load_transactions, save_transaction, get_total_income, 
    get_total_expenses, get_category_breakdown, get_monthly_totals
)

# Import visualization functions
from src.python.visualization import (
    get_category_pie_chart, get_monthly_trend_chart, get_budget_bar_chart,
    get_prediction_chart, get_savings_projection_chart
)

# Import ML prediction models
from src.python.models import (
    ExpensePredictionModel, SavingsPredictionModel, FinancialGoalModel
)

# Try to import C++ finance module
try:
    # Check if compiled module exists
    cpp_module_path = os.path.join('src', 'python', 'finance.so')
    if os.path.exists(cpp_module_path):
        # If the compiled module exists, import it directly
        spec = importlib.util.spec_from_file_location("finance", cpp_module_path)
        if spec and spec.loader:
            finance = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(finance)
            logger.info("Loaded C++ finance module successfully")
        else:
            raise ImportError("Could not load the finance module spec")
    else:
        # Fallback to pure Python implementation if C++ module is not available
        logger.warning("C++ finance module not found, using Python fallback")
        
        # Create a fallback Python module with the same functions
        class PythonFinance:
            @staticmethod
            def calculate_budget(income, expense):
                return income - expense
                
            @staticmethod
            def compound_interest(principal, rate, time, n=12):
                return principal * ((1 + (rate / n)) ** (n * time))
                
            @staticmethod
            def investment_growth(principal, contribution, rate, time, frequency=12):
                r = rate / frequency
                n = time * frequency
                future_value = principal * ((1 + r) ** n)
                
                if contribution > 0:
                    future_value += contribution * (((1 + r) ** n - 1) / r)
                    
                return future_value
                
            @staticmethod
            def mortgage_payment(principal, rate, years):
                monthly_rate = rate / 12
                months = years * 12
                
                return principal * (monthly_rate * ((1 + monthly_rate) ** months)) / \
                       (((1 + monthly_rate) ** months) - 1)
                       
            @staticmethod
            def weighted_expense_average(amounts, weights):
                if len(amounts) != len(weights):
                    raise ValueError("Amounts and weights must have the same length")
                    
                sum_of_weighted_values = sum(a * w for a, w in zip(amounts, weights))
                sum_of_weights = sum(weights)
                
                if sum_of_weights == 0:
                    raise ValueError("Sum of weights cannot be zero")
                    
                return sum_of_weighted_values / sum_of_weights
                
            @staticmethod
            def return_on_investment(initial_investment, final_value):
                if initial_investment <= 0:
                    raise ValueError("Initial investment must be greater than zero")
                    
                return ((final_value - initial_investment) / initial_investment) * 100.0
                
            @staticmethod
            def loan_amortization_schedule(principal, annual_rate, years):
                """
                Generate a complete loan amortization schedule
                
                Args:
                    principal (float): Loan amount
                    annual_rate (float): Annual interest rate (as a decimal)
                    years (int): Loan term in years
                    
                Returns:
                    list: List of lists [payment_number, payment_amount, principal_payment, interest_payment, remaining_balance]
                """
                if principal <= 0 or annual_rate < 0 or years <= 0:
                    raise ValueError("Invalid loan parameters")
                    
                monthly_rate = annual_rate / 12
                months = years * 12
                payment = principal * (monthly_rate * ((1 + monthly_rate) ** months)) / \
                         (((1 + monthly_rate) ** months) - 1)
                         
                schedule = []
                balance = principal
                
                for month in range(1, months + 1):
                    interest = balance * monthly_rate
                    principal_payment = payment - interest
                    balance -= principal_payment
                    
                    # Ensure the final balance is exactly 0
                    if month == months:
                        principal_payment = balance + principal_payment
                        balance = 0
                        
                    schedule.append([month, payment, principal_payment, interest, balance])
                    
                return schedule
                
            @staticmethod
            def net_present_value(rate, cash_flows, initial_investment):
                """
                Calculate Net Present Value (NPV) for a series of cash flows
                
                Args:
                    rate (float): Discount rate (as a decimal)
                    cash_flows (list): List of future cash flows
                    initial_investment (float): Initial investment amount (positive value)
                    
                Returns:
                    float: Net Present Value
                """
                if rate <= -1:
                    raise ValueError("Discount rate must be greater than -100%")
                    
                # Calculate NPV
                npv = -initial_investment
                
                for i, cf in enumerate(cash_flows):
                    npv += cf / ((1 + rate) ** (i + 1))
                    
                return npv
                
            @staticmethod
            def internal_rate_of_return(cash_flows, initial_investment, guess=0.1, tolerance=1e-6, max_iterations=100):
                """
                Calculate Internal Rate of Return (IRR) using numerical method
                
                Args:
                    cash_flows (list): List of future cash flows
                    initial_investment (float): Initial investment amount (positive value)
                    guess (float): Initial guess for IRR (default: 0.1)
                    tolerance (float): Convergence tolerance (default: 1e-6)
                    max_iterations (int): Maximum number of iterations (default: 100)
                    
                Returns:
                    float: Internal Rate of Return as a decimal
                """
                if not cash_flows:
                    raise ValueError("Cash flows cannot be empty")
                    
                # Newton-Raphson method to find IRR
                r = guess
                
                for i in range(max_iterations):
                    # Calculate NPV at current rate
                    npv = -initial_investment
                    npv_prime = 0
                    
                    for j, cf in enumerate(cash_flows):
                        t = j + 1
                        npv += cf / ((1 + r) ** t)
                        npv_prime -= t * cf / ((1 + r) ** (t + 1))
                        
                    # Check if converged
                    if abs(npv) < tolerance:
                        return r
                        
                    # Update rate
                    r = r - npv / npv_prime if npv_prime != 0 else r + 0.01
                    
                # If not converged, return the best guess
                raise RuntimeError(f"IRR calculation did not converge after {max_iterations} iterations")
                
            @staticmethod
            def calculate_financial_ratios(current_assets, total_assets, current_liabilities, 
                                           total_liabilities, equity, net_income, revenue):
                """
                Calculate key financial ratios
                
                Args:
                    current_assets (float): Current assets value
                    total_assets (float): Total assets value
                    current_liabilities (float): Current liabilities value
                    total_liabilities (float): Total liabilities value
                    equity (float): Equity value
                    net_income (float): Net income value
                    revenue (float): Revenue value
                    
                Returns:
                    dict: Dictionary of financial ratios
                """
                ratios = {}
                
                # Liquidity Ratios
                ratios["current_ratio"] = current_assets / current_liabilities if current_liabilities > 0 else 0
                
                # Solvency Ratios
                ratios["debt_to_equity"] = total_liabilities / equity if equity > 0 else 0
                
                # Profitability Ratios
                ratios["roe"] = net_income / equity * 100 if equity > 0 else 0  # Return on Equity (%)
                ratios["roa"] = net_income / total_assets * 100 if total_assets > 0 else 0  # Return on Assets (%)
                ratios["profit_margin"] = net_income / revenue * 100 if revenue > 0 else 0  # Profit Margin (%)
                
                # Efficiency Ratios
                ratios["asset_turnover"] = revenue / total_assets if total_assets > 0 else 0
                
                return ratios
        
        finance = PythonFinance()
        logger.info("Using Python fallback for finance calculations")
        
except Exception as e:
    logger.error(f"Error importing finance module: {e}")
    
    # Define a minimal fallback with all required methods
    class MinimalFinance:
        @staticmethod
        def calculate_budget(income, expense):
            return income - expense
            
        @staticmethod
        def compound_interest(principal, rate, time, n=12):
            return principal * ((1 + (rate / n)) ** (n * time))
            
        @staticmethod
        def investment_growth(principal, contribution, rate, time, frequency=12):
            r = rate / frequency
            n = time * frequency
            future_value = principal * ((1 + r) ** n)
            
            if contribution > 0:
                future_value += contribution * (((1 + r) ** n - 1) / r)
                
            return future_value
            
        @staticmethod
        def mortgage_payment(principal, rate, years):
            monthly_rate = rate / 12
            months = years * 12
            
            return principal * (monthly_rate * ((1 + monthly_rate) ** months)) / \
                   (((1 + monthly_rate) ** months) - 1)
                   
        @staticmethod
        def weighted_expense_average(amounts, weights):
            if len(amounts) != len(weights):
                raise ValueError("Amounts and weights must have the same length")
                
            sum_of_weighted_values = sum(a * w for a, w in zip(amounts, weights))
            sum_of_weights = sum(weights)
            
            if sum_of_weights == 0:
                raise ValueError("Sum of weights cannot be zero")
                
            return sum_of_weighted_values / sum_of_weights
            
        @staticmethod
        def return_on_investment(initial_investment, final_value):
            if initial_investment <= 0:
                raise ValueError("Initial investment must be greater than zero")
                
            return ((final_value - initial_investment) / initial_investment) * 100.0
    
    finance = MinimalFinance()
    logger.warning("Using minimal fallback for finance calculations")

# Initialize ML models
expense_model = ExpensePredictionModel()
savings_model = SavingsPredictionModel()
financial_goal_model = FinancialGoalModel()

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/reset_demo_data')
def reset_demo_data():
    """Reset transaction data with sample demo data"""
    # Remove existing transaction file
    if os.path.exists(os.path.join('data', 'transactions.csv')):
        os.remove(os.path.join('data', 'transactions.csv'))
    
    # Re-initialize with sample data
    from src.python.utils import ensure_data_dir
    ensure_data_dir()
    
    flash('Demo data has been reset successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page with financial overview and ML predictions"""
    # Get financial data
    total_income = get_total_income()
    total_expenses = get_total_expenses()
    
    # Use C++ function for budget calculation
    remaining_budget = finance.calculate_budget(total_income, total_expenses)
    
    # Get expense predictions with ML model
    predictions = {}
    prediction_details = {}
    predicted_total = 0
    
    # Train the model if we have enough data
    if expense_model.train():
        # Get detailed predictions with confidence intervals
        prediction_details = expense_model.predict_next_month()
        
        # For backwards compatibility with existing templates
        for category, details in prediction_details.items():
            predictions[category] = details['amount']
            
        predicted_total = sum(details['amount'] for details in prediction_details.values())
    
    # Get savings projection for next 6 months
    savings_projection = {}
    if savings_model.train():
        savings_projection = savings_model.predict_future_savings(months=6)
    
    # Get budget recommendations
    budget_recommendations = financial_goal_model.recommend_budget_adjustments()
    
    return render_template(
        'dashboard.html',
        total_income=total_income,
        total_expenses=total_expenses,
        remaining_budget=remaining_budget,
        predictions=predictions,
        prediction_details=prediction_details,
        predicted_total=predicted_total,
        savings_projection=savings_projection,
        budget_recommendations=budget_recommendations
    )

@app.route('/transactions')
def transactions():
    """Render the transactions page"""
    # Get transaction data
    transactions_df = load_transactions()
    
    # Sort by date (most recent first)
    if not transactions_df.empty:
        transactions_df = transactions_df.sort_values(by='date', ascending=False)
    
    # Get financial summary
    total_income = get_total_income()
    total_expenses = get_total_expenses()
    remaining_budget = finance.calculate_budget(total_income, total_expenses)
    
    # Get category breakdown for expenses
    category_totals = get_category_breakdown()
    
    return render_template(
        'transactions.html',
        transactions=transactions_df,
        total_income=total_income,
        total_expenses=total_expenses,
        remaining_budget=remaining_budget,
        category_totals=category_totals
    )

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    """Add a new transaction"""
    if request.method == 'POST':
        transaction_type = request.form.get('transaction_type')
        category = request.form.get('category')
        description = request.form.get('description')
        amount = request.form.get('amount')
        
        try:
            # Validate input
            if not all([transaction_type, category, description, amount]):
                flash('All fields are required', 'danger')
                return redirect(url_for('index'))
            
            # Convert amount to float
            amount = float(amount)
            if amount <= 0:
                flash('Amount must be greater than zero', 'danger')
                return redirect(url_for('index'))
            
            # Save transaction
            success = save_transaction(
                category=category,
                description=description,
                amount=amount,
                trans_type=transaction_type
            )
            
            if success:
                flash('Transaction added successfully', 'success')
                
                # Retrain the ML models with new data
                expense_model.train()
                savings_model.train()
            else:
                flash('Error adding transaction', 'danger')
                
        except ValueError:
            flash('Invalid amount value', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            
        return redirect(url_for('index'))

@app.route('/calculate_investment', methods=['POST'])
def calculate_investment():
    """Calculate investment growth using C++ function"""
    if request.method == 'POST':
        try:
            principal = float(request.form.get('principal', 0))
            contribution = float(request.form.get('contribution', 0))
            rate = float(request.form.get('rate', 0)) / 100  # Convert percentage to decimal
            years = float(request.form.get('years', 0))
            
            # Calculate using C++ function
            result = finance.investment_growth(principal, contribution, rate, years)
            
            # Get other financial data for the dashboard
            total_income = get_total_income()
            total_expenses = get_total_expenses()
            remaining_budget = finance.calculate_budget(total_income, total_expenses)
            
            # Get expense predictions with ML model
            predictions = {}
            prediction_details = {}
            predicted_total = 0
            
            if expense_model.is_trained:
                prediction_details = expense_model.predict_next_month()
                
                # For backwards compatibility with existing templates
                for category, details in prediction_details.items():
                    predictions[category] = details['amount']
                    
                predicted_total = sum(details['amount'] for details in prediction_details.values())
            
            # Get savings projection for next 6 months
            savings_projection = {}
            if savings_model.is_trained:
                savings_projection = savings_model.predict_future_savings(months=6)
            
            # Get budget recommendations
            budget_recommendations = financial_goal_model.recommend_budget_adjustments()
            
            return render_template(
                'dashboard.html',
                total_income=total_income,
                total_expenses=total_expenses,
                remaining_budget=remaining_budget,
                investment_result=result,
                predictions=predictions,
                prediction_details=prediction_details,
                predicted_total=predicted_total,
                savings_projection=savings_projection,
                budget_recommendations=budget_recommendations
            )
            
        except ValueError:
            flash('Please enter valid numbers for all fields', 'danger')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f'Error in calculation: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))

@app.route('/calculate_mortgage', methods=['POST'])
def calculate_mortgage():
    """Calculate mortgage payment using C++ function"""
    if request.method == 'POST':
        try:
            loan_amount = float(request.form.get('loan_amount', 0))
            interest_rate = float(request.form.get('interest_rate', 0)) / 100  # Convert percentage to decimal
            loan_term = int(request.form.get('loan_term', 0))
            
            # Calculate using C++ function
            result = finance.mortgage_payment(loan_amount, interest_rate, loan_term)
            
            # Get other financial data for the dashboard
            total_income = get_total_income()
            total_expenses = get_total_expenses()
            remaining_budget = finance.calculate_budget(total_income, total_expenses)
            
            # Get expense predictions with ML model
            predictions = {}
            prediction_details = {}
            predicted_total = 0
            
            if expense_model.is_trained:
                prediction_details = expense_model.predict_next_month()
                
                # For backwards compatibility with existing templates
                for category, details in prediction_details.items():
                    predictions[category] = details['amount']
                    
                predicted_total = sum(details['amount'] for details in prediction_details.values())
            
            # Get savings projection
            savings_projection = {}
            if savings_model.is_trained:
                savings_projection = savings_model.predict_future_savings(months=6)
            
            # Get budget recommendations
            budget_recommendations = financial_goal_model.recommend_budget_adjustments()
            
            # Get amortization schedule if needed
            amortization_schedule = finance.loan_amortization_schedule(loan_amount, interest_rate, loan_term)
            
            return render_template(
                'dashboard.html',
                total_income=total_income,
                total_expenses=total_expenses,
                remaining_budget=remaining_budget,
                mortgage_result=result,
                predictions=predictions,
                prediction_details=prediction_details,
                predicted_total=predicted_total,
                savings_projection=savings_projection,
                budget_recommendations=budget_recommendations,
                amortization_schedule=amortization_schedule[:12]  # Show only first year of payments
            )
            
        except ValueError:
            flash('Please enter valid numbers for all fields', 'danger')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f'Error in calculation: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))

@app.route('/analyze_goal', methods=['POST'])
def analyze_goal():
    """Analyze a financial goal using ML predictions"""
    if request.method == 'POST':
        try:
            goal_amount = float(request.form.get('goal_amount', 0))
            current_savings = float(request.form.get('current_savings', 0))
            
            # Analyze the goal using ML model
            goal_analysis = financial_goal_model.analyze_savings_goal(goal_amount, current_savings)
            
            # Get financial data for the dashboard
            total_income = get_total_income()
            total_expenses = get_total_expenses()
            remaining_budget = finance.calculate_budget(total_income, total_expenses)
            
            # Get predictions
            predictions = {}
            prediction_details = {}
            predicted_total = 0
            
            if expense_model.is_trained:
                prediction_details = expense_model.predict_next_month()
                for category, details in prediction_details.items():
                    predictions[category] = details['amount']
                predicted_total = sum(details['amount'] for details in prediction_details.values())
            
            # Get savings projection
            savings_projection = {}
            if savings_model.is_trained:
                savings_projection = savings_model.predict_future_savings(months=6)
            
            return render_template(
                'dashboard.html',
                total_income=total_income,
                total_expenses=total_expenses,
                remaining_budget=remaining_budget,
                predictions=predictions,
                prediction_details=prediction_details,
                predicted_total=predicted_total,
                savings_projection=savings_projection,
                goal_analysis=goal_analysis
            )
            
        except ValueError:
            flash('Please enter valid numbers for all fields', 'danger')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f'Error analyzing goal: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))

@app.route('/category_chart')
def category_chart():
    """Generate and return category pie chart"""
    img_data = get_category_pie_chart()
    return Response(
        base64.b64decode(img_data),
        mimetype='image/png'
    )

@app.route('/trend_chart')
def trend_chart():
    """Generate and return monthly trend chart"""
    img_data = get_monthly_trend_chart()
    return Response(
        base64.b64decode(img_data),
        mimetype='image/png'
    )

@app.route('/budget_chart')
def budget_chart():
    """Generate and return budget bar chart"""
    img_data = get_budget_bar_chart()
    return Response(
        base64.b64decode(img_data),
        mimetype='image/png'
    )

@app.route('/prediction_chart')
def prediction_chart():
    """Generate and return prediction chart with confidence intervals"""
    # Use the ML model to get predictions
    if expense_model.is_trained:
        prediction_details = expense_model.predict_next_month()
        img_data = get_prediction_chart(prediction_details)
        return Response(
            base64.b64decode(img_data),
            mimetype='image/png'
        )
    else:
        # Return empty response if no prediction data
        return Response('', mimetype='image/png')

@app.route('/savings_projection_chart')
def savings_projection_chart():
    """Generate and return savings projection chart"""
    # Use the ML model to get projections
    if savings_model.is_trained:
        projection = savings_model.predict_future_savings(months=6)
        img_data = get_savings_projection_chart(projection)
        return Response(
            base64.b64decode(img_data),
            mimetype='image/png'
        )
    else:
        # Return empty response if no projection data
        return Response('', mimetype='image/png')

@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    """API endpoint for ML predictions"""
    try:
        if expense_model.is_trained:
            predictions = expense_model.predict_next_month()
            return jsonify({
                'status': 'success',
                'predictions': predictions
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Prediction model is not trained yet'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/savings_projection', methods=['GET'])
def api_savings_projection():
    """API endpoint for savings projections"""
    try:
        # Get months parameter from query string, default to 6
        months = int(request.args.get('months', 6))
        
        if savings_model.is_trained:
            projection = savings_model.predict_future_savings(months=months)
            return jsonify({
                'status': 'success',
                'projection': projection
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Savings model is not trained yet'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Template filter for current year
@app.template_filter('year')
def get_current_year(value):
    """Return the current year for the footer"""
    import datetime
    return datetime.datetime.now().year

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
