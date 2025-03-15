"""
Utility functions for the Finance Manager application.
"""

import os
import pandas as pd
import datetime
import csv
from pathlib import Path

# Define data directory
DATA_DIR = Path(__file__).parent.parent.parent / 'data'
TRANSACTIONS_FILE = DATA_DIR / 'transactions.csv'

def ensure_data_dir():
    """Ensure the data directory exists"""
    DATA_DIR.mkdir(exist_ok=True)
    
    # Create transactions file with headers if it doesn't exist
    if not TRANSACTIONS_FILE.exists():
        with open(TRANSACTIONS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'category', 'description', 'amount', 'type'])
        
        # For demo purposes, initialize with sample transactions
        initialize_sample_data()

def initialize_sample_data():
    """Initialize with sample transaction data for demonstration purposes"""
    # Create past dates for 6 months of transaction history
    today = datetime.datetime.now()
    sample_transactions = []
    
    # Income transactions - Monthly salary
    for i in range(6):
        date = (today - datetime.timedelta(days=30 * i)).strftime('%Y-%m-%d')
        sample_transactions.append({
            'date': date,
            'category': 'Salary',
            'description': 'Monthly salary',
            'amount': 5000.00,
            'type': 'income'
        })
    
    # Additional income transactions
    sample_transactions.append({
        'date': (today - datetime.timedelta(days=45)).strftime('%Y-%m-%d'),
        'category': 'Investments',
        'description': 'Stock dividend',
        'amount': 320.50,
        'type': 'income'
    })
    
    sample_transactions.append({
        'date': (today - datetime.timedelta(days=15)).strftime('%Y-%m-%d'),
        'category': 'Side Business',
        'description': 'Freelance work',
        'amount': 750.00,
        'type': 'income'
    })
    
    # Expense transactions
    expense_categories = ['Housing', 'Food', 'Transportation', 'Utilities', 'Entertainment', 'Healthcare']
    expense_descriptions = {
        'Housing': ['Rent payment', 'Home insurance', 'Property maintenance'],
        'Food': ['Grocery shopping', 'Restaurant dinner', 'Lunch at work'],
        'Transportation': ['Car payment', 'Fuel', 'Public transit', 'Car insurance'],
        'Utilities': ['Electricity bill', 'Water bill', 'Internet service', 'Mobile phone'],
        'Entertainment': ['Movie tickets', 'Concert tickets', 'Streaming services', 'Books'],
        'Healthcare': ['Doctor visit', 'Prescription medication', 'Health insurance', 'Gym membership']
    }
    
    expense_amounts = {
        'Housing': [1500, 1600, 1500],
        'Food': [350, 120, 75],
        'Transportation': [300, 85, 50, 150],
        'Utilities': [120, 60, 80, 65],
        'Entertainment': [25, 80, 30, 20],
        'Healthcare': [100, 40, 200, 50]
    }
    
    # Generate expense transactions over multiple months
    for i in range(6):
        month_date = today - datetime.timedelta(days=30 * i)
        
        for category in expense_categories:
            # Add 1-3 transactions per category per month
            num_transactions = min(len(expense_descriptions[category]), 3)
            for j in range(num_transactions):
                # Vary the amounts slightly each month (±10%)
                base_amount = expense_amounts[category][j % len(expense_amounts[category])]
                variation = (0.9 + 0.2 * (hash(f"{i}-{category}-{j}") % 100) / 100)
                amount = round(base_amount * variation, 2)
                
                # Vary the day of the month
                day_offset = (hash(f"{category}-{j}-{i}") % 28) + 1
                date = (month_date.replace(day=day_offset)).strftime('%Y-%m-%d')
                
                sample_transactions.append({
                    'date': date,
                    'category': category,
                    'description': expense_descriptions[category][j % len(expense_descriptions[category])],
                    'amount': amount,
                    'type': 'expense'
                })
    
    # Save the sample transactions
    df = pd.DataFrame(sample_transactions)
    df.to_csv(TRANSACTIONS_FILE, index=False)

def load_transactions():
    """
    Load transaction data from CSV file
    
    Returns:
        pandas.DataFrame: DataFrame containing transaction data
    """
    ensure_data_dir()
    
    try:
        if not TRANSACTIONS_FILE.exists() or os.path.getsize(TRANSACTIONS_FILE) == 0:
            # Return empty DataFrame with correct columns if file is empty
            return pd.DataFrame(columns=['date', 'category', 'description', 'amount', 'type'])
        
        df = pd.read_csv(TRANSACTIONS_FILE)
        
        # Convert date string to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert amount to float
        df['amount'] = df['amount'].astype(float)
        
        return df
    except Exception as e:
        print(f"Error loading transactions: {e}")
        # Return empty DataFrame if there's an error
        return pd.DataFrame(columns=['date', 'category', 'description', 'amount', 'type'])

def save_transaction(category, description, amount, trans_type):
    """
    Save a new transaction to the CSV file
    
    Args:
        category (str): Transaction category
        description (str): Transaction description
        amount (float): Transaction amount
        trans_type (str): Transaction type ('income' or 'expense')
    
    Returns:
        bool: True if successful, False otherwise
    """
    ensure_data_dir()
    
    try:
        # Create a new row
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        new_transaction = {
            'date': today,
            'category': category,
            'description': description,
            'amount': float(amount),
            'type': trans_type
        }
        
        # Append to existing data
        transactions_df = load_transactions()
        transactions_df = pd.concat([transactions_df, pd.DataFrame([new_transaction])], ignore_index=True)
        
        # Save back to CSV
        transactions_df.to_csv(TRANSACTIONS_FILE, index=False)
        
        return True
    except Exception as e:
        print(f"Error saving transaction: {e}")
        return False

def get_total_income():
    """
    Get the total income from all transactions
    
    Returns:
        float: Total income
    """
    transactions_df = load_transactions()
    income = transactions_df[transactions_df['type'] == 'income']['amount'].sum()
    return income if not pd.isna(income) else 0.0

def get_total_expenses():
    """
    Get the total expenses from all transactions
    
    Returns:
        float: Total expenses
    """
    transactions_df = load_transactions()
    expenses = transactions_df[transactions_df['type'] == 'expense']['amount'].sum()
    return expenses if not pd.isna(expenses) else 0.0

def get_category_breakdown():
    """
    Get a breakdown of expenses by category
    
    Returns:
        dict: Dictionary with categories as keys and total amounts as values
    """
    transactions_df = load_transactions()
    
    # Filter for expenses only
    expenses_df = transactions_df[transactions_df['type'] == 'expense']
    
    # Group by category and sum the amounts
    if not expenses_df.empty:
        category_totals = expenses_df.groupby('category')['amount'].sum().to_dict()
        return category_totals
    else:
        return {}

def get_monthly_totals(months=6):
    """
    Get monthly income and expense totals for the last specified number of months
    
    Args:
        months (int): Number of months to include
    
    Returns:
        tuple: (monthly_income, monthly_expenses) - dictionaries with month as key and total as value
    """
    transactions_df = load_transactions()
    
    if transactions_df.empty:
        return {}, {}
    
    # Ensure date column is datetime
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    
    # Filter for last X months
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30 * months)
    recent_df = transactions_df[(transactions_df['date'] >= start_date) & 
                               (transactions_df['date'] <= end_date)]
    
    # Create a month-year column
    recent_df['month_year'] = recent_df['date'].dt.strftime('%b %Y')
    
    # Group by month-year and type, then sum amounts
    monthly_grouped = recent_df.groupby(['month_year', 'type'])['amount'].sum().unstack()
    
    # Convert to dictionaries
    monthly_income = monthly_grouped.get('income', pd.Series()).to_dict()
    monthly_expenses = monthly_grouped.get('expense', pd.Series()).to_dict()
    
    return monthly_income, monthly_expenses
