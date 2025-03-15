"""
Visualization utilities for the Finance Manager application.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, timedelta
import seaborn as sns

from src.python.utils import load_transactions, get_category_breakdown, get_monthly_totals

# Set a modern style for charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")

def get_category_pie_chart():
    """
    Generate a pie chart of expenses by category
    
    Returns:
        str: Base64 encoded image data
    """
    category_data = get_category_breakdown()
    
    if not category_data:
        # Return a placeholder empty chart
        plt.figure(figsize=(8, 6))
        plt.title('No Expense Data Available')
        plt.axis('off')
    else:
        # Create pie chart
        plt.figure(figsize=(8, 6))
        plt.pie(
            category_data.values(), 
            labels=category_data.keys(),
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
            explode=[0.05] * len(category_data),  # Slight explode for visual appeal
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        plt.axis('equal')  # Equal aspect ratio ensures pie is circular
        plt.title('Expenses by Category', fontsize=16, pad=20)
        plt.tight_layout()
    
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#F0F0F0')
    plt.close()
    
    # Encode as base64 string
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def get_monthly_trend_chart(months=6):
    """
    Generate a line chart showing income and expenses over time
    
    Args:
        months (int): Number of months to show
        
    Returns:
        str: Base64 encoded image data
    """
    monthly_income, monthly_expenses = get_monthly_totals(months)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    if not monthly_income and not monthly_expenses:
        plt.title('No Monthly Data Available')
        plt.axis('off')
    else:
        # Create a DataFrame for clean plotting
        months = sorted(list(set(list(monthly_income.keys()) + list(monthly_expenses.keys()))))
        
        # Format data for plotting
        income_values = [monthly_income.get(month, 0) for month in months]
        expense_values = [monthly_expenses.get(month, 0) for month in months]
        
        # Plot the data
        plt.plot(months, income_values, 'o-', color='#28a745', linewidth=3, label='Income', markersize=8)
        plt.plot(months, expense_values, 'o-', color='#dc3545', linewidth=3, label='Expenses', markersize=8)
        
        # Fill area between curves
        plt.fill_between(months, income_values, expense_values, 
                         where=(np.array(income_values) > np.array(expense_values)),
                         interpolate=True, alpha=0.2, color='#28a745', label='Savings')
        plt.fill_between(months, income_values, expense_values, 
                         where=(np.array(income_values) < np.array(expense_values)),
                         interpolate=True, alpha=0.2, color='#dc3545', label='Deficit')
        
        # Add labels and title
        plt.title('Monthly Income vs Expenses', fontsize=16, pad=20)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Amount ($)', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.legend(fontsize=10)
        
        # Add dollar signs to y-axis
        formatter = plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # Add annotations for latest values
        if income_values and expense_values:
            plt.annotate(f'${income_values[-1]:,.2f}', 
                         xy=(months[-1], income_values[-1]), 
                         xytext=(10, 10),
                         textcoords='offset points', 
                         fontsize=10, 
                         color='#28a745',
                         fontweight='bold')
            plt.annotate(f'${expense_values[-1]:,.2f}', 
                         xy=(months[-1], expense_values[-1]), 
                         xytext=(10, -15),
                         textcoords='offset points', 
                         fontsize=10, 
                         color='#dc3545',
                         fontweight='bold')
    
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#F0F0F0')
    plt.close()
    
    # Encode as base64 string
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def get_budget_bar_chart():
    """
    Generate a bar chart comparing income and expenses
    
    Returns:
        str: Base64 encoded image data
    """
    transactions_df = load_transactions()
    
    income = transactions_df[transactions_df['type'] == 'income']['amount'].sum()
    expenses = transactions_df[transactions_df['type'] == 'expense']['amount'].sum()
    
    # If no data, return an empty chart
    if pd.isna(income) and pd.isna(expenses):
        plt.figure(figsize=(8, 6))
        plt.title('No Budget Data Available')
        plt.axis('off')
    else:
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Set default values if NaN
        income = income if not pd.isna(income) else 0
        expenses = expenses if not pd.isna(expenses) else 0
        remaining = max(0, income - expenses)
        
        # Create bar chart
        categories = ['Income', 'Expenses', 'Remaining']
        values = [income, expenses, remaining]
        colors = ['#28a745', '#dc3545', '#007bff']  # Green, Red, Blue
        
        bars = plt.bar(categories, values, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
        plt.title('Budget Overview', fontsize=16, pad=20)
        plt.ylabel('Amount ($)', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=10)
        
        # Format y-axis with dollar signs
        formatter = plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (max(values) * 0.02),
                    f'${height:,.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add a legend explaining the colors
        legend_elements = [
            plt.Rectangle((0,0),1,1, color='#28a745', label='Total Income'),
            plt.Rectangle((0,0),1,1, color='#dc3545', label='Total Expenses'),
            plt.Rectangle((0,0),1,1, color='#007bff', label='Remaining Budget')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
    
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#F0F0F0')
    plt.close()
    
    # Encode as base64 string
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def get_prediction_chart(prediction_details):
    """
    Generate a chart visualizing the ML expense predictions with confidence intervals
    
    Args:
        prediction_details (dict): Dictionary with prediction details from ML model
        
    Returns:
        str: Base64 encoded image data
    """
    if not prediction_details:
        # Return a placeholder empty chart
        plt.figure(figsize=(10, 6))
        plt.title('No Prediction Data Available')
        plt.axis('off')
        
    else:
        # Extract data from prediction details
        categories = []
        predicted_amounts = []
        lower_bounds = []
        upper_bounds = []
        historical_avgs = []
        
        for category, details in prediction_details.items():
            categories.append(category)
            predicted_amounts.append(details['amount'])
            lower_bounds.append(details['lower_bound'])
            upper_bounds.append(details['upper_bound'])
            historical_avgs.append(details['historical_avg'])
        
        # Sort by predicted amount (descending)
        sorted_indices = np.argsort(predicted_amounts)[::-1]
        categories = [categories[i] for i in sorted_indices]
        predicted_amounts = [predicted_amounts[i] for i in sorted_indices]
        lower_bounds = [lower_bounds[i] for i in sorted_indices]
        upper_bounds = [upper_bounds[i] for i in sorted_indices]
        historical_avgs = [historical_avgs[i] for i in sorted_indices]
        
        # Calculate error margins for matplotlib's errorbar
        errors_minus = np.array(predicted_amounts) - np.array(lower_bounds)
        errors_plus = np.array(upper_bounds) - np.array(predicted_amounts)
        errors = np.vstack([errors_minus, errors_plus])
        
        # Create the figure
        plt.figure(figsize=(10, 7))
        
        # Create x positions
        x_pos = np.arange(len(categories))
        
        # Plot predicted amounts with error bars (confidence intervals)
        plt.errorbar(x_pos, predicted_amounts, yerr=errors, fmt='o', color='#007bff', 
                     ecolor='#007bff', elinewidth=2, capsize=5, markersize=8, label='Predicted')
        
        # Plot historical averages
        plt.scatter(x_pos, historical_avgs, color='#6c757d', s=80, marker='D', label='Historical Avg')
        
        # Connect the points
        for i in range(len(categories)):
            plt.plot([x_pos[i], x_pos[i]], [predicted_amounts[i], historical_avgs[i]], 
                     'k--', alpha=0.3, linewidth=1)
        
        # Add horizontal grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels and title
        plt.title('ML-Predicted Expenses by Category (With Confidence Intervals)', fontsize=16, pad=20)
        plt.ylabel('Amount ($)', fontsize=12)
        plt.xticks(x_pos, categories, rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        # Add a legend
        plt.legend(fontsize=10, loc='upper right')
        
        # Format y-axis with dollar signs
        formatter = plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # Annotate bars with values
        for i, (pred, hist) in enumerate(zip(predicted_amounts, historical_avgs)):
            plt.text(i, pred + (max(predicted_amounts) * 0.03), f'${pred:.2f}', 
                     ha='center', fontsize=9, fontweight='bold', color='#007bff')
            plt.text(i, hist - (max(predicted_amounts) * 0.05), f'${hist:.2f}', 
                     ha='center', fontsize=9, fontweight='bold', color='#6c757d')
        
        plt.tight_layout()
    
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#F0F0F0')
    plt.close()
    
    # Encode as base64 string
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def get_savings_projection_chart(projection):
    """
    Generate a chart visualizing the ML savings projections
    
    Args:
        projection (dict): Dictionary with savings projection data
        
    Returns:
        str: Base64 encoded image data
    """
    if not projection or 'months' not in projection:
        # Return a placeholder empty chart
        plt.figure(figsize=(10, 6))
        plt.title('No Savings Projection Data Available')
        plt.axis('off')
        
    else:
        # Create figure with two subplots (one for income/expenses, one for savings)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Data for plotting
        months = projection['months']
        incomes = projection['predicted_income']
        expenses = projection['predicted_expenses']
        savings = projection['predicted_savings']
        cumulative = projection['cumulative_savings']
        
        # Plot 1: Income and Expenses
        ax1.plot(months, incomes, 'o-', color='#28a745', linewidth=2.5, label='Predicted Income', markersize=7)
        ax1.plot(months, expenses, 'o-', color='#dc3545', linewidth=2.5, label='Predicted Expenses', markersize=7)
        
        # Fill between the curves
        ax1.fill_between(months, incomes, expenses, 
                        where=(np.array(incomes) > np.array(expenses)),
                        interpolate=True, alpha=0.2, color='#28a745', label='Savings')
        
        # Add labels and formatting for first plot
        ax1.set_title('Future Income and Expense Projections (ML Model)', fontsize=16, pad=20)
        ax1.set_ylabel('Amount ($)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.tick_params(axis='y', labelsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=10, loc='upper left')
        
        # Format y-axis with dollar signs
        formatter = plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        ax1.yaxis.set_major_formatter(formatter)
        
        # Plot 2: Savings and Cumulative Savings
        ax2.bar(months, savings, color='#007bff', label='Monthly Savings', alpha=0.7, width=0.4)
        ax2.plot(months, cumulative, 'D-', color='#6610f2', linewidth=2.5, label='Cumulative Savings', markersize=7)
        
        # Add labels and formatting for second plot
        ax2.set_title('Projected Monthly and Cumulative Savings', fontsize=14, pad=15)
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('Amount ($)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.tick_params(axis='y', labelsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10, loc='upper left')
        
        # Format y-axis with dollar signs
        ax2.yaxis.set_major_formatter(formatter)
        
        # Annotate last points
        ax1.annotate(f'${incomes[-1]:,.2f}', 
                    xy=(months[-1], incomes[-1]), 
                    xytext=(10, 10),
                    textcoords='offset points', 
                    fontsize=10, 
                    color='#28a745',
                    fontweight='bold')
                    
        ax1.annotate(f'${expenses[-1]:,.2f}', 
                    xy=(months[-1], expenses[-1]), 
                    xytext=(10, -15),
                    textcoords='offset points', 
                    fontsize=10, 
                    color='#dc3545',
                    fontweight='bold')
                    
        ax2.annotate(f'${cumulative[-1]:,.2f}', 
                    xy=(months[-1], cumulative[-1]), 
                    xytext=(10, 10),
                    textcoords='offset points', 
                    fontsize=10, 
                    color='#6610f2',
                    fontweight='bold')
        
        plt.tight_layout()
    
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#F0F0F0')
    plt.close()
    
    # Encode as base64 string
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str
