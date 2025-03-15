/**
 * @file finance.h
 * @brief Header file for high-performance finance calculations
 * 
 * This file contains declarations for highly optimized financial calculations
 * implemented in C++ for maximum performance in the Finance Manager application.
 * 
 * @note All functions are implemented with performance optimizations and thorough error checking
 */

#ifndef FINANCE_H
#define FINANCE_H

#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace finance {

/**
 * Calculate the remaining budget (income minus expense) with optimized C++ performance
 * @param income Total income
 * @param expense Total expense
 * @return Remaining budget
 */
double calculate_budget(double income, double expense);

/**
 * Calculate compound interest using optimized C++ calculation
 * @param principal Initial amount
 * @param rate Interest rate (as a decimal, e.g., 0.05 for 5%)
 * @param time Time period in years
 * @param n Number of times interest is compounded per year
 * @return Final amount after compound interest
 * @throws std::invalid_argument for invalid input parameters
 */
double compound_interest(double principal, double rate, double time, int n = 12);

/**
 * Calculate the future value of an investment with regular contributions
 * High-performance implementation with comprehensive input validation
 * @param principal Initial amount
 * @param contribution Regular contribution amount
 * @param rate Annual interest rate (as a decimal)
 * @param time Time period in years
 * @param frequency Contribution frequency per year (12 for monthly)
 * @return Future value
 * @throws std::invalid_argument for invalid input parameters
 */
double investment_growth(double principal, double contribution, double rate, 
                         double time, int frequency = 12);

/**
 * Calculate monthly mortgage payment with precision C++ floating point operations
 * @param principal Loan amount
 * @param rate Annual interest rate (as a decimal)
 * @param years Loan term in years
 * @return Monthly payment amount
 * @throws std::invalid_argument for invalid loan parameters
 */
double mortgage_payment(double principal, double rate, int years);

/**
 * Calculate weighted average of expenses by category using optimized vector operations
 * @param amounts Vector of expense amounts
 * @param weights Vector of weights (e.g., frequency or importance)
 * @return Weighted average
 * @throws std::invalid_argument if vectors have different sizes or weights sum to zero
 */
double weighted_expense_average(const std::vector<double>& amounts, const std::vector<double>& weights);

/**
 * Calculate total return on investment with high precision
 * @param initial_investment Initial investment amount
 * @param final_value Final investment value
 * @return Return on investment as a percentage
 * @throws std::invalid_argument if initial investment is not positive
 */
double return_on_investment(double initial_investment, double final_value);

/**
 * Generate a complete loan amortization schedule using optimized C++ calculation
 * @param principal Loan amount
 * @param annual_rate Annual interest rate (as a decimal)
 * @param years Loan term in years
 * @return Vector of payment details: [payment_number, payment_amount, principal_payment, interest_payment, remaining_balance]
 * @throws std::invalid_argument for invalid loan parameters
 */
std::vector<std::vector<double>> loan_amortization_schedule(double principal, double annual_rate, int years);

/**
 * Calculate Net Present Value (NPV) for a series of cash flows
 * @param rate Discount rate (as a decimal)
 * @param cash_flows Vector of future cash flows
 * @param initial_investment Initial investment amount (positive value)
 * @return Net Present Value
 * @throws std::invalid_argument if rate is less than or equal to -100%
 */
double net_present_value(double rate, const std::vector<double>& cash_flows, double initial_investment);

/**
 * Calculate Internal Rate of Return (IRR) using numerical method
 * @param cash_flows Vector of future cash flows
 * @param initial_investment Initial investment amount (positive value)
 * @param guess Initial guess for IRR (default: 0.1)
 * @param tolerance Convergence tolerance (default: 1e-6)
 * @param max_iterations Maximum number of iterations (default: 100)
 * @return Internal Rate of Return as a decimal
 * @throws std::invalid_argument if cash flows vector is empty
 * @throws std::runtime_error if the calculation does not converge
 */
double internal_rate_of_return(const std::vector<double>& cash_flows, double initial_investment, 
                              double guess = 0.1, double tolerance = 1e-6, int max_iterations = 100);

/**
 * Calculate key financial ratios with high-performance C++ implementation
 * @param current_assets Current assets value
 * @param total_assets Total assets value
 * @param current_liabilities Current liabilities value
 * @param total_liabilities Total liabilities value
 * @param equity Equity value
 * @param net_income Net income value
 * @param revenue Revenue value
 * @return Unordered map of financial ratios (current_ratio, debt_to_equity, roe, roa, profit_margin, asset_turnover)
 */
std::unordered_map<std::string, double> calculate_financial_ratios(
    double current_assets, double total_assets, double current_liabilities, 
    double total_liabilities, double equity, double net_income, double revenue);

} // namespace finance

#endif // FINANCE_H
