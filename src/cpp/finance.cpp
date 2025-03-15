/**
 * @file finance.cpp
 * @brief Implementation of finance calculations
 * 
 * This file contains high-performance financial calculations for the
 * Finance Manager application. All critical financial operations are
 * implemented in optimized C++ for maximum efficiency.
 */

#include "finance.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace finance {

// Basic budget calculation - optimized for performance
double calculate_budget(double income, double expense) {
    // Direct calculation for maximum performance
    return income - expense;
}

// Optimized compound interest calculation
double compound_interest(double principal, double rate, double time, int n) {
    // Input validation
    if (rate < 0 || time < 0 || n <= 0) {
        throw std::invalid_argument("Invalid input parameters for compound interest calculation");
    }
    
    // Optimized calculation to minimize operations
    const double base = 1.0 + (rate / static_cast<double>(n));
    const double exponent = n * time;
    
    // Use efficient power calculation
    return principal * std::pow(base, exponent);
}

// High-performance investment growth calculation
double investment_growth(double principal, double contribution, double rate, 
                        double time, int frequency) {
    // Thorough input validation
    if (rate < 0) {
        throw std::invalid_argument("Interest rate must be non-negative");
    }
    if (time < 0) {
        throw std::invalid_argument("Time period must be non-negative");
    }
    if (frequency <= 0) {
        throw std::invalid_argument("Contribution frequency must be positive");
    }
    if (principal < 0) {
        throw std::invalid_argument("Initial principal must be non-negative");
    }
    if (contribution < 0) {
        throw std::invalid_argument("Contribution amount must be non-negative");
    }
    
    // Optimized calculation parameters
    const double r = rate / static_cast<double>(frequency);
    const double n = time * frequency;
    
    // Calculate initial investment growth with compound interest
    double future_value = principal * std::pow(1.0 + r, n);
    
    // Calculate future value of regular contributions if applicable
    if (contribution > 0.0) {
        const double growth_factor = std::pow(1.0 + r, n) - 1.0;
        future_value += contribution * (growth_factor / r);
    }
    
    return future_value;
}

// Precise mortgage payment calculation
double mortgage_payment(double principal, double rate, int years) {
    // Enhanced input validation
    if (principal <= 0.0) {
        throw std::invalid_argument("Loan amount must be positive");
    }
    if (rate <= 0.0) {
        throw std::invalid_argument("Interest rate must be positive");
    }
    if (years <= 0) {
        throw std::invalid_argument("Loan term must be positive");
    }
    
    // Calculate monthly parameters
    const double monthly_rate = rate / 12.0;
    const int months = years * 12;
    
    // Optimize calculation to minimize repeated operations
    const double factor = std::pow(1.0 + monthly_rate, months);
    
    // Return monthly payment amount
    return principal * (monthly_rate * factor) / (factor - 1.0);
}

// Advanced weighted expense calculation
double weighted_expense_average(const std::vector<double>& amounts, const std::vector<double>& weights) {
    // Vector size validation
    if (amounts.size() != weights.size()) {
        throw std::invalid_argument("Amounts and weights vectors must have the same size");
    }
    
    // Empty vector handling
    if (amounts.empty()) {
        return 0.0;
    }
    
    // Optimized vector operations using std::inner_product
    double sum_of_weighted_values = std::inner_product(
        amounts.begin(), amounts.end(), weights.begin(), 0.0);
    
    // Calculate sum of weights efficiently
    double sum_of_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    
    // Validation for zero divisor
    if (std::abs(sum_of_weights) < 1e-10) {
        throw std::invalid_argument("Sum of weights cannot be zero");
    }
    
    // Return weighted average
    return sum_of_weighted_values / sum_of_weights;
}

// High-precision ROI calculation
double return_on_investment(double initial_investment, double final_value) {
    // Input validation with detailed error messages
    if (initial_investment <= 0.0) {
        throw std::invalid_argument("Initial investment must be greater than zero");
    }
    
    // Calculate percentage return
    return ((final_value - initial_investment) / initial_investment) * 100.0;
}

// NEW: Loan amortization schedule calculator
std::vector<std::vector<double>> loan_amortization_schedule(double principal, double annual_rate, int years) {
    // Validate inputs
    if (principal <= 0.0 || annual_rate <= 0.0 || years <= 0) {
        throw std::invalid_argument("Invalid loan parameters");
    }
    
    const double monthly_rate = annual_rate / 12.0;
    const int num_payments = years * 12;
    
    // Calculate fixed monthly payment
    const double payment = mortgage_payment(principal, annual_rate, years);
    
    std::vector<std::vector<double>> schedule;
    double remaining_balance = principal;
    
    for (int month = 1; month <= num_payments; ++month) {
        // Calculate interest for this period
        double interest_payment = remaining_balance * monthly_rate;
        
        // Calculate principal for this period
        double principal_payment = payment - interest_payment;
        
        // Update remaining balance
        remaining_balance -= principal_payment;
        
        // Ensure we don't have negative remaining balance due to floating point errors
        if (remaining_balance < 0.01) {
            remaining_balance = 0.0;
        }
        
        // Add this payment to the schedule
        schedule.push_back({
            static_cast<double>(month),  // Payment number
            payment,                     // Payment amount
            principal_payment,           // Principal portion
            interest_payment,            // Interest portion
            remaining_balance            // Remaining balance
        });
    }
    
    return schedule;
}

// NEW: Net Present Value (NPV) calculation
double net_present_value(double rate, const std::vector<double>& cash_flows, double initial_investment) {
    if (rate <= -1.0) {
        throw std::invalid_argument("Discount rate must be greater than -100%");
    }
    
    double npv = -initial_investment;
    
    for (size_t i = 0; i < cash_flows.size(); ++i) {
        npv += cash_flows[i] / std::pow(1.0 + rate, static_cast<double>(i + 1));
    }
    
    return npv;
}

// NEW: Internal Rate of Return (IRR) calculation using numerical method
double internal_rate_of_return(const std::vector<double>& cash_flows, double initial_investment, 
                              double guess = 0.1, double tolerance = 1e-6, int max_iterations = 100) {
    if (cash_flows.empty()) {
        throw std::invalid_argument("Cash flows vector cannot be empty");
    }
    
    // Define NPV function for a given rate
    auto npv_at_rate = [&](double rate) -> double {
        return net_present_value(rate, cash_flows, initial_investment);
    };
    
    // Newton-Raphson method for finding IRR
    double r = guess;
    
    for (int i = 0; i < max_iterations; ++i) {
        double npv = npv_at_rate(r);
        
        // If NPV is close enough to zero, we found the IRR
        if (std::abs(npv) < tolerance) {
            return r;
        }
        
        // Calculate derivative (slope) numerically
        double delta = 0.0001;
        double slope = (npv_at_rate(r + delta) - npv) / delta;
        
        // Avoid division by zero
        if (std::abs(slope) < 1e-10) {
            break;
        }
        
        // Newton-Raphson update
        double r_new = r - npv / slope;
        
        // Check for convergence
        if (std::abs(r_new - r) < tolerance) {
            return r_new;
        }
        
        r = r_new;
    }
    
    throw std::runtime_error("IRR calculation did not converge");
}

// NEW: Financial Ratios calculation
std::unordered_map<std::string, double> calculate_financial_ratios(
    double current_assets, double total_assets, double current_liabilities, 
    double total_liabilities, double equity, double net_income, double revenue) {
    
    std::unordered_map<std::string, double> ratios;
    
    // Current Ratio
    if (current_liabilities == 0.0) {
        ratios["current_ratio"] = std::numeric_limits<double>::infinity();
    } else {
        ratios["current_ratio"] = current_assets / current_liabilities;
    }
    
    // Debt to Equity Ratio
    if (equity == 0.0) {
        ratios["debt_to_equity"] = std::numeric_limits<double>::infinity();
    } else {
        ratios["debt_to_equity"] = total_liabilities / equity;
    }
    
    // Return on Equity (ROE)
    if (equity == 0.0) {
        ratios["roe"] = std::numeric_limits<double>::infinity();
    } else {
        ratios["roe"] = (net_income / equity) * 100.0;
    }
    
    // Return on Assets (ROA)
    if (total_assets == 0.0) {
        ratios["roa"] = std::numeric_limits<double>::infinity();
    } else {
        ratios["roa"] = (net_income / total_assets) * 100.0;
    }
    
    // Profit Margin
    if (revenue == 0.0) {
        ratios["profit_margin"] = std::numeric_limits<double>::infinity();
    } else {
        ratios["profit_margin"] = (net_income / revenue) * 100.0;
    }
    
    // Asset Turnover
    if (total_assets == 0.0) {
        ratios["asset_turnover"] = std::numeric_limits<double>::infinity();
    } else {
        ratios["asset_turnover"] = revenue / total_assets;
    }
    
    return ratios;
}

} // namespace finance

PYBIND11_MODULE(finance, m) {
    m.doc() = "High-performance financial calculation module implemented in C++";
    
    m.def("calculate_budget", &finance::calculate_budget, 
          "Calculate the remaining budget (income minus expense)",
          py::arg("income"), py::arg("expense"));
    
    m.def("compound_interest", &finance::compound_interest,
          "Calculate compound interest with optimal C++ performance",
          py::arg("principal"), py::arg("rate"), py::arg("time"), py::arg("n") = 12);
    
    m.def("investment_growth", &finance::investment_growth,
          "Calculate the future value of an investment with regular contributions",
          py::arg("principal"), py::arg("contribution"), py::arg("rate"), 
          py::arg("time"), py::arg("frequency") = 12);
    
    m.def("mortgage_payment", &finance::mortgage_payment,
          "Calculate monthly mortgage payment with precision C++ floating point",
          py::arg("principal"), py::arg("rate"), py::arg("years"));
    
    m.def("weighted_expense_average", &finance::weighted_expense_average,
          "Calculate weighted average of expenses by category using optimized vector operations");
    
    m.def("return_on_investment", &finance::return_on_investment,
          "Calculate total return on investment as a percentage");
          
    m.def("loan_amortization_schedule", &finance::loan_amortization_schedule,
          "Generate full loan amortization schedule with optimized C++ calculation",
          py::arg("principal"), py::arg("annual_rate"), py::arg("years"));
          
    m.def("net_present_value", &finance::net_present_value,
          "Calculate Net Present Value (NPV) for a series of cash flows",
          py::arg("rate"), py::arg("cash_flows"), py::arg("initial_investment"));
          
    m.def("internal_rate_of_return", &finance::internal_rate_of_return,
          "Calculate Internal Rate of Return (IRR) using numerical method",
          py::arg("cash_flows"), py::arg("initial_investment"), 
          py::arg("guess") = 0.1, py::arg("tolerance") = 1e-6, py::arg("max_iterations") = 100);
          
    m.def("calculate_financial_ratios", &finance::calculate_financial_ratios,
          "Calculate key financial ratios with high-performance C++ implementation",
          py::arg("current_assets"), py::arg("total_assets"), 
          py::arg("current_liabilities"), py::arg("total_liabilities"),
          py::arg("equity"), py::arg("net_income"), py::arg("revenue"));
}
