/**
 * JavaScript for Finance Manager dynamic charts
 */

// Refresh charts periodically
document.addEventListener('DOMContentLoaded', function() {
    // Set up refresh for dashboard if we're on that page
    if (document.getElementById('budget-chart') || 
        document.getElementById('category-chart') || 
        document.getElementById('trend-chart') ||
        document.getElementById('prediction-chart') ||
        document.getElementById('savings-projection-chart')) {
        
        // Refresh charts every 5 minutes
        setInterval(function() {
            refreshCharts();
        }, 300000); // 5 minutes
    }
});

/**
 * Refresh all chart images on the dashboard
 */
function refreshCharts() {
    // Get current timestamp to prevent caching
    const timestamp = new Date().getTime();
    
    // Refresh budget chart
    const budgetChart = document.getElementById('budget-chart');
    if (budgetChart) {
        const budgetSrc = budgetChart.src.split('?')[0];
        budgetChart.src = `${budgetSrc}?t=${timestamp}`;
    }
    
    // Refresh category chart
    const categoryChart = document.getElementById('category-chart');
    if (categoryChart) {
        const categorySrc = categoryChart.src.split('?')[0];
        categoryChart.src = `${categorySrc}?t=${timestamp}`;
    }
    
    // Refresh trend chart
    const trendChart = document.getElementById('trend-chart');
    if (trendChart) {
        const trendSrc = trendChart.src.split('?')[0];
        trendChart.src = `${trendSrc}?t=${timestamp}`;
    }
    
    // Refresh ML prediction chart
    const predictionChart = document.getElementById('prediction-chart');
    if (predictionChart) {
        const predictionSrc = predictionChart.src.split('?')[0];
        predictionChart.src = `${predictionSrc}?t=${timestamp}`;
    }
    
    // Refresh savings projection chart
    const savingsProjectionChart = document.getElementById('savings-projection-chart');
    if (savingsProjectionChart) {
        const savingsSrc = savingsProjectionChart.src.split('?')[0];
        savingsProjectionChart.src = `${savingsSrc}?t=${timestamp}`;
    }
    
    console.log('Charts refreshed at', new Date().toLocaleTimeString());
}

/**
 * Toggle between income and expense form
 */
function toggleTransactionType(type) {
    const incomeForm = document.getElementById('income-form');
    const expenseForm = document.getElementById('expense-form');
    const incomeTab = document.getElementById('income-tab');
    const expenseTab = document.getElementById('expense-tab');
    
    if (type === 'income') {
        incomeForm.classList.remove('d-none');
        expenseForm.classList.add('d-none');
        incomeTab.classList.add('active');
        expenseTab.classList.remove('active');
    } else {
        incomeForm.classList.add('d-none');
        expenseForm.classList.remove('d-none');
        incomeTab.classList.remove('active');
        expenseTab.classList.add('active');
    }
}

/**
 * Validate transaction form before submission
 */
function validateTransactionForm(formId) {
    const form = document.getElementById(formId);
    const amount = form.querySelector('[name="amount"]').value;
    const description = form.querySelector('[name="description"]').value;
    
    if (!amount || isNaN(parseFloat(amount)) || parseFloat(amount) <= 0) {
        alert('Please enter a valid positive amount');
        return false;
    }
    
    if (!description.trim()) {
        alert('Please enter a description');
        return false;
    }
    
    return true;
}

/**
 * Refresh ML expense predictions
 */
function refreshPredictions() {
    const predictionChart = document.getElementById('prediction-chart');
    if (predictionChart) {
        // Show loading indicator
        predictionChart.classList.add('loading-chart');
        
        // Add loading message
        const loadingMsg = document.createElement('div');
        loadingMsg.className = 'chart-loading-message';
        loadingMsg.textContent = 'Refreshing ML predictions...';
        predictionChart.parentNode.appendChild(loadingMsg);
        
        // Get current timestamp to prevent caching
        const timestamp = new Date().getTime();
        
        // Make AJAX request to update ML predictions
        fetch('/api/predictions?' + timestamp)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to refresh predictions');
                }
                return response.json();
            })
            .then(data => {
                console.log('Predictions refreshed successfully');
                
                // Refresh the chart image
                const predictionSrc = predictionChart.src.split('?')[0];
                predictionChart.src = `${predictionSrc}?t=${timestamp}`;
                
                // Remove loading state after a short delay to ensure chart loads
                setTimeout(() => {
                    predictionChart.classList.remove('loading-chart');
                    const loadMsg = predictionChart.parentNode.querySelector('.chart-loading-message');
                    if (loadMsg) loadMsg.remove();
                }, 1000);
                
                // Reload page to reflect updated data in tables
                setTimeout(() => {
                    window.location.href = "#predictions";
                    location.reload();
                }, 1500);
            })
            .catch(error => {
                console.error('Error refreshing predictions:', error);
                alert('Could not refresh ML predictions. Please try again later.');
                
                // Remove loading state
                predictionChart.classList.remove('loading-chart');
                const loadMsg = predictionChart.parentNode.querySelector('.chart-loading-message');
                if (loadMsg) loadMsg.remove();
            });
    }
}

/**
 * Refresh ML savings projections
 */
function refreshSavingsProjection() {
    const savingsChart = document.getElementById('savings-projection-chart');
    if (savingsChart) {
        // Show loading indicator
        savingsChart.classList.add('loading-chart');
        
        // Add loading message
        const loadingMsg = document.createElement('div');
        loadingMsg.className = 'chart-loading-message';
        loadingMsg.textContent = 'Refreshing savings projections...';
        savingsChart.parentNode.appendChild(loadingMsg);
        
        // Get current timestamp to prevent caching
        const timestamp = new Date().getTime();
        
        // Make AJAX request to update savings projections
        fetch('/api/savings_projection?' + timestamp)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to refresh savings projection');
                }
                return response.json();
            })
            .then(data => {
                console.log('Savings projections refreshed successfully');
                
                // Refresh the chart image
                const savingsSrc = savingsChart.src.split('?')[0];
                savingsChart.src = `${savingsSrc}?t=${timestamp}`;
                
                // Remove loading state after a short delay to ensure chart loads
                setTimeout(() => {
                    savingsChart.classList.remove('loading-chart');
                    const loadMsg = savingsChart.parentNode.querySelector('.chart-loading-message');
                    if (loadMsg) loadMsg.remove();
                }, 1000);
                
                // Reload page to reflect updated data in tables
                setTimeout(() => {
                    location.reload();
                }, 1500);
            })
            .catch(error => {
                console.error('Error refreshing savings projections:', error);
                alert('Could not refresh savings projections. Please try again later.');
                
                // Remove loading state
                savingsChart.classList.remove('loading-chart');
                const loadMsg = savingsChart.parentNode.querySelector('.chart-loading-message');
                if (loadMsg) loadMsg.remove();
            });
    }
}

/**
 * Refresh ML budget recommendations
 */
function refreshBudgetRecommendations() {
    // Show loading spinner
    const recommendationsDiv = document.querySelector('.card:has(.fa-lightbulb)');
    if (recommendationsDiv) {
        // Add loading message
        const loadingMsg = document.createElement('div');
        loadingMsg.className = 'alert alert-info';
        loadingMsg.innerHTML = '<i class="fas fa-spin fa-spinner me-2"></i>Refreshing budget recommendations...';
        recommendationsDiv.querySelector('.card-body').prepend(loadingMsg);
        
        // Get current timestamp to prevent caching
        const timestamp = new Date().getTime();
        
        // Make page reload request
        setTimeout(() => {
            location.reload();
        }, 1500);
    }
}
