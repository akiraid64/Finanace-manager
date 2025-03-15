# Finance Manager App

A personal finance tool to track expenses, plan your budget, and calculate investments. This app uses C++ for all calculations, making it super fast!

![Finance Manager](./generated-icon.png)

## What This App Does

- 📊 **Track your money**: See where your money is going with easy-to-read charts
- 💰 **Budget planning**: Know how much you can spend
- 📈 **Investment calculator**: See how your investments will grow over time
- 🏠 **Mortgage calculator**: Find out your monthly house payments
- 📱 **Easy to use**: Simple, clean interface that's easy to understand

## How to Install and Run (The Easy Way)

You only need Docker installed on your computer. Don't worry if you don't know what that is - it's just a free program that makes installing complicated apps easy! 

### Step 1: Install Docker
* **Windows**: Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
* **Mac**: Download and install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
* **Linux**: Follow the [Docker Engine installation guide](https://docs.docker.com/engine/install/)

### Step 2: Download the Finance Manager

Open a command prompt or terminal and type:

```
git clone https://github.com/yourusername/finance-manager.git
cd finance-manager
```

Don't have git? No problem! Just [download the ZIP file](https://github.com/yourusername/finance-manager/archive/refs/heads/main.zip) and extract it to a folder.

### Step 3: Start the App (One Command!)

In the same terminal window, type:

```
docker-compose up
```

The first time you run this, it might take a few minutes to download everything it needs. Next time will be much faster!

### Step 4: Use the App

Once everything is ready, open your web browser and go to:
```
http://localhost:5000
```

That's it! You're now using the Finance Manager app!

## Features

### 💸 Budget Tracking
* Add income and expenses
* See your spending by category
* Track your remaining budget

### 📊 Financial Charts
* Pie charts for spending categories
* Monthly income vs expense trends
* Budget overview charts

### 🧮 Financial Calculators
* Investment growth calculator
* Mortgage payment calculator
* Return on investment (ROI) calculator
* Loan amortization schedule

### 🔮 Future Planning
* Expense predictions based on your history
* Investment growth projections

## Problems or Questions?

If you run into problems:

1. Make sure Docker is running
2. Try restarting Docker
3. Try the command `docker-compose down` and then `docker-compose up` again

## For Advanced Users

If you want to install without Docker (requires technical knowledge):

1. Make sure you have:
   - Python 3.8 or higher
   - C++ compiler (like GCC)
   - CMake

2. Install dependencies:
   ```
   pip install flask pandas matplotlib scikit-learn pybind11 gunicorn
   ```

3. Build the C++ module:
   ```
   python build_cpp.py
   ```

4. Run the app:
   ```
   python main.py
   ```

## Contributing

Want to help make this app better? Great! Check out our [contribution guidelines](CONTRIBUTING.md).

