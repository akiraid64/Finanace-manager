# Contributing to Finance Manager

Thank you for your interest in improving the Finance Manager app! This guide will help you get started.

## Easy Ways to Contribute

Even if you're not a programmer, you can help:

- **Report bugs**: If something doesn't work right, let us know by creating an issue
- **Suggest features**: Have an idea to make the app better? Create an issue with your suggestion
- **Improve documentation**: Help make instructions clearer

## For Developers

### Setting Up for Development

1. Fork the repository on GitHub
2. Clone your fork to your local machine
3. Install dependencies:
   ```
   pip install flask pandas matplotlib scikit-learn pybind11 gunicorn
   ```
4. Build the C++ module:
   ```
   python build_cpp.py
   ```

### Making Changes

1. Create a new branch for your changes
2. Make your changes to the code
3. Test your changes thoroughly
4. Commit your changes with a clear message
5. Push to your fork
6. Submit a pull request

### C++ Development

The core financial calculations are implemented in C++ for speed. If you're modifying these:

1. Edit files in `src/cpp/` directory
2. Run `python build_cpp.py` to rebuild the module
3. Test thoroughly to ensure calculations remain accurate

### Python Development

The web interface and data handling are implemented in Python:

1. Follow Flask best practices for route handlers
2. Use pandas for data manipulation
3. Follow the existing code style

## Code Style

- Use consistent indentation (4 spaces for Python, 2 spaces for C++)
- Add comments for complex logic
- Write clear function and variable names
- Include docstrings for Python functions

## Testing

Before submitting your changes:

1. Test all affected features manually
2. Ensure the app works on different browsers
3. Make sure the C++ module builds successfully

Thank you for contributing to Finance Manager!