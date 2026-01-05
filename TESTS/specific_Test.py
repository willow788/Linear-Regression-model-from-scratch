# Run only unit tests for linear regression
pytest tests/test_linear_regression.py

# Run only integration tests
pytest tests/test_integration.py

# Run specific test class
pytest tests/test_linear_regression.py::TestLinearRegressionFit

# Run specific test function
pytest tests/test_linear_regression.py::TestLinearRegressionFit::test_batch_gradient_descent_fit
