This project includes:
model.py: A lightweight CNN model for MNIST with less than 25,000 parameters
train.py: Training script that trains the model for one epoch
test_model.py: Tests to verify parameter count and accuracy
.github/workflows/model_tests.yml: GitHub Actions workflow to run tests
requirements.txt: Required dependencies
The model architecture is designed to be lightweight while maintaining high accuracy:
Uses only 8 filters in first conv layer and 16 in second
Small fully connected layers (32 neurons in hidden layer)
Dropout for regularization
MaxPooling to reduce parameters
To use this project:
1. Create a new GitHub repository
Push these files to the repository
GitHub Actions will automatically run the tests on push
The tests will verify:
The model has less than 25,000 parameters
The model achieves >95% accuracy in one epoch