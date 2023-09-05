# Fake News Detection Using Python and Machine Learning

This repository contains Python code for text classification using the Naive Bayes and Passive Aggressive Classifier algorithms. The code is designed to work with a dataset provided in a CSV file named `data.csv`. The dataset contains text data in the "Body" column and corresponding labels in the "Label" column.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and handling.
- `matplotlib`: For creating plots and visualizations.
- `sklearn`: Scikit-learn library for machine learning tasks.

You can install these libraries using `pip` or `conda`.

```
pip install numpy pandas matplotlib scikit-learn nltk textblob
```

## Usage

1. Clone this repository or download the code to your local machine.
2. Place the `data.csv` file containing your dataset in the same directory as the code.
3. Run the code using your preferred Python environment (e.g., Jupyter Notebook, VSCode, or a Python IDE).

## Code Structure

The code is organized as follows:

1. **Importing Libraries**: The necessary libraries are imported to perform various tasks.

2. **Data Loading and Exploration**:
   - The dataset is loaded from `data.csv`.
   - Various information about the dataset is printed, such as the first few rows, shape, summary statistics, and data types.

3. **Data Preprocessing**:
   - The target variable `y` (labels) is separated from the feature dataset `df`.
   - Missing values in the "Body" column are filled with empty strings.
   - Missing label values are filled with empty strings.
   - The dataset is split into training and testing sets using a 70/30 split ratio.

4. **Feature Extraction**:
   - Text data is vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
   - The vectorizer is fit to the training data and then transformed for both training and testing data.
   - Feature names are obtained from the vectorizer.
   - A DataFrame is created to visualize the extracted features.

5. **Model Training**:
   - A Passive Aggressive Classifier model is trained on the training data.
   - The `max_iter` parameter controls the number of iterations for training.

6. **Prediction and Accuracy Score**:
   - The model predicts labels for the testing data.
   - The accuracy of the model is calculated and printed.

## License

This code is provided under the MIT License. You are free to use, modify, and distribute it as needed. See the [LICENSE](LICENSE) file for more details.
