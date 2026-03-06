# Indian House Price Classification

A machine learning project that predicts whether an Indian residential property falls in a Low, Medium, or High price category based on features like location, size, amenities, and market indicators. Built with a full MLOps pipeline and deployed as an interactive Streamlit web app.

---

## What This Project Does

The model takes property details as input and classifies it into one of three price categories:

- **Low** - below ₹100 Lakhs
- **Medium** - between ₹100 and ₹300 Lakhs
- **High** - above ₹300 Lakhs

The best model (Gradient Boosting) achieves 99.96% accuracy on the test set.

---

## Dataset

The dataset contains 250,000 Indian residential property records across 20 states and 42 cities. It has 22 input features including BHK, size, price per square foot, furnishing status, location, amenities, nearby schools and hospitals, floor number, and owner type. There are no missing values.

---

## Project Structure

The project is organised into the following folders:

- **data/** - raw CSV dataset and processed output
- **notebooks/** - EDA notebook with 67 cells of analysis and commentary
- **src/** - all ML source modules (feature engineering, preprocessing, training, evaluation, prediction)
- **pipeline/** - end-to-end training and inference pipelines
- **models/** - saved trained model
- **config/** - YAML configuration file for all settings
- **artifacts/** - auto-generated evaluation plots and metrics

---

## How to Run

**Step 1 - Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/house-price-classification-mlops.git
cd house-price-classification-mlops
```

**Step 2 - Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3 - Add the dataset**

Place `india_housing_prices.csv` inside the `data/raw/` folder.

**Step 4 - Train the model**

```bash
python pipeline/training_pipeline.py
```

---

## Model Results

Four models were trained and compared:

- Gradient Boosting - 99.96% accuracy
- Random Forest - 92.67% accuracy
- Logistic Regression - 85.50% accuracy
- XGBoost - install separately via pip

Gradient Boosting was selected as the best model. The large gap between Gradient Boosting and Logistic Regression exists because the relationship between features and price is non-linear. Tree-based models capture this naturally while linear models cannot.

---

## Feature Engineering

Three new features are derived before training:

- **Property_Age** - calculated from the year the property was built
- **Price_per_BHK** - price divided by number of bedrooms
- **Amenities_Count** - number of amenities counted from the amenities column

These engineered features, especially Price_per_BHK and Price_per_SqFt, turned out to be the strongest predictors in the model.

---

## Evaluation

The following diagnostic plots are automatically generated and saved to the artifacts folder after training:

- Confusion matrix with raw counts and row-normalised percentages
- Model comparison across all four metrics
- Feature importance with cumulative importance curve
- Learning curves showing training vs validation accuracy
- Prediction probability distribution per class
- Error analysis showing only misclassifications

---

## Tech Stack

- Python 3.9+
- scikit-learn for the ML pipeline
- XGBoost for gradient boosted trees
- pandas and numpy for data handling
- matplotlib and seaborn for visualisations
- joblib for model saving
- PyYAML for configuration

---

## License

MIT License
