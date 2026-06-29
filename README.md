# 🚗 CarDekho Car Price Prediction

A machine learning pipeline that predicts the resale price of used cars from listing data scraped from CarDekho. The model takes raw fields like car name, year, mileage, and engine specs, and outputs an estimated selling price.

## 📊 Dataset

The dataset (`datasets/car_details.csv`) contains 8,128 used car listings with the following columns:

| Column | Description |
|---|---|
| `name` | Full car name (brand + model + variant) |
| `year` | Manufacturing year |
| `selling_price` | Target variable |
| `km_driven` | Total kilometers driven |
| `fuel` | Fuel type (Petrol, Diesel, CNG, LPG) |
| `seller_type` | Individual, Dealer, or Trustmark Dealer |
| `transmission` | Manual or Automatic |
| `owner` | Ownership history (First Owner, Second Owner, etc.) |
| `mileage` | Fuel efficiency (raw string, e.g. "23.4 kmpl") |
| `engine` | Engine displacement (raw string, e.g. "1248 CC") |
| `max_power` | Max power output (raw string, e.g. "74 bhp") |
| `torque` | Torque (raw string, mixed Nm/kgm units) |
| `seats` | Number of seats |

Several columns (`mileage`, `engine`, `max_power`, `torque`, `seats`) contain missing values, which the pipeline handles during preprocessing.

## ⚙️ Pipeline

The project builds a single `scikit-learn` pipeline that takes raw data straight to predictions:

1. **Feature extraction**
   - Splits `name` into `brand` and `car_model`.
   - Converts `year` into `car_age` (current year minus manufacturing year).
   - Strips units from `mileage`, `engine`, `max_power`, and `torque`, converting each to a numeric value (torque also normalizes kgm to Nm).

2. **Imputation**
   - Fills missing `seats` values with the most frequent value.
   - Fills missing `mileage`, `car_age`, `torque`, `max_power`, and `engine` values using `IterativeImputer` with a `RandomForestRegressor` as the estimator.

3. **Encoding**
   - `owner` uses an ordinal encoding (Fourth & Above → Test Drive Car) since ownership history has a natural order.
   - `fuel`, `seller_type`, and `transmission` use one-hot encoding.
   - `car_model` and `brand` use one-hot encoding with infrequent categories grouped together, since both fields have a long tail of rare values.

4. **Transforming numeric features**
   - `engine`, `torque`, and `car_age` get a log transform followed by standard scaling (right-skewed).
   - `km_driven` gets a Yeo-Johnson power transform (heavily right-skewed).
   - `max_power` gets a square transform followed by standard scaling (left-skewed).
   - `mileage` gets standard scaling.

5. **Model**
   - `RandomForestRegressor` trained on the fully transformed feature set.

## 📈 Results

5-fold cross-validation on the full dataset gives an R² score averaging around **0.967**, with individual folds ranging from 0.955 to 0.977.

## 🛠️ Tech Stack

- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Modeling:** `scikit-learn` (`Pipeline`, `ColumnTransformer`, `RandomForestRegressor`, `IterativeImputer`)
- **Serialization:** `pickle`

## 📁 Project Structure

```
.
├── CarDekho.ipynb
├── datasets/
│   └── car_details.csv
├── models/
│   └── Car_price_prediction.pkl
└── README.md
```

## 🚀 Running the Project

1. Clone the repo and install the dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
2. Place `car_details.csv` inside a `datasets/` folder.
3. Run `CarDekho.ipynb` end to end. The notebook trains the pipeline and saves the final model to `models/Car_price_prediction.pkl`.
4. Load the saved model for predictions on new data:
   ```python
   import pickle
   model = pickle.load(open('models/Car_price_prediction.pkl', 'rb'))
   model.predict(new_data)
   ```

## 💡 Possible Improvements

- Compare `RandomForestRegressor` against gradient boosting models (XGBoost, LightGBM).
- Hold out a proper test set for a final, unbiased R² score instead of relying on cross-validation alone.
- Add hyperparameter tuning (`GridSearchCV` or `RandomizedSearchCV`) for the random forest.
