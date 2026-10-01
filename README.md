# 🔋 EV Battery Failure Predictor

An AI/ML application that predicts whether an electric vehicle battery is likely to fail based on vehicle, battery, charging, driving, environmental, and maintenance-related features.

The project uses **XGBoost** for binary classification and includes data preprocessing, exploratory data analysis, SMOTE-based class balancing, model evaluation, and model serialization.

---

## 🏷️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-189FDD)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-8A2BE2)

**Technologies:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Joblib, Streamlit, Plotly, Matplotlib, Seaborn.

---

## 🚀 Features

- 🔋 EV battery failure prediction
- 🤖 XGBoost-based binary classification
- 🧹 Data preprocessing and missing-value handling
- ⚖️ SMOTE-based class balancing
- 📊 Exploratory Data Analysis
- 📈 Model evaluation using multiple metrics
- 🔲 Confusion matrix analysis
- 🧠 Comparison of multiple machine-learning models
- 💾 Trained XGBoost model saved using Joblib
- 🖥️ Streamlit-based application
- 📊 Interactive visualizations where implemented

---

## 🧠 Machine Learning Model

The project evaluates four classification algorithms:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

### Why XGBoost?

XGBoost is a gradient-boosted decision-tree algorithm designed for structured/tabular datasets.

It is used in this project as the final classification model after comparing multiple machine-learning approaches.

The trained model is saved as:

```text
model/xgb_model.pkl
```

The Streamlit application loads the trained model and uses the supplied EV and battery features to generate a prediction.

---

## 📊 Dataset

The original dataset contains:

- **200,000 records**
- **70 columns**
- **59 float columns**
- **1 integer column**
- **10 object/categorical columns**
- Target variable: `battery_failure`

### 🚗 Vehicle Features

- `vehicle_brand`
- `vehicle_model`
- `vehicle_type`
- `manufacturing_year`
- `drive_type`

### 🔋 Battery Features

- `battery_manufacturer`
- `battery_chemistry`
- `battery_capacity_kwh`
- `cycle_count`
- `battery_health_percent`
- `state_of_charge`
- `state_of_health`
- `remaining_capacity`
- `capacity_loss_percent`

### ⚡ Electrical & Thermal Features

- `cell_voltage_avg`
- `cell_voltage_std`
- `pack_voltage`
- `cell_temperature_avg`
- `cell_temperature_max`
- `internal_resistance`
- `thermal_runaway_risk`
- `voltage_imbalance`
- `temperature_variance`

### 🔌 Charging Features

- `charging_cycles_last_month`
- `fast_charge_ratio`
- `slow_charge_ratio`
- `average_charge_power_kw`
- `average_charging_time`
- `overnight_charging_ratio`
- `home_charging_ratio`
- `charging_interruptions`
- `overcharge_events`

### 🛣️ Driving & Environmental Features

- `average_speed`
- `average_trip_distance`
- `aggressive_acceleration_score`
- `hard_braking_score`
- `regenerative_braking_usage`
- `highway_driving_ratio`
- `city_driving_ratio`
- `daily_distance`
- `average_ambient_temperature`
- `maximum_temperature`
- `minimum_temperature`
- `humidity`
- `altitude`
- `terrain_type`
- `dust_exposure`

### 🔧 Maintenance & Battery Condition

- `last_service_days`
- `cooling_system_health`
- `firmware_updates`
- `previous_faults`
- `maintenance_score`
- `sensor_fault_count`
- `BMS_warning_count`
- `abnormal_voltage_events`
- `battery_stress_index`
- `aging_score`
- `thermal_health_score`
- `charging_quality_score`
- `driving_stress_score`
- `predicted_remaining_life_cycles`

The EDA workflow removed:

```python
dropped_cols = [
    'vehicle_id',
    'battery_serial',
    'vehicle_model'
]
```

---

## ⚙️ How It Works

```text
                    ┌───────────────────┐
                    │    User Input     │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Data Preprocessing │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Feature Mapping / │
                    │ Transformation    │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │   XGBoost Model   │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │    Prediction     │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Result /          │
                    │ Visualization     │
                    └───────────────────┘
```

---

## 🧹 Data Preprocessing

The dataset contained missing values across multiple features.

### Categorical Missing Values

Categorical missing values were replaced using the mode:

```python
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

### Numerical Missing Values

Numerical missing values were replaced using the column mean:

```python
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].mean().round(2))
```

After preprocessing, the missing-value check showed zero remaining missing values.

---

## ⚖️ Handling Class Imbalance

The original target distribution was:

```text
battery_failure
0    144167
1     15833
```

Because the target classes were imbalanced, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_sm, y_train_sm = sm.fit_resample(
    X_train,
    y_train
)
```

After SMOTE:

```text
Class 0    144167
Class 1    144167
```

The test set remained separate for evaluation.

---

## 🧪 Model Training

The feature matrix and target variable were separated using:

```python
X = df.drop(columns='battery_failure', axis=1)
y = df['battery_failure']
```

The data was divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The following models were then trained:

```python
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

dtc = DecisionTreeClassifier(
    random_state=42
)

rfc = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

xgc = XGBClassifier(
    eval_metric='logloss'
)
```

---

## 📈 Model Evaluation

The SMOTE-based experiment evaluated the models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Performance Comparison

| Model | Accuracy | ROC-AUC | Class 1 Precision | Class 1 Recall | Class 1 F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 95.01% | 0.9880 | 0.93 | 0.69 | 0.79 |
| Decision Tree | 93.00% | 0.8424 | 0.73 | 0.64 | 0.68 |
| Random Forest | 95.60% | 0.9848 | 0.86 | 0.75 | 0.80 |
| **XGBoost** | **96.52%** | **0.9885** | **0.82** | **0.84** | **0.83** |

### XGBoost Results

```text
Accuracy:  96.52%
ROC-AUC:   0.98846
```

Classification performance:

```text
Class 0
Precision: 0.98
Recall:    0.98
F1-score:  0.98

Class 1
Precision: 0.82
Recall:    0.84
F1-score:  0.83
```

The test set contained **40,000 observations**.

---

## 🔲 Confusion Matrix

The project generates confusion matrices for the evaluated classification models.

```python
ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    ax=ax
)
```

This allows the model predictions to be examined through:

- True Negatives
- False Positives
- False Negatives
- True Positives

---

## 💾 Model Serialization

The final XGBoost model is saved using Joblib:

```python
import joblib

joblib.dump(
    xgc,
    "xgb_model.pkl"
)
```

Expected model location:

```text
model/
└── xgb_model.pkl
```

---

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd YOUR_PROJECT_FOLDER
```

### 3. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser using the local Streamlit URL displayed in the terminal.

---

## 📁 Project Structure

```text
EV-Battery-Failure-Predictor/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   └── xgb_model.pkl
│
├── dataset/
│   └── ev_battery_failure_dataset.csv
│
└── screenshots/
    ├── dashboard.png
    └── prediction.png
```

---

## 🖥️ Streamlit Application

The Streamlit application provides a user-facing interface for entering EV and battery-related information.

The application uses the trained XGBoost model to process the supplied features and generate a battery-failure prediction.

If implemented in the current application, prediction probability and interactive Plotly visualizations can also be displayed.


---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **Scikit-learn** | ML models, preprocessing and evaluation |
| **XGBoost** | Final classification model |
| **Imbalanced-learn** | SMOTE oversampling |
| **Joblib** | Model serialization |
| **Streamlit** | Web application |
| **Plotly** | Interactive visualizations |
| **Matplotlib** | Data visualization |
| **Seaborn** | Exploratory data analysis |

---

## 🔮 Future Improvements

- 🧠 Add SHAP-based model explainability
- 📊 Add feature-importance visualization
- 📈 Add probability calibration
- 🔬 Add cross-validation
- 🤖 Compare additional ML algorithms
- 🔄 Add automated model retraining
- 📡 Add real-time battery monitoring
- 🧹 Add automated data-quality validation
- ☁️ Deploy the Streamlit application
- 📱 Improve responsive UI design

---

## ⚠️ Disclaimer

This project is an educational machine-learning application.

Predictions are based on the available dataset and trained model. They should not be treated as a guaranteed diagnosis of actual EV battery failure or as a replacement for professional battery diagnostics.

---

## 👨‍💻 Author

**Siddhesh Dinkar Mange**

### Connect With Me

- 💻 [GitHub](https://github.com/Siddhesh45687)
- 💼 [LinkedIn](https://www.linkedin.com/in/siddheshmange/)

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
