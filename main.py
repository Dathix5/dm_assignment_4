#%% md
# # Data Mining Assignment 4 – Classification
#%% md
# Imports
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
import shap
shap.initjs()

SEED = 99
#%% md
# Datasets
#%%
df_income = pd.read_csv('income.csv')
df_income_test = pd.read_csv('income_test.csv')
df_predictions_template = pd.read_csv('predictions_template.csv')
#%% md
# ## Task 1: Building and Evaluating Classification Models
#%% md
# Inspect dataset
#%%
# missing values
print("missing values")
print(df_income.isnull().sum())
# head
df_income.head()
#%% md
# Fill missing values
#%%
for data in [df_income, df_income_test]:
    # treating NaN as 'No' for birth records
    data['gave birth this year'] = data['gave birth this year'].fillna('No')

    # treating NaN as 'Yes' for english ability
    data['ability to speak english'] = data['ability to speak english'].fillna('Yes')
#%% md
# Encoding of Features
#%%
# map binary features to 1/0
binary_mapping = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}
binary_cols = ['sex', 'ability to speak english', 'gave birth this year']

for col in binary_cols:
    df_income[col] = df_income[col].map(binary_mapping)
    df_income_test[col] = df_income_test[col].map(binary_mapping)

df_income['income'] = df_income['income'].map({'high': 1, 'low': 0})

# one-hot encoding of nominal columns
nominal_cols = ['workclass', 'marital status', 'occupation']
df_income = pd.get_dummies(df_income, columns=nominal_cols)
df_income_test = pd.get_dummies(df_income_test, columns=nominal_cols)

# convert columns to 1/0
fix_cols = df_income.select_dtypes(include=['bool']).columns.tolist() + ['ability to speak english']
df_income[fix_cols] = df_income[fix_cols].fillna(0).astype(int)
df_income_test[fix_cols] = df_income_test[fix_cols].fillna(0).astype(int)
df_income[fix_cols] = df_income[fix_cols].astype(int)
df_income_test[fix_cols] = df_income_test[fix_cols].astype(int)

# show head
df_income.head()
#%% md
# Normalizing
#%%
# initialize the scaler
num_cols = ['age', 'education', 'workinghours']
scaler = StandardScaler()

# fit on training data and transform both
df_income[num_cols] = scaler.fit_transform(df_income[num_cols])
df_income_test[num_cols] = scaler.transform(df_income_test[num_cols])

df_income.head()
#%% md
# Split target
#%%
# define X and y
X = df_income.drop('income', axis=1)
y = df_income['income']

# split into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

print(f"Training features: {X_train.shape}")
print(f"Validation features: {X_val.shape}")
#%% md
# Model 1: Logistic Regression (Baseline)
#%%
# initialize and train
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# check for overfitting
train_preds = log_model.predict(X_train)
val_preds = log_model.predict(X_val)

print(f"Logistic Regression Training Accuracy: {accuracy_score(y_train, train_preds):.4f}")
print(f"Logistic Regression Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")
#%% md
# Model 2: Random Forest (Ensemble + Overfitting Technique)
#%%
# we use GridSearchCV to optimize hyperparameters
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 4]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=SEED), rf_param_grid, cv=3, scoring='accuracy')
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
print(f"Best Parameters: {rf_grid.best_params_}")
#%% md
# Evaluation
#%%
def evaluate_model(model, X_v, y_v, name):
    preds = model.predict(X_v)
    probs = model.predict_proba(X_v)[:, 1]

    print(f"--- Evaluation for {name} ---")
    print(f"Accuracy: {accuracy_score(y_v, preds):.4f}")
    print(f"AUC Score: {roc_auc_score(y_v, probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_v, preds))

# evaluate both models
evaluate_model(log_model, X_val, y_val, "Logistic Regression")
evaluate_model(best_rf, X_val, y_val, "Optimized Random Forest")
#%% md
# Check for overfitting
#%%
# investigating overfitting: comparing train vs validation accuracy
models = [("Logistic Regression", log_model), ("Random Forest", best_rf)]

print("--- OVERFITTING INVESTIGATION ---")
for name, model in models:
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    diff = train_acc - val_acc

    print(f"{name}:")
    print(f"  Training Accuracy:   {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Gap:                 {diff:.4f}")
    print("-" * 30)
#%% md
# ## Task 2: Model Explainability
#%% md
# SHAP values
#%%
# initialize the explainer
explainer = shap.TreeExplainer(best_rf)

# calculate SHAP values
shap_values = explainer.shap_values(X_val)
final_shap_values = shap_values[:, :, 1]

# generate the plot
shap.summary_plot(final_shap_values, X_val)
#%% md
# CHAP waterfall
#%%
# initialize explainer and handle baseline extraction
explainer = shap.TreeExplainer(best_rf)
base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

# plot the first "Low Income" instance (index = 0)
print(f"Person 1 Actual Label: {y_val.iloc[0]}")
exp_low = shap.Explanation(values=final_shap_values[0], base_values=base_val, data=X_val.iloc[0], feature_names=X_val.columns)
shap.plots.waterfall(exp_low)

# fnd and plot the first "High Income" instance
for idx in range(len(X_val)):
    if (final_shap_values[idx].sum() + base_val) >= 0.50:
        print(f"Person 2 Actual Label: {y_val.iloc[idx]}")
        exp_high = shap.Explanation(values=final_shap_values[idx], base_values=base_val, data=X_val.iloc[idx], feature_names=X_val.columns)
        shap.plots.waterfall(exp_high)
        break
#%% md
# ## Task 3: Applying the Model to New Instances
#%% md
# Filling predictions
#%%
# match the exact feature columns our Random Forest expects
X_test = pd.get_dummies(df_income_test)
X_test = X_test.reindex(columns=X_val.columns, fill_value=0)

# generate predictions (0 for <=50k, 1 for >50k)
test_predictions = best_rf.predict(X_test)

# map the 0 and 1 outputs to the required target text format
income_mapping = {0: "low", 1: "high"}
final_preds = [income_mapping[pred] if pred in income_mapping else pred for pred in test_predictions]

# populate the 'income' column in the template and save
df_predictions_template['income'] = final_preds
df_predictions_template.to_csv("predictions_template.csv", index=False)

print(f"Successfully generated predictions for {len(df_predictions_template)} rows")
print("File saved as 'predictions_template.csv'")
#%% md
# Accuracy prediction and gender fairness
#%%
# count how many people are predicted to earn more than $50k
high_income_count = final_preds.count("high")
total_people = len(final_preds)

# calculate gender fairness percentages
male_mask = (X_test['sex'] == 1).values
female_mask = (X_test['sex'] == 0).values

# filter the final_preds list using these masks
male_preds = [final_preds[i] for i in range(total_people) if male_mask[i]]
female_preds = [final_preds[i] for i in range(total_people) if female_mask[i]]

male_selection_rate = male_preds.count("high") / len(male_preds) if len(male_preds) > 0 else 0
female_selection_rate = female_preds.count("high") / len(female_preds) if len(female_preds) > 0 else 0
disparate_impact_ratio = female_selection_rate / male_selection_rate if male_selection_rate > 0 else 0

print(f"1. Total people predicted to earn >$50k: {high_income_count} out of {total_people}")
print(f"2. Percentage of Men predicted as High Income: {male_selection_rate * 100:.2f}%")
print(f"3. Percentage of Women predicted as High Income: {female_selection_rate * 100:.2f}%")