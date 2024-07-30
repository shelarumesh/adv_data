import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score,confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import pickle


df = pd.read_csv('data/Advertising.csv')

X = df[['TV','Radio','Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)



def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    
    return mse, r2, cv_mse
        

# Initialize models
linear_regression = LinearRegression()
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Train and evaluate models
models = {
    'Linear Regression': linear_regression,
    'Lasso': lasso,
    'Ridge': ridge,
    'Elastic Net': elastic_net
}

results = {}
for model_name, model in models.items():
    mse, r2, cv_mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    results[model_name] = {'MSE': mse, 'R2': r2, 'CV_MSE': cv_mse}

# Display results
for model_name, metrics in results.items():
    print(f"{model_name} - MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}, CV_MSE: {metrics['CV_MSE']:.4f}")


# Save result
result = pd.DataFrame(results)
result.to_csv('model training/model_result.csv')
print(result)

EN = ElasticNet(alpha=1.0, l1_ratio=0.5)
EN.fit(X_train, y_train)

# Save in pickle
with open('model training/EN_Model.pickle','wb') as file:
    data = pickle.dump(EN,file)

