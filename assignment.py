import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load the files
trades_df = pd.read_csv('historical_data.csv')
sentiment_df = pd.read_csv('fear_greed_index.csv')
#convert into prper format
trades_df['time'] = pd.to_datetime(trades_df['Timestamp IST'], dayfirst=True) 
trades_df['date'] = trades_df['time'].dt.normalize()
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# Merge datasets on the Date column
merged_df = pd.merge(trades_df, sentiment_df, on='date', how='inner')
merged_df['is_profitable'] = merged_df['Closed PnL'] > 0

print("--- Performance by Market Sentiment ---")

sentiment_analysis = merged_df.groupby('classification').agg(
    total_pnl=('Closed PnL', 'sum'),
    average_pnl=('Closed PnL', 'mean'),
    win_rate=('is_profitable', 'mean'),
    trade_count=('Account', 'count') 
).reset_index()

print(sentiment_analysis)

#Visualization for the Final Report
plt.figure(figsize=(10, 6))
sns.barplot(data=sentiment_analysis, x='classification', y='average_pnl', palette='coolwarm')
plt.title('Average Trade PnL by Market Sentiment')
plt.ylabel('Average Closed PnL')
plt.xlabel('Market Sentiment')
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

print("\n--- Predictive Modeling: Trade Profitability ---")

#Select Features and Target
features = ['value'] 
X = merged_df[features].dropna()
y = merged_df.loc[X.index, 'is_profitable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

if len(features) > 1:
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:\n", importances)
