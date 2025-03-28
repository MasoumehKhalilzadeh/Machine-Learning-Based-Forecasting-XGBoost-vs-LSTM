# 📊 Machine Learning-Based Forecasting: XGBoost vs LSTM

## 📄 Project Overview
In this project, we applied two machine learning techniques, **XGBoost** and **LSTM**, to forecast weekly retail sales. The goal was to explore how tree-based and deep learning models handle time series forecasting, compare their performance using metrics such as RMSE and MAPE, and visualize the accuracy of their predictions.

---

## 📅 Dataset Description
We used a simplified time series dataset based on weekly aggregated sales from Walmart's historical sales data. Each row represents total sales for a specific week.

![1](https://github.com/user-attachments/assets/c957223e-a7b5-4e36-a851-18727cbfb328)


**Features engineered:**
- `lag_1`, `lag_2`: previous weeks' sales
- `rolling_mean_3`: rolling average of 3 weeks
- Time-based features: `year`, `month`, `week`

---

## ⚙️ Project Steps

### 1. **Data Preparation & Feature Engineering**
- Converted dates into usable time features
- Created lag variables and rolling averages
- Scaled data for LSTM compatibility

### 2. **Model 1: XGBoost Regressor**
- Input: Engineered features (lags, time)
- Trained on 80% of the data
- Forecasted sales for the final 20%

**XGBoost Results:**
- **RMSE:** ~1,559,039  
- **MAPE:** ~2.70%

![3](https://github.com/user-attachments/assets/7d728a43-456a-4722-9ef8-92e6e107f186)

#### 🔍 Interpretation:
XGBoost captured sales trends effectively and performed very well on tabular features. Its low MAPE and RMSE indicate highly accurate short-term forecasts.

---

### 3. **Model 2: LSTM Neural Network**
- Input: 4-week time windows from scaled series
- Used Keras LSTM with 1 hidden layer
- Trained for 20 epochs

**LSTM Results:**
- **RMSE:** ~1,817,851  
- **MAPE:** ~3.04%

![2](https://github.com/user-attachments/assets/123967d8-6151-4555-b7ac-f13124d268e8)


#### 🔍 Interpretation:
LSTM also performed well and followed the general sales trend, but with slightly higher error compared to XGBoost. With further tuning and feature expansion, LSTM can be improved for sequence learning tasks.

---

## 📊 Evaluation Metrics

| Model     | RMSE        | MAPE     |
|-----------|-------------|----------|
| XGBoost   | ~1.56M      | 2.70%    |
| LSTM      | ~1.82M      | 3.04%    |

> XGBoost outperformed LSTM in this forecasting task, delivering lower error rates while being faster to train on structured features.


---


**Tools used:** Python, Pandas, XGBoost, TensorFlow/Keras, Matplotlib, Scikit-learn

---

## 🧠 Conclusion & Business Insight

This project gave us a hands-on look at how machine learning models like **XGBoost** and **LSTM** can be used to forecast retail sales. Both models were able to follow the general sales trends, but **XGBoost came out on top** with a lower error and more accurate predictions overall.

With a **MAPE of just 2.70%**, XGBoost proved to be a great choice for structured, short-term sales forecasting. It handled lag features and calendar-based patterns really well. LSTM, while slightly less accurate, still showed potential—especially for modeling sequences over time. With more tuning or richer features (like holidays or promotions), it could improve even more.

> 💡 **Bottom line:** If you're working with structured sales data and need quick, reliable forecasts, XGBoost is a solid and efficient choice. But if your data is more complex or deeply time-dependent, LSTM could offer value with the right setup.

This project helped clarify not just the technical differences between the models, but when and why you'd choose one over the other in real-world forecasting scenarios.
