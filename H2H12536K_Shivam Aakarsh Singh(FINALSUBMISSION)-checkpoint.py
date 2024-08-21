#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("Final.csv")
print(df)


# In[2]:


df.describe()


# In[3]:


shape=df.shape
print("The numbers of row = ",shape[0])
print("The numbers of coloumn = ",shape[1])


# In[4]:


df.info()


# In[5]:


n=df.isnull().sum()
print("The number of null values are: ",n)


# In[6]:


import numpy as np
df["PURCHASE_ORDER_TYPE"]=df["PURCHASE_ORDER_TYPE"].fillna(np.nan)
df["CREDIT_STATUS"]=df["CREDIT_STATUS"].fillna(np.nan)


# In[7]:


df["ORDER_CREATION_DATE"] = pd.to_datetime(df["ORDER_CREATION_DATE"], format="%Y%m%d")
print(df["ORDER_CREATION_DATE"])


# In[8]:


df["REQUESTED_DELIVERY_DATE"] = pd.to_datetime(df["REQUESTED_DELIVERY_DATE"], format="%Y%m%d")
print(df["REQUESTED_DELIVERY_DATE"])


# In[9]:


mask = df["ORDER_CREATION_DATE"] > df["REQUESTED_DELIVERY_DATE"]
count = mask.sum()
print("Number of records with order date greater than delivery date:", count)


# In[10]:


df=df[df["ORDER_CREATION_DATE"] <= df["REQUESTED_DELIVERY_DATE"]]
print(df)


# In[11]:


count = df["ORDER_AMOUNT"].str.contains("-").sum()
print("The number of records with - is ", count)


# In[12]:


df["ORDER_AMOUNT"] = df["ORDER_AMOUNT"].str.replace("-","")
print(df["ORDER_AMOUNT"])


# In[13]:


count = df["ORDER_AMOUNT"].str.contains(",").sum()
print("The number of records with , is ", count)


# In[14]:


df["ORDER_AMOUNT"] = df["ORDER_AMOUNT"].str.replace(",",".")
print(df["ORDER_AMOUNT"])


# In[15]:


sol = (df["ORDER_CREATION_DATE"].dt.date == df["REQUESTED_DELIVERY_DATE"].dt.date).sum()
print(sol)


# In[16]:


currency_counts = df["ORDER_CURRENCY"].value_counts()
print(currency_counts)


# In[17]:


conversion_rates = {
    'USD': 1.0,
    'EUR': 1.08,
    'AUD': 0.66,
    'CAD': 0.74,
    'GBP': 1.24,
    'MYR': 0.22,
    'PLN': 0.24,
    'AED': 0.27,
    'HKD': 0.13,
    'CHF': 1.11,
    'RON': 0.22,
    'SGD': 0.74,
    'CZK': 0.046,
    'HU1': 0.0028,
    'NZD': 0.61,
    'BHD': 2.65,
    'SAR': 0.27,
    'QAR': 0.27,
    'KWD': 3.25,
    'SEK': 0.094
}

df['amount_in_usd'] = df['ORDER_AMOUNT'].astype(float)* df['ORDER_CURRENCY'].map(conversion_rates)

        
print(df['amount_in_usd'])


# In[18]:


zero_count = (df['amount_in_usd'] == 0).sum()
print("Count of '0' values in the 'amount_in_usd' column:", zero_count)


# In[19]:


df['unique_cust_id'] = df['CUSTOMER_NUMBER'].astype(str) + '_' + df['COMPANY_CODE'].astype(str)
print(df['unique_cust_id'])


# In[20]:


print(df.columns)


# 
# Milestone 2-
# ========
# 

# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

top_n = 10
channel_counts = df['DISTRIBUTION_CHANNEL'].value_counts()
top_channels = channel_counts.head(top_n)
other_channels = channel_counts.iloc[top_n:].sum()
filtered_df = df[df['DISTRIBUTION_CHANNEL'].isin(top_channels.index)]
other_channels_df = df[~df['DISTRIBUTION_CHANNEL'].isin(top_channels.index)].copy()
other_channels_df.loc[:, 'DISTRIBUTION_CHANNEL'] = 'Others'
filtered_df = pd.concat([filtered_df, other_channels_df])
plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_df, x='DISTRIBUTION_CHANNEL', color='green', discrete=True)
plt.xlabel('\nDistribution Channel')
plt.ylabel('Count')
plt.title('Distribution of Orders by Channel')
plt.xticks(rotation=90)
plt.ylim(0, len(top_channels)*2500)
plt.show()


# In[22]:


currency_counts = df["ORDER_CURRENCY"].value_counts()
plt.pie(currency_counts, labels=currency_counts.index, autopct="%1.1f%%", startangle=90, colors=["skyblue", "pink", "lightgreen", "orange"])
plt.title("Distribution of Order Currency")
plt.legend(title="Currency")
plt.axis("equal")
fig = plt.gcf()
fig.set_size_inches(8, 8)
plt.rcParams["font.size"] = 8
plt.show()


# In[23]:


top_purchase_order_types = df['PURCHASE_ORDER_TYPE'].value_counts().head(10).index
top_distribution_channels = df['DISTRIBUTION_CHANNEL'].value_counts().head(10).index

df_filtered = df[df['PURCHASE_ORDER_TYPE'].isin(top_purchase_order_types) & df['DISTRIBUTION_CHANNEL'].isin(top_distribution_channels)]

grouped_data = df_filtered.groupby(['PURCHASE_ORDER_TYPE', 'DISTRIBUTION_CHANNEL']).size().reset_index(name='COUNT')

pivot_data = grouped_data.pivot(index='PURCHASE_ORDER_TYPE', columns='DISTRIBUTION_CHANNEL', values='COUNT')

pivot_data.plot(kind='line', marker='o', figsize=(10, 6))

plt.xlabel('Purchase Order Type')
plt.ylabel('Count')
plt.title('Top Purchase Order Types vs. Top Distribution Channels')

plt.show()


# In[24]:


df['amount_in_usd'] = pd.to_numeric(df['amount_in_usd'], errors='coerce')
grouped_data = df.groupby('ORDER_CREATION_DATE')['amount_in_usd'].sum()
grouped_data.plot(kind='line', marker='o', color='red')
plt.ylabel('Amount in Dollars')
plt.xlabel('ORDER_CREATION_DATE')
plt.title('Order Amount and USD')
plt.show()










# In[25]:


df['ORDER_AMOUNT'] = pd.to_numeric(df['ORDER_AMOUNT'], errors='coerce') 
sns.boxplot(x=df["ORDER_AMOUNT"]) 
plt.title("Boxplot of Order Amount") 
plt.xlabel("Order Amount") 
plt.show()


# In[26]:



grouped_data = df.groupby('COMPANY_CODE')['ORDER_AMOUNT'].sum().reset_index()
sorted_data = grouped_data.sort_values('ORDER_AMOUNT', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='COMPANY_CODE', y='ORDER_AMOUNT', data=sorted_data)
plt.xlabel('\nCOMPANY_CODE')
plt.ylabel('Order Amount')
plt.title('Total Order Amount by Company')
plt.xticks(rotation='vertical')
plt.show()


# Milestone-3 

# Q1- Check for the outliers in the “amount_in_usd” column and replace the outliers with appropriate values, discussed in the sessions.

# In[27]:


# we will be using IQR method to find and delete the outliers 


#finding the amount of outliers in amount_in_usd coloumn


import numpy as np
data = np.array(df['amount_in_usd'])
# assign your quartiles, limits and iq3
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5*iqr
upper_bound = q3 + 1.5*iqr
#create conditions to isolate the outliers
outliers = data[(data < lower_bound) | (data > upper_bound)]
len(outliers)




# OBSERVATION
# 1.number of outliers is aprroximately 10% of the dataset 
# 2.we are going to delete the outliers


# Finding the upper limt and lower limt based on IQR 
percentile25 = df['amount_in_usd'].quantile(0.25)
percentile75 = df['amount_in_usd'].quantile(0.75)
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr


# triming and plotting the outliers

new_df = df[df['amount_in_usd'] < upper_limit]
new_df.shape
plt.boxplot(new_df['amount_in_usd'])
plt.show()



#capping the remaining outliers
new_df_cap = df.copy()
new_df_cap['amount_in_usd'] = np.where(
    new_df_cap['amount_in_usd'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['amount_in_usd'] < lower_limit,
        lower_limit,
        new_df_cap['amount_in_usd']))


#plotting before and after outlier removal
plt.figure(figsize=(16,6))
plt.subplot(2,2,1)
sns.distplot(df['amount_in_usd'])
plt.subplot(2,2,2)
sns.boxplot(df['amount_in_usd'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['amount_in_usd'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['amount_in_usd'])
plt.show()





# Q2.Label encoding or One hot Encoding on all the categorical columns.

# In[28]:


from sklearn.preprocessing import LabelEncoder

categorical_columns = ['DISTRIBUTION_CHANNEL', 'DIVISION', 'PURCHASE_ORDER_TYPE', 'CREDIT_CONTROL_AREA', 'ORDER_CURRENCY']

label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print(df.head(3))


# Q3-Log Transformations on continuous columns 

# In[29]:



continuous_columns = ['DISTRIBUTION_CHANNEL', 'PURCHASE_ORDER_TYPE', 'CREDIT_STATUS']

# Check if selected columns have zero or negative values
for column in continuous_columns:
    if (df[column] <= 0).any():
        print(f"Column {column} contains zero or negative values.")

# Apply log transformation
for column in continuous_columns:
    if (df[column] > 0).all():
        df[column] = np.log1p(df[column])

print(df.head(3))


# Q.4-Try to extract new features by grouping existing columns 

# In[30]:


from datetime import date
import pandas as pd

# Calculate the number of days for delivery
df['No_of_Days_for_Delivery'] = (df['REQUESTED_DELIVERY_DATE'] - df['ORDER_CREATION_DATE']).dt.days

# Convert 'RELEASED_CREDIT_VALUE' and 'CREDIT_STATUS' to numeric types
df['RELEASED_CREDIT_VALUE'] = pd.to_numeric(df['RELEASED_CREDIT_VALUE'], errors='coerce')
df['CREDIT_STATUS'] = pd.to_numeric(df['CREDIT_STATUS'], errors='coerce')

# Fill missing values with the mean or any other strategy
df['RELEASED_CREDIT_VALUE'].fillna(df['RELEASED_CREDIT_VALUE'].mean(), inplace=True)
df['CREDIT_STATUS'].fillna(df['CREDIT_STATUS'].mean(), inplace=True)

# Calculate the 'released_credit' column
df['released_credit'] = df['RELEASED_CREDIT_VALUE'] / df['CREDIT_STATUS']

# Select the desired columns
selected_columns = df[['REQUESTED_DELIVERY_DATE', 'ORDER_CREATION_DATE', 'No_of_Days_for_Delivery', 'released_credit']]

print(selected_columns.head(3))


# Q.5-Create a heatmap to find correlation between the columns

# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Q-6.Try to identify important or relevant columns for feature extraction

# In[32]:


corr_matrix = df.corr().abs()
high_corr_var = np.where(corr_matrix > 0.7)
high_corr_var = [(corr_matrix.columns[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]
print(high_corr_var)


# MILESTONE 4

# Q1-  Modify the dataset to pass into any type of machine learning models. 

# In[33]:


# Log transformation on "amount_in_usd" column
df['amount_in_usd_log'] = np.log1p(df['amount_in_usd'])


# In[39]:



column_names = df.columns

# Printing the column names
for column in column_names:
    print(column)


# In[41]:


def difference_in_days(melt, lags, ffday, customer_id_col, create_date_col, net_amount_col):
    for i in range(ffday, lags+1):
        melt['Last-' + str(i) + 'day_Sales'] = melt.groupby([customer_id_col])[net_amount_col].shift(i)
    
    melt = melt.reset_index(drop=True)
    
    for i in range(ffday, lags+1):
        melt['Last-' + str(i) + 'day_Diff'] = melt.groupby([customer_id_col])['Last-' + str(i) + 'day_Sales'].diff()
    melt = melt.fillna(0)
    
    melt['days_to_deliver'] = (melt["REQUESTED_DELIVERY_DATE"] - melt["ORDER_CREATION_DATE"]).dt.days
    
    return melt


melt='df'
lags=7
ffday=1
customer_id_col= 'CUSTOMER_NUMBER'
create_date_col= 'ORDER_CREATION_DATE'
net_amount_col= 'ORDER_AMOUNT'

new_df = difference_in_days(df.copy(), lags, ffday, customer_id_col, create_date_col, net_amount_col)
new_df


# In[45]:



column_names = new_df.columns

# Printing the column names
for column in column_names:
    print(column)


# In[49]:


df_Sorted = new_df.sort_values(by=['ORDER_CREATION_DATE','ORDER_CREATION_TIME'])
df_Sorted


# In[53]:


df_for_training = df_Sorted.copy()
# Create a list of column names to be dropped
columns_to_drop = ['CUSTOMER_ORDER_ID', 'SALES_ORG', 'COMPANY_CODE','ORDER_CREATION_DATE',
                   'ORDER_CREATION_TIME','CREDIT_CONTROL_AREA','REQUESTED_DELIVERY_DATE','CUSTOMER_NUMBER',
                   'unique_cust_id','SOLD_TO_PARTY','amount_in_usd','ORDER_AMOUNT','RELEASED_CREDIT_VALUE'
                   ,'Last-1day_Diff','Last-2day_Diff','Last-3day_Diff',
                   'Last-4day_Diff','Last-5day_Diff','Last-6day_Diff','Last-7day_Diff','ORDER_CURRENCY','days_to_deliver']

# Use the drop function to remove the specified columns
df_for_training = df_for_training.drop(columns=columns_to_drop)

column_name = 'amount_in_usd_log'
df_for_training = df_for_training[[col for col in df_for_training.columns if col != column_name] + [column_name]]

df_for_training.head(3)


# In[55]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score,make_scorer
from sklearn.model_selection import cross_val_score


# In[59]:


# Calculate the index position for splitting
split_index = int(0.8 * len(df_Sorted))
# Split the sorted dataframe into training and testing sets
train_df = df_for_training.iloc[:split_index]
test_df = df_for_training.iloc[split_index:]

# Split X and y
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values


# Linear Regression

# In[61]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_linear = regressor.predict(X_test)


# Decision Tree

# In[62]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred_tree = regressor.predict(X_test)


# Random Forest 

# In[63]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train, y_train)
y_pred_random = regressor.predict(X_test)


# Xg Boost

# In[65]:


pip install xgboost


# In[ ]:





# In[66]:


from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train,y_train)
y_pred_xgb = regressor.predict(X_test)


# AdaBoost

# In[69]:


from sklearn.ensemble import AdaBoostRegressor
# Create an AdaBoostRegressor model
ada_boost = AdaBoostRegressor(n_estimators=100)
# Train the model
ada_boost.fit(X_train, y_train)
y_pred_ada = regressor.predict(X_test)


# Cat Boost

# In[71]:


pip install catboost


# In[72]:


from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(X_train,y_train)
y_pred_cat = regressor.predict(X_test)


# SVM

# In[ ]:


from sklearn.svm import SVR

# Note: The following code may take a significant amount of time to run due to SVM's computational complexity

regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)
y_pred_svr = regressor.predict(X_test)


# 4.Compare the accuracies of all the models 

# In[73]:


import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

# Define the list of model names and corresponding y_pred values
model_names = ['Linear Regression','Descision Tree','Random Forest', 'XGBoost','AdaBoost','CatBoost']
y_preds = [y_pred_linear,y_pred_tree,y_pred_random, y_pred_xgb,y_pred_ada,y_pred_cat]  # Replace with the actual y_pred values for each model

# Initialize an empty dictionary to store the evaluation metrics
eval_metrics = {
    'Model': [],
    'MSE': [],
    'MAE': [],
    'RMSE': [],
    'MedAE': [],
    'R-squared': []
}

# Iterate through each model
for model_name, y_pred in zip(model_names, y_preds):
    # Calculate the evaluation metrics for the current model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    # Format the metric values with four decimal places
    mse_formatted = "{:.4f}".format(mse)
    mae_formatted = "{:.4f}".format(mae)
    rmse_formatted = "{:.4f}".format(rmse)
    medae_formatted = "{:.4f}".format(medae)
    r2_formatted = "{:.4f}".format(r2)

    # Add the metrics to the eval_metrics dictionary
    eval_metrics['Model'].append(model_name)
    eval_metrics['MSE'].append(mse_formatted)
    eval_metrics['MAE'].append(mae_formatted)
    eval_metrics['RMSE'].append(rmse_formatted)
    eval_metrics['MedAE'].append(medae_formatted)
    eval_metrics['R-squared'].append(r2_formatted)

# Create a DataFrame from the eval_metrics dictionary
metrics_df = pd.DataFrame(eval_metrics)

# Display the metrics table
metrics_df


# Q-5.Select the best possible model

# In[74]:


# Sort the metrics DataFrame based on a specific evaluation metric, such as R-squared
sorted_metrics_df = metrics_df.sort_values(by='R-squared', ascending=False)

# Get the best model (the one with the highest R-squared)
best_model = sorted_metrics_df.iloc[0]

# Display the metrics for the best model
print("Best Model Metrics:")
print("-------------------")
print("Model:", best_model['Model'])
print("MSE:", best_model['MSE'])
print("MAE:", best_model['MAE'])
print("RMSE:", best_model['RMSE'])
print("MedAE:", best_model['MedAE'])
print("R-squared:", best_model['R-squared'])


# In[ ]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb

parameters = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0.0, 0.1, 0.2],
    'subsample': [0.5,0.8, 1.0],
    'colsample_bytree': [0.5,0.8, 1.0],
}

xgb_regressor = xgb.XGBRegressor()

scoring = {
    'Negative MSE': 'neg_mean_squared_error',
    'R2 Score': make_scorer(r2_score)
}

grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=parameters, scoring=scoring, refit='R2 Score', cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_neg_mse = grid_search.best_score_
best_r2_score = grid_search.cv_results_['mean_test_R2 Score'][grid_search.best_index_]
best_params = grid_search.best_params_

print("Best Negative MSE: {:.2f}".format(best_neg_mse))
print("Best R2 Score: {:.2f}".format(best_r2_score))
print("Best Parameters: ", best_params)

