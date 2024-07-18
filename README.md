
# Employee Churn Analysis and Prediction

## Project Overview

This project aims to analyze and predict employee churn using a dataset that includes various demographic, contract, and service-related attributes. By understanding the factors contributing to employee churn, the company can implement targeted strategies to improve retention rates.


## Table of Contents


1. [Dataset Description](#dataset-description)
2. [Tools and Libraries Used](#tools-and-libraries-used)
3. [Steps Taken for Analysis](#steps-taken-for-analysis)
-  [Data Cleaning](#data-cleaning)
-  [Exploratory Data Analysis (EDA)/ Data Visualization](#exploratory-data-analysis-eda)
-   [Feature Engineering](#feature-engineering)
-   [Model Training](#model-training)
-   [Model Prediction](#model-prediction)
-   [Post-Prediction Analysis](#post-prediction-analysis)
-   [Report Creation](#report-creation)
4. [Key Findings](#key-findings)
- [Analysis of Actual Churn Data](#analysis-of-actual-churn-data)
- [Analysis of Predicted Employee Churn Using Machine Learning Model](#analysis-of-predicted-churn-data)
5. [Recommendations](#recommendations)


## Dataset Description

The dataset contains the following columns:

- `customerID`: Unique identifier for each employee.
- `gender`: Gender of the employee.
- `SeniorCitizen`: Indicates if the employee is a senior citizen (1) or not (0).
- `Partner`: Indicates if the employee has a partner (Yes or No).
- `Dependents`: Indicates if the employee has dependents (Yes or No).
- `tenure`: Number of months the employee has been with the company.
- `PhoneService`: Indicates if the employee has phone service (Yes or No).
- `MultipleLines`: Indicates if the employee has multiple lines (Yes, No, or No phone service).
- `InternetService`: Type of internet service the employee has (DSL, Fiber optic, or No).
- `OnlineSecurity`: Indicates if the employee has online security (Yes, No, or No internet service).
- `OnlineBackup`: Indicates if the employee has online backup (Yes, No, or No internet service).
- `DeviceProtection`: Indicates if the employee has device protection (Yes, No, or No internet service).
- `TechSupport`: Indicates if the employee has tech support (Yes, No, or No internet service).
- `StreamingTV`: Indicates if the employee has streaming TV (Yes, No, or No internet service).
- `StreamingMovies`: Indicates if the employee has streaming movies (Yes, No, or No internet service).
- `Contract`: Type of contract the employee has (Month-to-month, One year, or Two year).
- `PaperlessBilling`: Indicates if the employee has paperless billing (Yes or No).
- `PaymentMethod`: Method of payment (Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)).
- `MonthlyCharges`: The amount charged to the employee monthly.
- `TotalCharges`: The total amount charged to the employee.
- `Churn`: Indicates if the employee has churned (Yes or No).

## Tools and Libraries Used

- **Python** for data manipulation, analysis, and model building.
- **Pandas** for data manipulation and cleaning.
- **Seaborn, Matplotlib, Plotly** for data visualization.
- **Scikit-learn** for machine learning model building.
- **PowerBI** for creating interactive visual reports.

## Steps Taken for Analysis

### 1. Data Cleaning

- **Handling Missing Values:** Identify and fill or remove missing values to ensure a complete dataset.
```python
data.isnull().sum()
```
- **Data Type Conversion:** Convert data types where necessary (e.g., `TotalCharges` from string to float).

```python
# it will give the values that have space in the TotalCharges column
pd.to_numeric(data['TotalCharges'],errors='coerce')
```
```python
# this code give null/empty value in column that w're looking for
data[pd.to_numeric(data['TotalCharges'],errors='coerce').isnull()]
```
```python
# another method to find the empty value
data[data['TotalCharges'] == ' ']
```
```python
# consider only those rows and in which TotalCharges are not empty and store it into a new DataFrame named as 'df'
df = data[data['TotalCharges'] != ' ']
```

- **Consistency Checks:** Ensure the dataset is consistent and accurate by checking for duplicates and correcting inconsistencies.

```python
data.duplicated().sum()
```

### 2. Exploratory Data Analysis (EDA)/ Data Visualization

- **Descriptive Statistics:** Calculate summary statistics to understand the central tendency, dispersion, and shape of the dataset's distribution.
- **Data Visualization:** Create visualizations using Seaborn, Matplotlib, and Plotly to explore data distributions and relationships:
  - Gender vs Churn

  ```python
  plt.title(' % of Gender in the company')
  plt.pie(df['gender'].value_counts(),labels=['Male','Female',autopct='%.02f%%')
  plt.show()
  ```
![Screenshot 2024-07-18 184159](https://github.com/user-attachments/assets/99e54d36-385d-45e4-a509-7f9ff1ffee2b)

`Churn rate according to gender`

```python
male_churn_no = df[df['gender'] == 'Male']['Churn'].value_counts()[0]
male_churn_yes = df[df['gender'] == 'Male']['Churn'].value_counts()[1]
print('Male Leaving the company :',male_churn_yes)
print('Male Stayed the company :',male_churn_no)
```

```python
values = [male_churn_no, male_churn_yes]
categories = ['Churn No', 'Churn Yes']

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=categories, y=values)

# Customize the plot
plt.xlabel('Churn Status')
plt.ylabel('Count')
plt.title('Churn Count for Males')

# Show the plot
plt.show()
```
![Screenshot 2024-07-18 184338](https://github.com/user-attachments/assets/a4aa7ef6-d821-48da-b10b-46450a095d21)


```python
female_churn_no = df[df['gender'] == 'Female']['Churn'].value_counts()[0]
female_churn_yes = df[df['gender'] == 'Female']['Churn'].value_counts()[1]
print('Female Leaving the company :',female_churn_yes)
print('Female Stayed the company :',female_churn_no)
```
```python
values = [female_churn_no, female_churn_yes]
categories = ['Churn No', 'Churn Yes']

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=categories, y=values)

# Customize the plot
plt.xlabel('Churn Status')
plt.ylabel('Count')
plt.title('Churn Count for Females')

# Show the plot
plt.show()
```

![Screenshot 2024-07-18 184526](https://github.com/user-attachments/assets/69f95f6d-dda6-44cb-a4f6-246bf895fed9)


- Tenure vs Churn

`Let's see the churn rate according to churn rate`

```python
sns.distplot(df['tenure'])
```
![Screenshot 2024-07-18 184613](https://github.com/user-attachments/assets/ba39ff92-b0b5-4d57-a80c-fc8fafe084c4)

```python
df['tenure_bin'] = df['tenure']
```
```python
# Create bins
bins = pd.cut(df['tenure_bin'], bins=[0, 10, 20, 30, 40, 50, 60, 70 ,80], labels=['0-10', '10-20', '20-30', '30-40', '40-50','50-60','60-70','70-80'])
```
```python
# Add bins to DataFrame
df['tenure_bin'] = bins
```
```python
sns.countplot(x='tenure_bin',hue='Churn',data=df)
```

![Screenshot 2024-07-18 184649](https://github.com/user-attachments/assets/578a7923-577d-4cd7-b6c3-5270d4b75ea6)

- Contract vs Churn
`Here there are lots of graph that are maded to analyze this perticular relationships`

`Showing some basic one here refer to the notebook for full visualization`

```python
sns.countplot(x='Contract',data=df)
plt.title('Contract of the employee')
plt.show()
```
![Screenshot 2024-07-18 184807](https://github.com/user-attachments/assets/cc897191-e4fd-46a3-8e95-37d684647838)

- MonthlyCharges vs Contract - Churn rate
`again giving some basics graph to see in-depth follow the notebook`
```python
values = [df[df['Contract'] == 'Month-to-month'][['MonthlyCharges','Churn_num']].sum()[1],df[df['Contract'] == 'One year'][['MonthlyCharges','Churn_num']].sum()[1],df[df['Contract'] == 'Two year'][['MonthlyCharges','Churn_num']].sum()[1]]
label = ['Month-to-month','One year','Two year']
sns.barplot(y=values,x=label)
plt.title('MonthlyCharges vs Churned')
```

![Screenshot 2024-07-18 184943](https://github.com/user-attachments/assets/12048b08-3c32-4edf-80d6-ae16505a8777)

- MonthlyCharges vs Tenure

```python
sns.scatterplot(data=df,y='MonthlyCharges',x='tenure_bin',hue='Churn')
plt.title('Monthlycharges vs tenure churned')
```
![Screenshot 2024-07-18 185016](https://github.com/user-attachments/assets/0dfae50e-5306-4e3e-be4b-4ca15e553732)

 - TotalCharges vs Churn according to Contract
 ```python
 values = [df[df['Contract'] == 'Month-to-month'][['TotalCharges','Churn_num']].sum()[1],df[df['Contract'] == 'One year'][['TotalCharges','Churn_num']].sum()[1],df[df['Contract'] == 'Two year'][['TotalCharges','Churn_num']].sum()[1]]
label = ['Month-to-month','One year','Two year']
sns.barplot(y=values,x=label)
plt.title('TotalCharges vs Churned')
```
![Screenshot 2024-07-18 185107](https://github.com/user-attachments/assets/60a1199f-7af9-4b39-9df9-d24f87d15918)

 - PaymentMethod vs Churn
 ```python
 fig = px.histogram(data_frame=df,x='PaymentMethod',color='Churn')
# customize the plot
fig.update_layout(title='PaymentMethod vs Churn',
                  xaxis_title='PaymentMethod',
                  yaxis_title='Count')
fig.show()
```
![Screenshot 2024-07-18 185153](https://github.com/user-attachments/assets/9000ac25-180b-4ecd-95af-6156ef9396c6)


- MultipleLines vs Churn
```python
sns.countplot(data=df,x='MultipleLines',hue='Churn')
```

![Screenshot 2024-07-18 185249](https://github.com/user-attachments/assets/6db8174a-9af1-4a8f-ac33-488e18b9b812)

`Here is only one graph we're compared all the other parameters related to all the columns based on contract types and others also.`

- InternetService vs Churn
```python
sns.countplot(data=df,x='InternetService',hue='Churn')
```
![Screenshot 2024-07-18 185325](https://github.com/user-attachments/assets/ee83950f-53f3-42b9-a6db-c81f30ef4fa6)

`Here is only one graph we're compared all the other parameters related to all the columns based on contract types and others also.`

- StreamingTV vs Churn
```python
sns.countplot(data=df,x='StreamingTV',hue='Churn')
```

![Screenshot 2024-07-18 185358](https://github.com/user-attachments/assets/8ac21a5e-e163-4c5f-a5b0-339831a36d47)

`Here is only one graph we're compared all the other parameters related to all the columns based on contract types and others also.`

- StreamingMovies vs Churn

```python
sns.countplot(data=df,x='StreamingMovies',hue='Churn')
```
![Screenshot 2024-07-18 185428](https://github.com/user-attachments/assets/f8df6ee9-4ca0-4d1e-ae7a-cb8a1f9331f1)

`Here is only one graph we're compared all the other parameters related to all the columns based on contract types and others also.`
- OnlineSecurity & OnlineBackup vs Churn
```python
sns.countplot(data=df,x='OnlineSecurity',hue='Churn')
```
![Screenshot 2024-07-18 185510](https://github.com/user-attachments/assets/c63721b2-0185-4625-b846-d2c40dd71f9f)


```python
sns.countplot(data=df,x='OnlineBackup',hue='Churn')
```

![Screenshot 2024-07-18 185540](https://github.com/user-attachments/assets/a5deb7e4-4cdf-481d-bf0e-4f144745dbca)

`Here is only one graph we're compared all the other parameters related to all the columns based on contract types and others also.`

- Dependents & Partner vs Churn
```python
sns.countplot(data=df,x='Dependents',hue='Churn')
```

![Screenshot 2024-07-18 185616](https://github.com/user-attachments/assets/4408ec48-950e-438c-8829-27820d8c9c2f)

```python
fig = px.histogram(data_frame=df,x='tenure_bin',color='Partner')
fig.update_layout(title='tenure vs Partner',
                  xaxis_title='tenure',
                  yaxis_title='Count')
fig.show()
```

![Screenshot 2024-07-18 185714](https://github.com/user-attachments/assets/46dcdda5-7495-4e73-bdf3-d7c5938337d6)

`Here is only one graph we're compared all the other parameters related to all the columns based on contract types and others also.`

### 3. Feature Engineering

- **New Feature Creation:** Generate new features that capture additional information (e.g., tenure buckets).

`Since deleting customerID because it won't help us to give any prediction it's just useless for now.`
```python
df.drop(columns='customerID',inplace=True)
```


`Creating a function that will return only categorical columns`
```python
def cat_col(df):
  for column in df:
    if df[column].dtypes == 'object':
      print(f'{column}\n{df[column].unique()}\n')
```

`Firstly we change 'No internet service' and 'No phone service' as 'No' beacuse it's just the same for that model`
```python
df.replace('No internet service','No',inplace=True)
df.replace('No phone service','No',inplace=True)
```

`creating tenure bins for better understanding`

```python
df['tenure_bin'] = df['tenure']
```
```python
# Create bins
bins = pd.cut(df['tenure_bin'], bins=[0, 10, 20, 30, 40, 50, 60, 70 ,80], labels=['0-10', '10-20', '20-30', '30-40', '40-50','50-60','60-70','70-80'])
```
```python
# Add bins to DataFrame
df['tenure_bin'] = bins
```
- **Label Encoding:** Convert target variables into numerical format using techniques like one-hot encoding.

`LabelEncoding coverts the following column that we given to them into 'Yes' : 1 and 'No' : 0`
```python
le = LabelEncoder()
oe = OneHotEncoder(sparse=False)

df['Churn']= le.fit_transform(df['Churn'])
```

- **Categorical Encoding:** Convert categorical variables into numerical format using techniques like one-hot encoding.

`OneHotEncoder`
```python
one_hot_encoding_col = [['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService',
                        'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                        'StreamingMovies','Contract','PaperlessBilling','PaymentMethod']]
```

```python
df['gender'] = oe.fit_transform(df[['gender']])
df['Partner'] = oe.fit_transform(df[['Partner']])
df['Dependents'] = oe.fit_transform(df[['Dependents']])
df['PhoneService'] = oe.fit_transform(df[['PhoneService']])
df['MultipleLines'] = oe.fit_transform(df[['MultipleLines']])
df['InternetService'] = oe.fit_transform(df[['InternetService']])
df['OnlineSecurity'] = oe.fit_transform(df[['OnlineSecurity']])
df['OnlineBackup'] = oe.fit_transform(df[['OnlineBackup']])
df['DeviceProtection'] = oe.fit_transform(df[['DeviceProtection']])
df['TechSupport'] = oe.fit_transform(df[['TechSupport']])
df['StreamingTV'] = oe.fit_transform(df[['StreamingTV']])
df['StreamingMovies'] = oe.fit_transform(df[['StreamingMovies']])
df['Contract'] = oe.fit_transform(df[['Contract']])
df['PaperlessBilling'] = oe.fit_transform(df[['PaperlessBilling']])
df['PaymentMethod'] = oe.fit_transform(df[['PaymentMethod']])
```
- **Feature Scaling:** Standardize numerical features to ensure they contribute equally to the model performance.

`In order to do that we're using 'MinMaxScaler' from sklearn`
```python
mi = MinMaxScaler()

df['TotalCharges'] = mi.fit_transform(df[['TotalCharges']])
df['MonthlyCharges'] = mi.fit_transform(df[['MonthlyCharges']])
df['tenure'] = mi.fit_transform(df[['tenure']])
```

### 5. Model Training

- **Data Splitting:** Split the dataset into training and testing sets.

`Now, we're ready to build our model`
```python
X = df.drop(columns=['Churn'])
y = df['Churn']
```

```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


- **Model Selection:** Train various machine learning models (e.g., Logistic Regression, Decision Trees, Random Forest, Gradient Boosting) and compare their performance.

```python
def model_scorer(model_name,model):

    output=[]

    output.append(model_name)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    output.append(accuracy_score(y_test,y_pred))

    output.append(precision_score(y_test,y_pred))

    output.append(recall_score(y_test,y_pred))

    return output

model_dict={
    'log':LogisticRegression(),
    'decision_tree':DecisionTreeClassifier(),
    'random_forest':RandomForestClassifier(),
    'XGB':XGBClassifier()

}

model_output=[]
for model_name,model in model_dict.items():
    model_output.append(model_scorer(model_name,model))

model_output

```

`We're using 'LogisticRegression' as our model because it perform best among all of the other`

```python
lr = LogisticRegression()
```
```python
lr.fit(X_train,y_train)
```
```python
prediction = lr.predict(df.drop(columns=['Churn']))
```
- **Model Evaluation:** Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
`This will eveluated in the above part while Selecting the best models`

### 6. Model Prediction

- **Churn Prediction:** Use the best-performing model to predict employee churn.
```python
prediction = lr.predict(df.drop(columns=['Churn']))
```
- **Export Predicted Data:** Export the predicted churn data into a CSV file, focusing on employees predicted to churn despite not having done so yet.

```python
churn_df = data[data['Churn'] == 'No']
```
`Selecting only those data which have 'Yes' by the model but 'No' as in the main data`

```python
churn_df = churn_df[churn_df['predicted_churn'] == 'Yes']
```
```python
churn_df.to_csv('predicted_churn_dataframe.csv', index=False)
```

### 7. Post-Prediction Analysis

- **Analyze Predicted Churn:** Examine the characteristics of employees predicted to churn to validate the model and identify actionable insights.

- Gender vs Churn
```python
plt.title(' % of Gender in the company')
plt.pie(churn_df['gender'].value_counts(),labels=['Female','Male'],autopct='%.02f%%')
plt.show()
```
![Screenshot 2024-07-18 190052](https://github.com/user-attachments/assets/edd2c174-9819-4128-b590-ac32d9fd22c2)

- Churn vs tenure
```python
sns.distplot(churn_df['tenure'])
```
![Screenshot 2024-07-18 190146](https://github.com/user-attachments/assets/46e3242a-0305-4d89-b00c-9039e6195a87)

`Another one `

![Screenshot 2024-07-18 190135](https://github.com/user-attachments/assets/069f59a1-2b1d-41bd-a8e1-3e7206f96268)

- Contract vs churn
```python
sns.countplot(x='Contract',data=churn_df)
```
![Screenshot 2024-07-18 190250](https://github.com/user-attachments/assets/a89f1433-aed5-4a67-a1aa-7b3fcc8026f1)

- MonthlyCharges vs Churn
```python
fig = px.histogram(x='Contract',y='MonthlyCharges',data_frame=churn_df)
fig.update_layout(title='Contract type vs MonthlyCharges',
                  xaxis_title='Contract Type',
                  yaxis_title='MonthlyCharges')
fig.show()
```
```python
sns.scatterplot(data=churn_df,y='MonthlyCharges',x='tenure_bin',hue='predicted_churn')
plt.title('Monthlycharges vs tenure churned')
```
![Screenshot 2024-07-18 190336](https://github.com/user-attachments/assets/1d2364c8-4f1e-4da7-977a-71fb3945a028)

- PaymentMethod vs Churn
```python
sns.countplot(churn_df['PaymentMethod'])
```
![Screenshot 2024-07-18 190407](https://github.com/user-attachments/assets/0e8344e1-27c1-4251-9bc5-f910324ae6bc)

- MultipleLines vs Churn
```python
sns.countplot(data=churn_df,x='MultipleLines',hue='predicted_churn')
```
![Screenshot 2024-07-18 190438](https://github.com/user-attachments/assets/87fbdf61-679a-4df0-b504-8ba515a79234)

- InternetService vs Churn
```python
sns.countplot(data=churn_df,x='InternetService',hue='predicted_churn')
```
![Screenshot 2024-07-18 190514](https://github.com/user-attachments/assets/97e27d63-3da7-4343-95d7-5550b01b6c2f)

- StreamingTV vs Churn
```python
sns.countplot(data=churn_df,x='StreamingTV',hue='predicted_churn')
```
![Screenshot 2024-07-18 190612](https://github.com/user-attachments/assets/689ec893-99e6-4f28-9e0a-d34ed72d5100)

- StreamingMovies vs Churn
```python
sns.countplot(data=churn_df,x='StreamingMovies',hue='predicted_churn')
```
![Screenshot 2024-07-18 190644](https://github.com/user-attachments/assets/960cb021-3b58-40fe-b657-0ff123d5b050)

- Online-Security vs Churn
```python
fig = px.histogram(data_frame=churn_df,x='OnlineSecurity',color='predicted_churn')
fig.update_layout(title='Month-to-Month(contract type) having OnlineSecurity vs Churn',
                  xaxis_title='OnlineSecurity',
                  yaxis_title='Count')
fig.show()
```
![Screenshot 2024-07-18 190715](https://github.com/user-attachments/assets/928f2d7e-e304-4e0d-b196-2bbeded4bcfa)

- Partner vs Churn
```python
fig = px.histogram(data_frame=churn_df,x='Partner',color='predicted_churn')
fig.update_layout(title='Month-to-Month(contract type) having Partner vs Churn',
                  xaxis_title='Partner',
                  yaxis_title='Count')
fig.show()
```
![Screenshot 2024-07-18 190759](https://github.com/user-attachments/assets/86b503a3-c3ac-4127-a097-94476e8b01c5)

- **Insight Validation:** Compare predicted churn reasons with initial findings to ensure consistency and accuracy.

### 8. Report Creation

- **PowerBI Visualization:** Create a comprehensive PowerBI report to visualize both the main and predicted data:
  - Interactive dashboards to explore data and insights.
  - Visual comparisons of actual and predicted churn.
  - Key metrics and trends for quick understanding.

  `Main data prediction`
  ![Screenshot 2024-07-18 214524](https://github.com/user-attachments/assets/7a4c8f5d-8cb6-4e30-9d3e-11eae702cf4a)

`Post prediction file`

![Screenshot 2024-07-18 214557](https://github.com/user-attachments/assets/1346c6d4-33a5-4f7b-b860-e3640a38b130)

## Key Findings

### Analysis of Actual Churn Data

After a thorough examination of the dataset, several critical factors contributing to employee churn have been identified:

**Tenure and Dependents:**

 * Early Tenure Exodus : `Employees with a short tenure of 0-10 months, particularly those without partners or dependents, exhibit the highest churn rates. This suggests that new employees who lack familial or social anchors are more likely to leave.`

* Non-Dependent Employees : `Among all employees, 45% of those without dependents tend to leave, highlighting the potential instability in this group.`

**Senior Citizens:**

* Significant Senior Churn : `Notably, half of the senior citizen employees depart from the company, indicating a possible need for targeted retention strategies for this demographic.`

**Contract Type:**

* Month-to-Month Contracts : `Employees on month-to-month contracts are significantly more likely to leave, especially when they lack online backup or online security services. This contract type correlates strongly with higher churn due to perceived instability and lack of commitment.`

* Financial Burden : `High monthly charges are a critical factor driving away month-to-month contract employees, suggesting that cost sensitivity is a major concern.`

**Internet and Streaming Services:**

* FiberOptics Users : `Employees using FiberOptics Internet Service exhibit the highest churn rates compared to other internet services, possibly due to service issues or higher costs.`

* Streaming Services : `There is no clear correlation between streaming TV or movies and churn, indicating that these services do not significantly impact employee retention.`

**Payment Method:**

* Electronic Check Users : `A notable 45% of employees who use Electronic Check as their payment method choose to leave the company, hinting at potential dissatisfaction with this payment process.`

**Charges:**

* High Financial Burden : `Approximately 1,600 employees cite high total charges and monthly charges as their reasons for leaving, emphasizing the need for a review of the company's pricing structure.`


### Analysis of Predicted Employee Churn Using Machine Learning Model
`Based on the machine learning model's predictions and subsequent analysis of employees who are likely to churn despite not having done so yet, the following conclusions have been drawn:`

**Tenure:**

* Short Tenure Impact:` Employees with a tenure of 0-10 months show a high likelihood of leaving the company, indicating that short tenure is a strong predictor of churn.`

**Contract Type:**

* Month-to-Month Contracts: `Employees with month-to-month contracts are the most likely to churn, underscoring the instability associated with this contract type.`

**Monthly Charges:**

* Cost Sensitivity: `High monthly charges are a significant factor contributing to churn across all tenure ranges, suggesting that cost management is crucial for retention.`

**Payment Method:**

* Electronic Check: `Employees using Electronic Check as their payment method exhibit a higher propensity to churn, indicating dissatisfaction with this payment process.`

**Phone Service:**

* Multiple Lines: `Churn is prevalent among employees regardless of whether they have multiple phone lines, indicating that this factor alone is not a deterrent.`

**Internet Service:**

* Fiber-Optic Service: `Employees with Fiber-Optic Internet Service are the most likely to leave, suggesting issues related to this service type.`

**Streaming Services:**

* Streaming TV and Movies: `The presence or absence of streaming TV or streaming movies services does not significantly impact churn, as employees with both configurations are equally likely to leave.`

**Online Services:**

* Lack of Online Services: `Employees without online security or backup services show a higher likelihood of churn, indicating that these services are important for employee retention.`

**Dependents and Partners:**

* Absence of Partners: `Employees without partners tend to have a higher churn rate, suggesting that personal support systems play a role in retention.`

**Final Technical Insights:**

`The predictive analysis indicates that employees are at a higher risk of leaving the company due to a combination of short tenure, month-to-month contracts, high monthly charges, and the use of Electronic Check for payments. Additionally, Fiber-Optic Internet Service and the absence of online security or backup services are significant churn predictors. The presence of streaming services and multiple phone lines does not substantially influence churn rates, while the lack of partners is a notable factor contributing to employee attrition.`

`To mitigate these risks, targeted interventions focusing on improving contract stability, cost management, payment method satisfaction, and enhancing online services could be highly effective. Addressing these areas will help create a more supportive environment and reduce the likelihood of future churn.`

## Recommendations



To effectively reduce employee churn, the company must address these critical areas:

* **Enhancing New Employee Experience :** `Develop robust onboarding and retention programs for employees with short tenures, especially those without partners or dependents.`

* **Supporting Senior Employees :**`Implement targeted support and retention initiatives for senior citizen employees.`

* **Reviewing Contract Terms :** `Reassess the month-to-month contract offerings and consider adding benefits or incentives to enhance stability and commitment.`

* **Optimizing Costs :** `Evaluate and potentially revise the pricing structure to make monthly and total charges more manageable for employees.`

* **Improving Service Quality :** `Ensure high-quality internet services, particularly for those using FiberOptics, and enhance the availability and quality of online security and backup services.`

* **Streamlining Payment Methods :** `Revisit the payment methods, focusing on improving the experience for those using Electronic Check to reduce dissatisfaction.`

By addressing these areas, the company can create a more supportive and stable work environment, thereby significantly reducing employee churn and fostering long-term loyalty.
