# Wildfire Data Analysis Using Code Llama

## Introduction
Wildfires have been increasing in frequency and intensity, posing severe risks to life, property, and the environment. Analyzing wildfire data can help uncover long-term trends, optimize disaster response, and guide preventive measures. However, real-world wildfire datasets often contain missing values, outliers, and inconsistencies, making data preprocessing and analysis a challenging task.

In this study, we explore the capabilities of **Code Llama**, a code generation AI model, to optimize Python scripts for efficient data cleaning, automated outlier detection, and time series analysis of wildfire patterns.

## Research Questions
1. **How can we use Python scripts for handling and analyzing large wildfire datasets?**
2. **What is the best approach for outlier detection and missing value imputation in our dataset?**
3. **How can we use time series analysis to uncover long-term patterns in wildfires?**

## Setting Up Code Llama
To test Code Llama's efficiency, we set up a code generation environment using an open-source extension for VS Code that integrates with **Code Llama**. The setup involved:

- Installing the necessary dependencies:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```
- Integrating Code Llama’s **generate_code** function within our IDE.
- Providing a clear problem statement and structured input prompts to generate useful Python scripts.

## Code Llama in Action: Data Cleaning & Analysis

### 1. Data Cleaning and Preprocessing
One of the first tasks was handling missing values and outliers. Using Code Llama, we generated a Python script to:

- Detect missing values and apply appropriate imputation techniques.
- Identify and handle outliers using the **Interquartile Range (IQR)** method.

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("California Wildfire Damage.csv")

# Handling missing values
imputer = SimpleImputer(strategy='median')
df.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])

# Outlier detection using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
df[outlier_mask] = np.nan  # Convert outliers to NaN for imputation
```

### 2. Time Series Analysis
To analyze wildfire trends over time, we used Code Llama to generate a Python script for time series decomposition and visualization:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot wildfire occurrences over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='Acres Burned')
plt.title("Wildfire Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Acres Burned")
plt.show()
```

## Insights and Findings
1. **Missing Value Trends**: The dataset contained missing values primarily in attributes related to cause and containment efforts. Code Llama efficiently suggested median imputation for numerical data and mode imputation for categorical variables.
2. **Outlier Detection**: The IQR method identified extreme values in acres burned and fatalities, helping refine the dataset before analysis.
3. **Wildfire Growth Patterns**: The time series analysis revealed an **increasing trend in the number of acres burned over the past two decades**, highlighting the growing severity of wildfires.
4. **Peak Seasons**: Most wildfires occurred between **July and September**, correlating with dry and hot weather conditions.
5. **Correlation Analysis**: Using PandasAI, we found strong correlations between **temperature, drought severity, and wildfire size**.

## Evaluating Code Llama’s Performance
### Strengths:
- **Generated accurate, optimized Python scripts for data cleaning and visualization.**
- **Helped automate repetitive tasks**, reducing manual coding effort.
- **Provided multiple approaches** to missing value imputation and outlier handling.

### Challenges:
- Some **code snippets required fine-tuning** for efficiency and readability.
- **Limited flexibility** in handling highly customized prompts.
- **Generated code needed verification** before execution, especially for complex tasks.

## Future Applications
Code Llama’s AI-generated scripts can be extended to:
- **Predict wildfire occurrences** using machine learning models.
- **Automate real-time data processing** for early warning systems.
- **Develop interactive dashboards** for wildfire tracking and forecasting.

## Conclusion
This study demonstrated the power of **Code Llama** in automating wildfire data analysis. By leveraging AI for code generation, we optimized Python scripts for data preprocessing, outlier detection, and time series analysis. The results highlight the increasing severity of wildfires and the importance of AI-driven insights for disaster preparedness.

By refining AI-generated code and combining it with expert-driven analysis, we can create robust solutions for tackling real-world challenges in **environmental data science**.

---
### 🚀 **Next Steps & Repository Contribution**
🔗 *Contribute to this project or explore the full dataset and codebase on [GitHub](https://github.com/your-repo-link).*

