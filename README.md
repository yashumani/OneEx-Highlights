# OneEx DQ Check Script Execution Documentation

## Overview

This document details the execution and functionality of the **OneEx DQ Check Script** designed to analyze data quality (DQ) for large-scale datasets. It processes dimensional and metric combinations to detect anomalies, calculate thresholds, and provide clustering insights, producing four comprehensive tables.

---

## Theoretical Explanation

### **Purpose**
The script aims to automate data quality checks, flag anomalies, and offer structured insights for metrics across multiple dimensions and periods.

---

### **Key Features**

#### **1. Dimensional and Metric Analysis**
- Processes **unique combinations** of dimensions (`SLS_DIST_CHNL_TYPE_DESC`, `PREPAID_IND`, `SEGMENT`) to create granular intersections.
- Detects anomalies for measures (e.g., `GROSS_ADDS_TOTAL`, `NET_ADDS_PHONE`) using thresholds derived from quantiles.
- Supports metrics constrained to negative values (e.g., `NET_DEACTS_TOTAL`).

#### **2. Threshold Aggregation**
- Calculates **global ranges** (min, max) for all metrics.
- Provides **seasonal thresholds** for periods: `ALL`, `WTD`, `MTD`, `QTD`.
- Helps establish dynamic baselines for anomaly detection.

#### **3. Period-Based Analysis**
- Retains and analyzes **period-specific data**, providing temporal trends.
- Dynamically calculates period ranges:
  - **WTD:** From last Sunday.
  - **MTD:** From the first day of the current month.
  - **QTD:** From the first day of the current quarter.

#### **4. Clustering**
- Implements **KMeans clustering** to group data points into activity categories (`Low Activity`, `Medium Activity`, `High Activity`).
- Adds dimensional and temporal context to clusters.

#### **5. Scalability**
- Efficiently processes large datasets using modular functions for extensibility.
- Generates four structured tables:
  - **Table 1:** Thresholds and anomalies for all dimensions and periods.
  - **Table 2:** Period-specific insights with clustering.
  - **Table 3:** Aggregated thresholds across dimensions.
  - **Table 4:** Detailed cluster statistics with dates and metrics.

---

## Code Explanation

### **1. Input Data Validation**
- Ensures the presence of required columns: `DATE`, dimensions, and measures.
- Converts dimensions to strings and `DATE` to datetime format.

---

### **2. Period Calculations**
- Identifies start dates for `WTD`, `MTD`, and `QTD`, adjusting for leap years.
- Assigns periods to rows based on their `DATE`.

---

### **3. Threshold Calculations**
- For each measure and dimension combination, thresholds are derived using quantiles:
  - **Lower 2% (Quantile 0.02):** Identifies less extreme lower outliers.
  - **Lower 0.5% (Quantile 0.005):** Identifies more extreme lower outliers.
  - **Upper 98% (Quantile 0.98):** Identifies less extreme upper outliers.
  - **Upper 99.5% (Quantile 0.995):** Identifies more extreme upper outliers.

---

### **4. Anomaly Detection**
- Compares metrics against thresholds to assign anomaly types:
  - **More Extreme Outlier:** Beyond the 99.5% or 0.5% quantiles.
  - **Less Extreme Outlier:** Between the 98% and 99.5% or 0.5% and 2% quantiles.
  - **Unexpected Positive Value:** For negative-only metrics.

---

### **5. Clustering**
- Clusters data points based on metric values into three activity levels:
  - **Cluster 1:** Low Activity.
  - **Cluster 2:** Medium Activity.
  - **Cluster 3:** High Activity.

---

### **6. Table Outputs**

#### **Table 1: Dimensional and Temporal Anomalies**
- Includes anomalies, thresholds, and cluster details for all dimension combinations and periods.
- Provides period-specific date ranges (`From_Date`, `To_Date`).

#### **Table 2: Period-Specific Insights**
- Retains the original dataset structure with added clustering labels.

#### **Table 3: Aggregated Thresholds**
- Calculates min/max ranges for thresholds across dimensions.

#### **Table 4: Cluster Statistics**
- Includes:
  - Cluster counts.
  - Date ranges.
  - Dimension breakdown.
  - Cluster centers (mean values for each metric).

---

## Implementation Details

### **1. Threshold and Anomaly Calculations**
```python
def calculate_thresholds(df, measure):
    return {
        'lower_2': df[measure].quantile(0.02),
        'upper_98': df[measure].quantile(0.98),
        'lower_05': df[measure].quantile(0.005),
        'upper_995': df[measure].quantile(0.995)
}
```

```python
def determine_anomaly_type(df, measure, thresholds, negative_only):
    df[f'{measure}_Anomaly_Type'] = 'Normal'
    df.loc[df[measure] < thresholds['lower_05'], f'{measure}_Anomaly_Type'] = 'More Extreme Outlier'
    df.loc[df[measure] > thresholds['upper_995'], f'{measure}_Anomaly_Type'] = 'More Extreme Outlier'
    return df
```

---

### **2. Clustering**
```python
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[measures + negative_measures].fillna(0))
df['Cluster_Label'] = df['Cluster'].map({
    0: 'Low Activity',
    1: 'Medium Activity',
    2: 'High Activity'
})
```

---

### **3. Final Table Outputs**

#### **Table 1:**
Includes anomalies, thresholds, cluster details, and temporal ranges:
```python
output_table_1 = df_table_1_new[
    base_columns + 
    [col for col in df_table_1_new.columns if '_Has_Anomaly' in col or '_Anomaly_Details' in col]
]
```

#### **Table 2:**
Retains period-specific clustering:
```python
output_table_2 = df.copy()
```

#### **Table 3:**
Aggregated thresholds:
```python
df_table_3 = thresholds_df.groupby(['Metric'] + dimensions).agg({
    'lower_2': ['min', 'max'],
    'upper_98': ['min', 'max']
}).reset_index()
```

#### **Table 4:**
Cluster statistics with dimensional context:
```python
output_table_4 = pd.merge(
    cluster_stats,
    cluster_centers,
    on='Cluster'
)
```

---

## Outputs
1. **`output_table_1`**: Thresholds, anomalies, and clustering.
2. **`output_table_2`**: Period-specific clustering.
3. **`output_table_3`**: Aggregated thresholds.
4. **`output_table_4`**: Cluster statistics with dimensions and date ranges.

---

This script efficiently integrates anomaly detection, threshold aggregation, clustering, and dimensional analysis into a comprehensive data quality solution.
