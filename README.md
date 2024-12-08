### README.md

# Phase 3: Advanced Data Analytics and Narrative Generation

This document provides a comprehensive explanation of the functioning of the Phase 3 code for advanced data analytics and narrative generation. The code is tailored for business insights based on Verizon's KPIs, combining data aggregation, historical comparisons, machine learning insights, and advanced narrative generation to provide actionable intelligence.

---

## Part 1: **Holistic Explanation**

### Overview
The Phase 3 code processes complex operational and financial data to generate meaningful insights and reports. It automates the following steps:
1. **Data Aggregation**:
   - Combines data from multiple dimensions (`channel`, `segment`, `period`) and metrics to create a unified dataset.
   - Handles both total-level (ALL) and granular-level (specific dimension) aggregation.

2. **Historical Comparisons**:
   - Compares metrics across key periods:
     - **WTD** (Week-To-Date), **MTD** (Month-To-Date), **QTD** (Quarter-To-Date), **YTD** (Year-To-Date).
     - Historical benchmarks such as `SameDayLastWeek`, `SameDateLastMonth`, and `SameDayLastYear`.
   - Provides context for current performance and trends.

3. **Narrative Generation**:
   - Tailored narratives provide detailed descriptions of changes in metrics.
   - Contextualized by dimension (e.g., channel, customer segment) and metric type (e.g., activations, churn).

4. **ML Integration**:
   - Employs Random Forest to calculate the importance of metrics in predicting a target metric (e.g., `NET_ADDS_TOTAL`).
   - Integrates machine learning insights into the narratives to highlight impactful metrics.

5. **Output Reporting**:
   - Segregates data into actionable reports:
     - Top 10 changes for `WTD`, `MTD`, and `QTD`.
     - Detailed narratives by dimensions.
     - Comprehensive data with all metrics and insights.
     - ML/DL-generated stories for predictive analytics.

---

## Part 2: **Code Explanation**

### 1. **Metric and Data Definitions**
- The `metrics` and `reverse_logic_metrics` lists define the KPIs being analyzed. These include:
  - **Metrics**: Performance indicators like `GROSS_ADDS_TOTAL`, `NET_ADDS_TOTAL`, `CHURN_POSTPAID_FWA_BASE_TOTAL`.
  - **Reverse Logic Metrics**: Metrics where a decrease is favorable, such as `NET_DEACTS_TOTAL`.

### 2. **Utility Functions**
- **`add_all_rows`**:
  - Adds aggregated "ALL" rows to represent totals across dimensions.
  - Ensures reports include both granular and total-level insights.

- **`generate_advanced_narrative`**:
  - Produces human-like narratives tailored to specific metrics.
  - Includes:
    - Trend analysis (`increased` or `declined`).
    - Contextual significance (`substantial shift`, `notable trend`, `minor change`).
    - KPI-specific descriptions (e.g., churn, activations).
    - Dimension-based insights (`channel`, `prepaid/postpaid`, `segment`).

### 3. **Data Loading and Transformation**
- **Input Table**:
  - Data is loaded into `df`, which represents raw input for processing.
  - Converts `DATE` to a `datetime` format for period calculations.

- **Historical Periods**:
  - Defines start and end dates for:
    - Current periods (`WTD`, `MTD`, `QTD`, `YTD`).
    - Historical periods for comparison (`SameDayLastWeek`, `SameDateLastMonth`, etc.).
  - Filters data into these periods for aggregation.

### 4. **Aggregation and Pivoting**
- Aggregates data across dimensions (`channel`, `segment`) and metrics.
- Uses a "melted" format for flexible metric-wise analysis.

### 5. **Narrative Generation**
- Iterates over pivoted data to:
  - Compare current values to historical benchmarks.
  - Generate narratives using `generate_advanced_narrative`.
  - Skip extreme changes to maintain narrative relevance.

### 6. **ML Integration**
- **`rank_metrics_with_ml`**:
  - Trains a Random Forest model to calculate feature importance scores for metrics.
  - Scores indicate the significance of metrics in predicting a target metric (e.g., `NET_ADDS_TOTAL`).

- **Integration in Narratives**:
  - Adds ML insights to narratives to emphasize the importance of metrics.
  - Facilitates actionable decision-making by highlighting critical areas.

### 7. **Output Tables**
- **`output_table_1`**:
  - Top 10 narratives for `WTD`, `MTD`, and `QTD`.
- **`output_table_2`**:
  - Detailed narratives by dimension.
- **`output_table_3`**:
  - Comprehensive data with all metrics and insights.
- **`output_table_4`**:
  - ML/DL-generated stories based on metric importance.

### 8. **Key Features**
- **Scalability**:
  - Modular design allows easy integration of additional metrics or dimensions.
- **Customizability**:
  - Tailored narratives and thresholds for trend significance.
- **Accuracy**:
  - Uses historical data comparisons and ML-driven insights to ensure data reliability.

---

### Instructions for Use
1. **Input Data**:
   - Load raw data into `input_table_1`.
2. **Run Script**:
   - Execute the script to generate outputs.
3. **Review Reports**:
   - Analyze the generated narratives and outputs in `output_table_1`, `output_table_2`, `output_table_3`, and `output_table_4`.

---

This code streamlines the process of transforming raw data into actionable insights, enabling data-driven decision-making.
