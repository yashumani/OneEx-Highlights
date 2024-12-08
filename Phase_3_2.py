import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Define metrics columns
metrics = [
    "GROSS_ADDS_TOTAL",  "NET_ADDS_TOTAL", "GROSS_ADDS_PHONE", 
     "NET_ADDS_PHONE",  "GROSS_ADDS_TABLET", "NET_DEACTS_TABLET", "NET_ADDS_TABLET", 
     "GROSS_ADDS_OTHER",    "NET_DEACTS_OTHER", "NET_ADDS_OTHER", "UPGRADES", "UPGRADES_PHONE"
     ,"NEW_ACCOUNTS",    "UNLIMITED_NEW_ACCOUNTS", "NEW_ACCOUNTS_SINGLE_PHONE", "NEW_ACCOUNTS_MULTI_PHONE", 
    "LOST_ACCOUNTS", "UPGRADES_TABLET", "UPGRADES_APPLE_SMARTWATCH", 
    "UPGRADES_OTHER", "VSB_GROSS_ADDS",  "VSB_NET_ADDS",  "GROSS_ADDS_TOTAL_FWA", "NET_ADDS_TOTAL_FWA", 
     "TF_GROSS_ADDS_TOTAL", "TF_NET_DEACTS_TOTAL", 
    "TF_NET_ADDS_TOTAL", "GROSS_ADDS_OTHER_WEARABLES", "NET_DEACTS_OTHER_WEARABLES", 
    "NET_ADDS_OTHER_WEARABLES", "UPGRADES_OTHER_WEARABLES", "CHURN_POSTPAID_FWA_BASE_TOTAL", 
    "GROSS_ADDS_PHONE_AAL", "GROSS_ADDS_PHONE_NEW", "NEW_ACCOUNTS_PHONE_PREMIUM", 
    "NEW_ACCOUNTS_FWA_STAND_ALONE", "TF_SL_GROSS_ADDS", "TF_SL_NET_ADDS", 
    "GROSS_ADDS_2ND_LINE"
]
# Define reverse logic metrics
reverse_logic_metrics = [
    "NET_DEACTS_TOTAL", "NET_DEACTS_PHONE", "NET_DEACTS_SMARTPHONE", "NET_DEACTS_TABLET"
    ,"VSB_NET_DEACTS","VOLUNTARY_POSTPAID_DISCOS", "INVOLUNTARY_POSTPAID_DISCOS","NET_DEACTS_2ND_LINE", "TF_SL_NET_DEACTS"
    , "VOL_POSTPAID_PHONE_DISCOS", "VOL_POSTPAID_TABLET_DISCOS","NET_DEACTS_TOTAL_FWA", 
    "VOL_POSTPAID_OTHER_DISCOS", "INVOL_POSTPAID_PHONE_DISCOS", "INVOL_POSTPAID_TABLET_DISCOS", 
    "INVOL_POSTPAID_OTHER_DISCOS"
]

# Utility to add "ALL" rows
def add_all_rows(df, dimension_cols, metrics):
    """
    Adds rows to the dataframe representing totals (All) for each dimension.
    """
    all_rows = []
    for dim in dimension_cols:
        temp = df.copy()
        temp[dim] = "All"
        grouped = temp.groupby(dimension_cols + ['Period'], dropna=False)[metrics].sum().reset_index()
        all_rows.append(grouped)
    return pd.concat([df] + all_rows, ignore_index=True)

# Enhanced narrative generation function tailored for the dataset
def generate_advanced_narrative(metric, change, current, historical, channel, prepaid, segment, period="Yesterday", historical_period="SameDayLastWeek"):
    """
    Generate advanced narratives tailored to the dataset, focusing on actionable insights.
    """
    # Determine the trend based on the metric type
    trend = "declined" if change < 0 else "increased"
    change_percent = (abs(change) / historical) * 100 if historical != 0 else 0

    # Identify significance of change
    if abs(change_percent) > 25:
        significance = "a substantial shift"
    elif abs(change_percent) > 10:
        significance = "a notable trend"
    else:
        significance = "a minor change"

    # Narrative tailoring by metric type
    if metric == "GROSS_ADDS_TOTAL":
        context = f"{significance} in activations, indicating adjustments in customer acquisition strategies."
    elif metric == "CHURN_POSTPAID_FWA_BASE_TOTAL":
        context = f"{significance} in churn rates, reflecting customer retention challenges in fixed wireless access."
    elif metric == "NET_ADDS_TOTAL":
        context = "indicating overall growth or decline in the customer base."
    elif metric == "UPGRADES":
        context = f"{significance} in upgrades, highlighting changes in existing customer behavior."
    elif "MARGIN" in metric:
        context = f"{significance} in margins, suggesting fluctuations in profitability."
    else:
        context = "indicating operational trends."

    # Include dimensions in the narrative
    dimension_context = f"observed in the {channel} channel, focusing on {'prepaid' if prepaid == 'Y' else 'postpaid'} customers within the {segment} segment."

    # Generate the final narrative
    narrative = (
        f"During {period}, {metric} {trend} by {abs(change):,.0f} units ({change_percent:.2f}%), "
        f"{context} The current value is {current:,.0f}, compared to {historical:,.0f} observed in {historical_period}. "
        f"This was {dimension_context}"
    )

    return narrative

# Load Data
df = input_table_1.copy()

# Convert DATE to datetime
df['DATE'] = pd.to_datetime(df['DATE']).dt.date

# Define date ranges for historical periods
today = datetime.now().date()
yesterday = today - timedelta(days=1)
wtd_start = today - timedelta(days=today.weekday())
mtd_start = today.replace(day=1)
qtd_start = today.replace(month=((today.month - 1) // 3) * 3 + 1, day=1)
ytd_start = today.replace(month=1, day=1)
sdlw = yesterday - timedelta(days=7)
sdlm = yesterday - timedelta(days=30)
sdly = yesterday - timedelta(days=365)
sdlq = yesterday - timedelta(days=90)

# Filter data for each period
period_data = {
    "Yesterday": df[df['DATE'] == yesterday],
    "WTD": df[(df['DATE'] >= wtd_start) & (df['DATE'] <= today)],
    "MTD": df[(df['DATE'] >= mtd_start) & (df['DATE'] <= today)],
    "QTD": df[(df['DATE'] >= qtd_start) & (df['DATE'] <= today)],
    "YTD": df[(df['DATE'] >= ytd_start) & (df['DATE'] <= today)],
    "SameDayLastWeek": df[df['DATE'] == sdlw],
    "SameDateLastMonth": df[df['DATE'] == sdlm],
    "SameDayLastYear": df[df['DATE'] == sdly],
    "SameDateLastQuarter": df[df['DATE'] == sdlq]
}

# Aggregate data for each period
aggregated_metrics = {}
for period_name, period_df in period_data.items():
    if period_df.empty:
        print(f"No data available for period '{period_name}'. Skipping.")
        continue
    aggregated_metrics[period_name] = period_df.groupby(['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT'])[metrics].sum().reset_index()

# Combine Periods into a Single DataFrame
all_periods = []
for period_name, period_df in aggregated_metrics.items():
    period_df['Period'] = period_name
    all_periods.append(period_df)

if not all_periods:
    raise ValueError("No data available after aggregation. Check the input dataset.")

full_metrics = pd.concat(all_periods)

# Add 'ALL' rows
dimension_cols = ['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT']
full_metrics = add_all_rows(full_metrics, dimension_cols, metrics)

# Add 'Metric' column to the DataFrame
full_metrics = full_metrics.melt(id_vars=['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT', 'Period'], 
                                 value_vars=metrics, 
                                 var_name='Metric', 
                                 value_name='Value')

# Convert 'Value' column to numeric
full_metrics['Value'] = pd.to_numeric(full_metrics['Value'], errors='coerce')

# Modify the pivot table creation to preserve the Period column
pivot_table = full_metrics.pivot_table(
    index=['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT', 'Metric', 'Period'],
    columns=None,
    values='Value',
    aggfunc='first'
).reset_index()

# Add debug prints to verify data
print("Full Metrics columns:", full_metrics.columns.tolist())
print("Pivot Table columns:", pivot_table.columns.tolist())
print("Sample of pivot_table:")
print(pivot_table.head())

# Modify the narrative generation section with proper error handling
narratives = []
for _, row in pivot_table.iterrows():
    metric = row['Metric']
    channel = row['SLS_DIST_CHNL_TYPE_DESC']
    prepaid = row['PREPAID_IND']
    segment = row['SEGMENT']
    period = row['Period']
    current = row['Value']
    
    historical_periods = {
        "Yesterday": "SameDayLastWeek",
        "WTD": "SameDayLastWeek",
        "MTD": "SameDateLastMonth",
        "QTD": "SameDateLastQuarter",
        "YTD": "SameDayLastYear"
    }
    historical_period = historical_periods.get(period, None)
    
    if historical_period:
        # Get historical data with error handling
        historical_data = full_metrics[
            (full_metrics['Period'] == historical_period) &
            (full_metrics['Metric'] == metric) &
            (full_metrics['SLS_DIST_CHNL_TYPE_DESC'] == channel) &
            (full_metrics['PREPAID_IND'] == prepaid) &
            (full_metrics['SEGMENT'] == segment)
        ]['Value']
        
        # Check if historical data exists
        historical = historical_data.iloc[0] if not historical_data.empty else 0
        
        # Calculate change percentage and skip if more or less than 300%
        change_percent = ((current - historical) / historical) * 100 if historical != 0 else 0
        if abs(change_percent) > 2000:
            continue
        
        try:
            narrative = generate_advanced_narrative(
                metric, current - historical, current, historical, 
                channel, prepaid, segment, period, historical_period
            )
            narratives.append({
                'SLS_DIST_CHNL_TYPE_DESC': channel,
                'PREPAID_IND': prepaid,
                'SEGMENT': segment,
                'Metric': metric,
                'Period': period,
                'Current Value': current,
                'Historical Period': historical_period,
                'Change (%)': change_percent,
                'Narrative': narrative
            })
        except Exception as e:
            print(f"Error generating narrative for {metric} in {period}: {str(e)}")
            continue

# Convert narratives to DataFrame and continue with the rest of the code
narratives_df = pd.DataFrame(narratives) if narratives else pd.DataFrame()

# Debug print to check if narratives_df is empty
print("Narratives DataFrame:")
print(narratives_df.head())

# Ensure column names match for merging
if not narratives_df.empty:
    narratives_df.columns = ['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT', 'Metric', 'Period', 'Current Value', 'Historical Period', 'Change (%)', 'Narrative']

# Ensure 'Period' column is correctly added to narratives_df
if not narratives_df.empty:
    narratives_df['Period'] = narratives_df['Period'].astype(str)

# Debug print to check if 'Period' column exists in both DataFrames
print("Pivot Table Columns:", pivot_table.columns)
print("Narratives DataFrame Columns:", narratives_df.columns)

# Merge narratives with pivot table
final_table = pivot_table.merge(narratives_df, on=['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT', 'Metric', 'Period'], how='left')

# KNIME output table
output_table_1 = final_table

# Print the pivoted output table for verification
print(final_table)

# After generating narratives_df and before creating output tables:

# Table 1: Concatenate top 10 for WTD, MTD, QTD
if not narratives_df.empty:
    # Filter for WTD, MTD, QTD
    top_wtd = narratives_df[narratives_df['Period'] == 'WTD'].nlargest(10, 'Change (%)')
    top_mtd = narratives_df[narratives_df['Period'] == 'MTD'].nlargest(10, 'Change (%)')
    top_qtd = narratives_df[narratives_df['Period'] == 'QTD'].nlargest(10, 'Change (%)')
    output_table_1 = pd.concat([top_wtd, top_mtd, top_qtd]).reset_index(drop=True)
else:
    output_table_1 = pd.DataFrame()

# Table 2: Keep only those columns that are used for narratives
if not narratives_df.empty:
    output_table_2 = narratives_df[['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT', 'Metric', 'Period', 'Current Value', 'Change (%)', 'Narrative']]
else:
    output_table_2 = pd.DataFrame()

# Table 3: Complete table with all metrics and narratives by period
if not narratives_df.empty:
    output_table_3 = full_metrics.merge(
        narratives_df,
        on=['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT', 'Metric', 'Period'],
        how='left'
    )
    
    # Sort by Period and Metric for better readability
    output_table_3 = output_table_3.sort_values(['Period', 'Metric'])
    output_table_3 = output_table_3.reset_index(drop=True)  # Reset index to avoid duplicates
else:
    output_table_3 = pd.DataFrame()

# Print verification
print("\nTop 10 Stories with Narratives:")
print(output_table_1)
print("\nTop 10 Detailed Metrics by Dimension:")
print(output_table_2)
print("\nComplete Table Sample:")
print(output_table_3.head())

# Modify the ML function to handle the melted data format
def rank_metrics_with_ml(df, target_metric='NET_ADDS_TOTAL'):
    """
    Use Random Forest to rank metrics by importance in predicting target metric
    """
    # Pivot the data back to wide format for ML
    ml_data = df.pivot_table(
        index=['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT', 'Period'],
        columns='Metric',
        values='Value',
        aggfunc='mean'
    ).reset_index()

    # Create label encoders for categorical columns
    encoders = {}
    categorical_cols = ['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT', 'Period']
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        ml_data[f'{col}_encoded'] = encoders[col].fit_transform(ml_data[col].astype(str))

    # Prepare features
    feature_cols = [col for col in metrics if col != target_metric]
    encoded_cols = [f'{col}_encoded' for col in categorical_cols]
    all_features = feature_cols + encoded_cols

    # Prepare feature matrix
    X = ml_data[all_features].fillna(0)
    y = ml_data[target_metric].fillna(0)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    # Get feature importance scores for metrics only
    importance_scores = pd.DataFrame({
        'Metric': feature_cols,
        'ML_Importance_Score': rf_model.feature_importances_[:len(feature_cols)]
    })
    
    return importance_scores

# Calculate ML-based importance scores
ml_scores = rank_metrics_with_ml(full_metrics)

# Create a dictionary for mapping importance scores
importance_dict = dict(zip(ml_scores['Metric'], ml_scores['ML_Importance_Score']))

# Add ML insights to narratives and output tables
if not narratives_df.empty:
    # Add ML insights to narratives
    for idx, row in narratives_df.iterrows():
        metric = row['Metric']
        if metric in importance_dict:
            importance_score = importance_dict[metric]
            ml_insight = (
                f" Based on ML analysis, this metric has an importance score of {importance_score:.4f} "
                f"in predicting overall performance."
            )
            narratives_df.at[idx, 'Narrative'] += ml_insight
            narratives_df.at[idx, 'ML_Importance_Score'] = importance_score

    # Update output tables with ML insights
    output_table_1 = narratives_df.copy()
    output_table_1['ML_Importance_Score'] = output_table_1['Metric'].map(importance_dict)
    
    # Sort by absolute Change (%) and ML importance for each period separately
    top_wtd = output_table_1[output_table_1['Period'] == 'WTD'].nlargest(10, 'Change (%)')
    top_mtd = output_table_1[output_table_1['Period'] == 'MTD'].nlargest(10, 'Change (%)')
    top_qtd = output_table_1[output_table_1['Period'] == 'QTD'].nlargest(10, 'Change (%)')
    
    # Concatenate the top 10 for each period
    output_table_1 = pd.concat([top_wtd, top_mtd, top_qtd]).reset_index(drop=True)

    # Update output_table_2 with ML insights
    if not output_table_2.empty:
        output_table_2['ML_Importance_Score'] = output_table_2['Metric'].map(importance_dict)
        output_table_2 = output_table_2.sort_values('ML_Importance_Score', ascending=False)
        output_table_2 = output_table_2.reset_index(drop=True)

    # Update output_table_3 with ML insights
    if not output_table_3.empty:
        output_table_3['ML_Importance_Score'] = output_table_3['Metric'].map(importance_dict)
        output_table_3 = output_table_3.sort_values(['ML_Importance_Score', 'Period', 'Metric'], 
                                                   ascending=[False, True, True])
        output_table_3 = output_table_3.reset_index(drop=True)

# Print updated verification
print("\nML Importance Scores:")
print(ml_scores)

print("\nTop 10 Stories with ML Insights:")
if not output_table_1.empty and 'ML_Importance_Score' in output_table_1.columns:
    print(output_table_1[['Metric', 'ML_Importance_Score', 'Narrative']].head())
else:
    print("No stories available or ML_Importance_Score column not found")

print("\nTop 10 Detailed Metrics by Dimension:")
print(output_table_2.head())

print("\nComplete Table Sample:")
print(output_table_3.head())

# Create Table 4: ML/DL generated stories for WTD, MTD, QTD
def generate_ml_stories(df, periods, top_n=3):
    stories = []
    for period in periods:
        period_df = df[df['Period'] == period]
        if not period_df.empty:
            top_metrics = period_df.nlargest(top_n, 'ML_Importance_Score')
            for _, row in top_metrics.iterrows():
                stories.append({
                    'Period': period,
                    'Metric': row['Metric'],
                    'SLS_DIST_CHNL_TYPE_DESC': row['SLS_DIST_CHNL_TYPE_DESC'],
                    'PREPAID_IND': row['PREPAID_IND'],
                    'SEGMENT': row['SEGMENT'],
                    'Current Value': row['Current Value'],
                    'Change (%)': row['Change (%)'],
                    'Narrative': row['Narrative'],
                    'ML_Importance_Score': row['ML_Importance_Score']
                })
    return pd.DataFrame(stories)

# Generate stories for WTD, MTD, QTD
output_table_4 = generate_ml_stories(narratives_df, ['WTD', 'MTD', 'QTD'])

# Print verification
print("\nML/DL Generated Stories for WTD, MTD, QTD:")
print(output_table_4)
