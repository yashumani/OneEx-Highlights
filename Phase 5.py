


# Phase 5 Code: Restructuring Tables 1, 2, 3, and 4

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Assuming 'input_table_1' is the KNIME input data (a pandas DataFrame)
df = input_table_1.copy()

# Define the columns explicitly
date_column = 'DATE'
dimensions = ['SLS_DIST_CHNL_TYPE_DESC', 'PREPAID_IND', 'SEGMENT']  # Example dimension columns
#measures = ['GROSS_ADDS_TOTAL', 'GROSS_ADDS_PHONE', 'NET_ADDS_TOTAL', 'NET_ADDS_PHONE']  # Example measure columns
#negative_measures = ['NET_DEACTS_TOTAL', 'NET_DEACTS_PHONE']  # Columns that should only have negative values


# Define metrics columns
measures = [
    'GROSS_ADDS_TOTAL',  'NET_ADDS_TOTAL', 'GROSS_ADDS_PHONE', 
     'NET_ADDS_PHONE',  'GROSS_ADDS_TABLET', 'NET_DEACTS_TABLET', 'NET_ADDS_TABLET', 
     'GROSS_ADDS_OTHER',    'NET_DEACTS_OTHER', 'NET_ADDS_OTHER', 'UPGRADES', 'UPGRADES_PHONE'
     ,'NEW_ACCOUNTS',    'UNLIMITED_NEW_ACCOUNTS', 'NEW_ACCOUNTS_SINGLE_PHONE', 'NEW_ACCOUNTS_MULTI_PHONE', 
    'LOST_ACCOUNTS', 'UPGRADES_TABLET', 'UPGRADES_APPLE_SMARTWATCH', 
    'UPGRADES_OTHER', 'VSB_GROSS_ADDS',  'VSB_NET_ADDS',  'GROSS_ADDS_TOTAL_FWA', 'NET_ADDS_TOTAL_FWA', 
     'TF_GROSS_ADDS_TOTAL', 'TF_NET_DEACTS_TOTAL', 
    'TF_NET_ADDS_TOTAL', 'GROSS_ADDS_OTHER_WEARABLES', 'NET_DEACTS_OTHER_WEARABLES', 
    'NET_ADDS_OTHER_WEARABLES', 'UPGRADES_OTHER_WEARABLES', 'CHURN_POSTPAID_FWA_BASE_TOTAL', 
    'GROSS_ADDS_PHONE_AAL', 'GROSS_ADDS_PHONE_NEW', 'NEW_ACCOUNTS_PHONE_PREMIUM', 
    'NEW_ACCOUNTS_FWA_STAND_ALONE', 'TF_SL_GROSS_ADDS', 'TF_SL_NET_ADDS', 
    'GROSS_ADDS_2ND_LINE'
]
# Define reverse logic metrics
negative_measures = [
    'NET_DEACTS_TOTAL', 'NET_DEACTS_PHONE', 'NET_DEACTS_SMARTPHONE', 'NET_DEACTS_TABLET'
    ,'VSB_NET_DEACTS','VOLUNTARY_POSTPAID_DISCOS', 'INVOLUNTARY_POSTPAID_DISCOS','NET_DEACTS_2ND_LINE', 'TF_SL_NET_DEACTS'
    , 'VOL_POSTPAID_PHONE_DISCOS', 'VOL_POSTPAID_TABLET_DISCOS','NET_DEACTS_TOTAL_FWA', 
    'VOL_POSTPAID_OTHER_DISCOS', 'INVOL_POSTPAID_PHONE_DISCOS', 'INVOL_POSTPAID_TABLET_DISCOS', 
    'INVOL_POSTPAID_OTHER_DISCOS'
]
# Ensure the specified columns exist
required_columns = [date_column] + dimensions + measures + negative_measures
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"The specified columns {missing_columns} do not exist in the DataFrame")

# Ensure other dimensions are converted to strings
df[dimensions] = df[dimensions].astype(str)

# Convert date column to datetime format
df[date_column] = pd.to_datetime(df[date_column])

# Sort by date
df = df.sort_values(by=date_column)

# Define date ranges for WTD, MTD, and QTD, accounting for leap year
max_date = df[date_column].max()

# Check if the current year is a leap year
is_leap_year = (max_date.year % 4 == 0 and (max_date.year % 100 != 0 or max_date.year % 400 == 0))

wtd_start_date = max_date - timedelta(days=max_date.weekday())  # Start from the previous Sunday
if wtd_start_date > max_date:
    wtd_start_date -= timedelta(weeks=1)  # Ensure WTD starts from the correct week
mtd_start_date = max_date.replace(day=1)
current_quarter = (max_date.month - 1) // 3 + 1
qtd_start_date = pd.Timestamp(max_date.year, 3 * (current_quarter - 1) + 1, 1)

# Adjust for leap year in QTD calculations if February is involved
if current_quarter == 1 and is_leap_year and max_date.month == 2:
    qtd_start_date = pd.Timestamp(max_date.year, 2, 29)

# Filter data for each period and add period labels
df['Period'] = 'ALL'
df.loc[(df[date_column] >= wtd_start_date) & (df['Period'] == 'ALL'), 'Period'] = 'WTD'
df.loc[(df[date_column] >= mtd_start_date) & (df['Period'] == 'ALL'), 'Period'] = 'MTD'
df.loc[(df[date_column] >= qtd_start_date) & (df['Period'] == 'ALL'), 'Period'] = 'QTD'

# Create "ALL" categories for each dimension
for dimension in dimensions:
    df[dimension] = df[dimension].astype(str)
    df = pd.concat([df, df.assign(**{dimension: 'ALL'})], ignore_index=True)

# Filter data for "ALL" periods for Table 1
df_all_periods = df[df['Period'] == 'ALL']

# Define the threshold calculation function
def calculate_thresholds(df, measure):
    lower_2 = df[measure].quantile(0.02)
    upper_98 = df[measure].quantile(0.98)
    lower_05 = df[measure].quantile(0.005)
    upper_995 = df[measure].quantile(0.995)
    return {
        'lower_2': lower_2,
        'upper_98': upper_98,
        'lower_05': lower_05,
        'upper_995': upper_995
    }

# Function to determine anomaly type based on thresholds
def determine_anomaly_type(df, measure, thresholds, negative_only):
    df[f'{measure}_Anomaly_Type'] = 'Normal'
    df.loc[df[measure] < thresholds['lower_05'], f'{measure}_Anomaly_Type'] = 'More Extreme Outlier'
    df.loc[df[measure] > thresholds['upper_995'], f'{measure}_Anomaly_Type'] = 'More Extreme Outlier'
    df.loc[(df[measure] < thresholds['lower_2']) & (df[measure] >= thresholds['lower_05']), f'{measure}_Anomaly_Type'] = 'Less Extreme Outlier'
    df.loc[(df[measure] > thresholds['upper_98']) & (df[measure] <= thresholds['upper_995']), f'{measure}_Anomaly_Type'] = 'Less Extreme Outlier'
    if negative_only:
        df.loc[df[measure] > 0, f'{measure}_Anomaly_Type'] = 'Unexpected Positive Value'
    return df

def calculate_seasonal_thresholds(df, measure, period):
    """Calculate thresholds based on seasonality and period"""
    if period == 'ALL':
        data = df[measure]
    else:
        data = df[df['Period'] == period][measure]
    
    return {
        'lower_2': data.quantile(0.02),
        'upper_98': data.quantile(0.98),
        'lower_05': data.quantile(0.005),
        'upper_995': data.quantile(0.995)
    }

# Process for Table 1
df_table_1 = df.copy()
thresholds = []
processed_rows = []

# Process each measure separately
for measure in measures + negative_measures:
    negative_only = measure in negative_measures
    measure_data = df_table_1.copy()
    measure_data['Metric'] = measure  # Add Metric column
    
    # Process each period
    for period in ['ALL', 'WTD', 'MTD', 'QTD']:
        period_data = measure_data[measure_data['Period'] == period].copy()
        
        # Group by dimension combinations
        for dimension_values, group in period_data.groupby(dimensions):
            # Calculate seasonal thresholds
            group_thresholds = calculate_seasonal_thresholds(group, measure, period)
            
            # Create threshold record
            threshold_record = {
                'Metric': measure,
                'Period': period,
                **dict(zip(dimensions, dimension_values)),
                **group_thresholds
            }
            thresholds.append(threshold_record)
            
            # Apply anomaly detection
            group = determine_anomaly_type(group, measure, group_thresholds, negative_only)
            processed_rows.append(group)

# Create Table 1
df_table_1 = pd.concat(processed_rows, ignore_index=True)

# Convert thresholds list to DataFrame
thresholds_df = pd.DataFrame(thresholds)

# Merge thresholds into Table 1
merge_columns = dimensions + ['Metric', 'Period']
df_table_1 = df_table_1.merge(
    thresholds_df,
    how='left',
    on=merge_columns,
    validate='many_to_one'
)

# Apply clustering
kmeans_features = measures + negative_measures
kmeans = KMeans(n_clusters=3, random_state=42)
df_table_1['Cluster'] = kmeans.fit_predict(df_table_1[kmeans_features].fillna(0))

# Map cluster labels
cluster_labels = {
    0: 'Low Activity',
    1: 'Medium Activity',
    2: 'High Activity'
}
df_table_1['Cluster_Label'] = df_table_1['Cluster'].map(cluster_labels)

# Select columns for Table 1
output_columns = (
    [date_column] +
    dimensions + 
    ['Period'] +
    ['Metric'] + 
    [f'{measure}_Anomaly_Type' for measure in measures + negative_measures] +
    ['lower_2', 'upper_98', 'lower_05', 'upper_995', 'Cluster_Label']
)

# Ensure all required columns exist
for col in output_columns:
    if col not in df_table_1.columns:
        df_table_1[col] = None

# Filter Table 1 for max date
max_date = df[date_column].max()
df_table_1_filtered = df_table_1[df_table_1[date_column] == max_date].copy()

# Calculate date ranges for thresholds
date_ranges = df_table_1.groupby(dimensions + ['Metric']).agg({
    date_column: ['min', 'max']
}).reset_index()
date_ranges.columns = dimensions + ['Metric', 'From_Date', 'To_Date']

# Pivot anomaly types
anomaly_columns = [f'{measure}_Anomaly_Type' for measure in measures + negative_measures]
pivot_table = df_table_1.pivot_table(
    index=dimensions + ['Metric', 'Period'],
    columns=anomaly_columns,
    aggfunc='size',
    fill_value=0
).reset_index()

# Prepare Table 1 with new structure
output_columns = (
    dimensions + 
    ['Period', 'Metric', 'From_Date', 'To_Date'] +
    ['lower_2', 'upper_98', 'lower_05', 'upper_995']
)

# Calculate date ranges for thresholds by period
def get_period_date_range(df, period, date_column, wtd_start_date, mtd_start_date, qtd_start_date):
    if period == 'ALL':
        return df[date_column].min(), df[date_column].max()
    elif period == 'WTD':
        return wtd_start_date, df[date_column].max()
    elif period == 'MTD':
        return mtd_start_date, df[date_column].max()
    elif period == 'QTD':
        return qtd_start_date, df[date_column].max()
    return None, None

# Calculate period-specific date ranges
date_ranges = []
for _, group in df_table_1.groupby(dimensions + ['Metric', 'Period']):
    period = group['Period'].iloc[0]
    from_date, to_date = get_period_date_range(
        df, 
        period,
        date_column,
        wtd_start_date,
        mtd_start_date,
        qtd_start_date
    )
    
    date_ranges.append({
        **{dim: group[dim].iloc[0] for dim in dimensions},
        'Metric': group['Metric'].iloc[0],
        'Period': period,
        'From_Date': from_date,
        'To_Date': to_date
    })

# Convert date ranges to DataFrame
date_ranges_df = pd.DataFrame(date_ranges)

# Create new Table 1 with period-specific ranges
df_table_1_new = df_table_1.merge(
    date_ranges_df,
    on=dimensions + ['Metric', 'Period']
)

# Select and order columns for final output
base_columns = (
    dimensions + 
    ['Period', 'Metric', 'From_Date', 'To_Date'] +
    ['lower_2', 'upper_98', 'lower_05', 'upper_995']
)

# Add anomaly flags and details
for measure in measures + negative_measures:
    anomaly_col = f'{measure}_Anomaly_Type'
    if anomaly_col in df_table_1_new.columns:
        df_table_1_new[f'{measure}_Has_Anomaly'] = df_table_1_new[anomaly_col].ne('Normal').astype(int)
        df_table_1_new[f'{measure}_Anomaly_Details'] = df_table_1_new[anomaly_col]

# Create final output_table_1
output_table_1 = df_table_1_new[
    base_columns + 
    [col for col in df_table_1_new.columns if '_Has_Anomaly' in col or '_Anomaly_Details' in col]
].drop_duplicates().sort_values(dimensions + ['Period', 'Metric'])

# Format dates in output_table_1
date_cols = ['From_Date', 'To_Date']
for col in date_cols:
    output_table_1[col] = pd.to_datetime(output_table_1[col]).dt.strftime('%Y-%m-%d')

# Table 2: Original structure with clustering
df['Cluster'] = kmeans.predict(df[kmeans_features].fillna(0))
df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
output_table_2 = df[df[date_column] >= mtd_start_date].copy()
output_table_2[date_column] = output_table_2[date_column].dt.strftime('%Y-%m-%d')


# Table 3: Enhanced with dimensions and renamed threshold columns
df_table_3 = thresholds_df.groupby(['Metric'] + dimensions).agg({
    'lower_2': ['min', 'max'],
    'upper_98': ['min', 'max'],
    'lower_05': ['min', 'max'],
    'upper_995': ['min', 'max']
}).reset_index()

# Rename threshold columns with more descriptive names
df_table_3.columns = ['Metric'] + dimensions + [
    'Lower_2nd_Percentile_Min', 'Lower_2nd_Percentile_Max',
    'Upper_98th_Percentile_Min', 'Upper_98th_Percentile_Max',
    'Lower_Point5_Percentile_Min', 'Lower_Point5_Percentile_Max',
    'Upper_99Point5_Percentile_Min', 'Upper_99Point5_Percentile_Max'
]
output_table_3 = df_table_3

# Table 4: Enhanced Clustering details with detailed rows
# First apply clustering to get cluster assignments
df['Cluster'] = kmeans.predict(df[kmeans_features].fillna(0))
df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

# Create long format table with measures
measure_df = pd.melt(
    df,
    id_vars=['Cluster', 'Cluster_Label', date_column] + dimensions + ['Period'],
    value_vars=measures + negative_measures,
    var_name='Metric',
    value_name='Value'
)

# Sort and organize the data
output_table_4 = measure_df.sort_values(
    ['Cluster_Label', 'Cluster', date_column] + dimensions + ['Metric']
).reset_index(drop=True)

# Add measure statistics by cluster
measure_stats = df.groupby('Cluster').agg({
    **{measure: ['mean', 'std', 'min', 'max'] 
       for measure in measures + negative_measures}
}).reset_index()


#QTD filtered output
output_table_4 = output_table_4[output_table_4[date_column] >= mtd_start_date].copy()

# Format date column
output_table_4[date_column] = pd.to_datetime(output_table_4[date_column]).dt.strftime('%Y-%m-%d')

# Select and order final columns
output_columns = [
    'Cluster_Label',
    'Cluster',
    date_column,
    'Period'
] + dimensions + ['Metric', 'Value']

output_table_4 = output_table_4[output_columns]

