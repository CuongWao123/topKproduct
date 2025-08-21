
import pandas as pd
import numpy as np




def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    return df



def clean_dataset(df):
    """
    Clean and preprocess the banking dataset
    
    Parameters:
    df: pandas DataFrame - Raw dataset
    
    Returns:
    df: pandas DataFrame - Cleaned dataset
    """
    missing_before = df[['payroll_final_label', 'pensions_2_final_label']].isnull().sum()
    df.fillna(value={
        'payroll_final_label': 0,
        'pensions_2_final_label': 0
    }, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'])
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    
    days_column = (df['date'] - df['registration_date']).dt.days
    
    df.insert(loc=6, column='customer_tenure_days', value=days_column)

    
    df.drop(columns=['registration_date'], inplace=True)
    
    original_nulls = df['last_primary_date'].isnull().sum()
    df['was_primary_customer'] = df['last_primary_date'].apply(
        lambda x: 1 if pd.notnull(x) else 0
    )
    

    # Drop the original last_primary_date column
    df.drop(columns=['last_primary_date'], inplace=True)
    if 'address_type' in df.columns:
        unique_values = df['address_type'].nunique()
        if unique_values <= 1:
            df.drop(columns=['address_type'], inplace=True)
        else:
            print(f"   ✅ Kept 'address_type' ({unique_values} unique values)")
    
    # Remove province_code as it's duplicate of province_name
    if 'province_code' in df.columns and 'province_name' in df.columns:
        df.drop(columns=['province_code'], inplace=True)
    


    if 'age' in df.columns:
        age_before = df['age'].dtype
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        age_nulls = df['age'].isnull().sum()
    
    # Convert seniority column - handle 'NA' strings and negative values
    if 'seniority' in df.columns:
        seniority_before = df['seniority'].dtype
        df['seniority'] = pd.to_numeric(df['seniority'], errors='coerce')
        
        # Handle special negative values (often -999999 means missing)
        negative_values = (df['seniority'] < 0).sum()
        if negative_values > 0:
            df['seniority'] = df['seniority'].where(df['seniority'] >= 0, np.nan)
        
        seniority_nulls = df['seniority'].isnull().sum()
    return df



payment_account_labels = [
    'current_accounts_final_label',
    'payroll_accounts_final_label',
    'junior_accounts_final_label',
    'more_particular_accounts_final_label',
    'particular_accounts_final_label',
    'particular_plus_accounts_final_label',
    'home_account_final_label',
    'payroll_final_label',
    'e_account_final_label'
]



customer_features = [
    'date', 'customer_id', 'employee_index', 'country_of_residence', 'gender',
    'age', 'customer_tenure_days', 'seniority', 'residence_index',  # Added customer_tenure_days
    'foreigner_index', 'spouse_index', 'channel', 'deceased_index', 
    'province_name', 'segment', 'was_primary_customer'  # Added was_primary_customer
]

def filter_data(df) :
    mask = ~(df[payment_account_labels] == -1).any(axis=1)
    columns_to_keep = customer_features + payment_account_labels
    df = df.loc[mask, columns_to_keep]
    return df


def create_enhanced_features(df):
    df_enhanced = df.copy()

    
    # Check available columns first
    available_cols = df_enhanced.columns.tolist()
    
    # Extract từ date column
    if 'date' in df_enhanced.columns:
        df_enhanced['year'] = df_enhanced['date'].dt.year
        df_enhanced['month'] = df_enhanced['date'].dt.month
        df_enhanced['quarter'] = df_enhanced['date'].dt.quarter
        df_enhanced['day_of_week'] = df_enhanced['date'].dt.dayofweek
        df_enhanced['is_weekend'] = (df_enhanced['day_of_week'] >= 5).astype(int)
        df_enhanced['is_month_end'] = (df_enhanced['date'].dt.day >= 25).astype(int)
        df_enhanced['is_quarter_end'] = df_enhanced['date'].dt.month.isin([3, 6, 9, 12]).astype(int)
    
    # Customer tenure features (using existing customer_tenure_days)
    if 'customer_tenure_days' in df_enhanced.columns:
        df_enhanced['years_since_registration'] = df_enhanced['customer_tenure_days'] / 365.25
        
        # Customer tenure categories
        df_enhanced['tenure_category'] = pd.cut(
            df_enhanced['customer_tenure_days'],
            bins=[-1, 90, 365, 1095, 2190, np.inf],
            labels=['Very_New', 'New', 'Medium', 'Long', 'Very_Long']
        )
    
    # Age-based features
    if 'age' in df_enhanced.columns:
        df_enhanced['age_group'] = pd.cut(
            df_enhanced['age'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
        
        age_young = (df_enhanced['age'] >= 18) & (df_enhanced['age'] <= 30)
        age_middle = (df_enhanced['age'] >= 31) & (df_enhanced['age'] <= 50)
        age_senior = (df_enhanced['age'] >= 60)
        
        df_enhanced['is_young_adult'] = age_young.fillna(False).astype(int)
        df_enhanced['is_middle_aged'] = age_middle.fillna(False).astype(int)
        df_enhanced['is_senior'] = age_senior.fillna(False).astype(int)
        df_enhanced['age_squared'] = df_enhanced['age'] ** 2
    
    
    # Seniority-based features
    if 'seniority' in df_enhanced.columns:
        df_enhanced['seniority_years'] = df_enhanced['seniority'] / 12
        df_enhanced['seniority_category'] = pd.cut(
            df_enhanced['seniority'],
            bins=[-1, 0, 6, 12, 24, 60, np.inf],
            labels=['New', 'Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
        )
        
        seniority_new = df_enhanced['seniority'] <= 6
        seniority_established = df_enhanced['seniority'] >= 24
        
        df_enhanced['is_new_relationship'] = seniority_new.fillna(False).astype(int)
        df_enhanced['is_established_relationship'] = seniority_established.fillna(False).astype(int)
    
    # Customer status features (check if columns exist)
    df_enhanced['is_employee'] = 0
    if 'employee_index' in df_enhanced.columns:
        df_enhanced['is_employee'] = (df_enhanced['employee_index'] == 1).astype(int)
    
    df_enhanced['is_primary_customer_flag'] = 0  
    if 'was_primary_customer' in df_enhanced.columns:
        df_enhanced['is_primary_customer_flag'] = df_enhanced['was_primary_customer']
    
    # Geographic features
    if 'country_of_residence' in df_enhanced.columns:
        domestic = df_enhanced['country_of_residence'] == 'ES'
        df_enhanced['is_domestic'] = domestic.fillna(False).astype(int)
    
    if 'foreigner_index' in df_enhanced.columns:
        foreigner = df_enhanced['foreigner_index'] == 1
        df_enhanced['is_foreigner'] = foreigner.fillna(False).astype(int)
    
    # Age-Seniority interactions
    if 'age' in df_enhanced.columns and 'seniority' in df_enhanced.columns:
        age_filled = df_enhanced['age'].fillna(0)
        seniority_filled = df_enhanced['seniority'].fillna(0)
        
        df_enhanced['age_seniority_interaction'] = (age_filled * seniority_filled) / 100
        df_enhanced['seniority_per_age'] = seniority_filled / (age_filled + 1)
    
    # Age-Tenure interactions
    if 'age' in df_enhanced.columns and 'customer_tenure_days' in df_enhanced.columns:
        age_filled = df_enhanced['age'].fillna(0)
        tenure_filled = df_enhanced['customer_tenure_days'].fillna(0)
        
        df_enhanced['age_tenure_ratio'] = age_filled / (tenure_filled/365 + 1)
    
    
    # Channel preference
    if 'channel' in df_enhanced.columns:
        channel_mapping = {
            'KAT': 'Traditional',
            'KFC': 'Phone', 
            'KHE': 'Digital',
            'KHM': 'Mobile',
            'KHN': 'Online'
        }
        df_enhanced['channel_type'] = df_enhanced['channel'].map(channel_mapping).fillna('Other')
        df_enhanced['is_digital_channel'] = df_enhanced['channel_type'].isin(['Digital', 'Mobile', 'Online']).astype(int)
    
    # Customer segment enhancement
    if 'segment' in df_enhanced.columns:
        df_enhanced['is_vip_segment'] = df_enhanced['segment'].str.contains('VIP', na=False).astype(int)
        df_enhanced['is_university_segment'] = df_enhanced['segment'].str.contains('UNIVERSITY', na=False).astype(int)
    
    
    # Customer stability score
    stability_score = 0
    
    if 'seniority' in df_enhanced.columns:
        seniority_stable = df_enhanced['seniority'] >= 12
        stability_score += seniority_stable.fillna(False).astype(int)
    
    if 'age' in df_enhanced.columns:
        age_stable = df_enhanced['age'] >= 30
        stability_score += age_stable.fillna(False).astype(int)
    
    stability_score += df_enhanced['is_primary_customer_flag']
    stability_score += df_enhanced['is_employee']
    
    df_enhanced['customer_stability_score'] = stability_score
    df_enhanced['is_stable_customer'] = (stability_score >= 2).astype(int)
    
    # Potential value score
    potential_score = 0
    
    if 'age' in df_enhanced.columns:
        age_prime = (df_enhanced['age'] >= 25) & (df_enhanced['age'] <= 55)
        potential_score += age_prime.fillna(False).astype(int)
    
    potential_score += df_enhanced['is_digital_channel'] if 'is_digital_channel' in df_enhanced.columns else 0
    potential_score += df_enhanced['is_domestic'] if 'is_domestic' in df_enhanced.columns else 0
    
    df_enhanced['customer_potential_score'] = potential_score
    df_enhanced['is_high_potential'] = (potential_score >= 2).astype(int)
    
    return df_enhanced
