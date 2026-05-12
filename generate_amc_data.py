import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_amc_data(num_rows=1200):
    np.random.seed(42)
    random.seed(42)

    brands = ['Daikin', 'Voltas', 'Blue Star', 'LG', 'Samsung', 'Carrier']
    eq_types = ['Split', 'Cassette', 'Central']
    tiers = ['Basic', 'Standard', 'Premium']
    cities = ['Nagpur', 'Pune', 'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata']
    
    # Adding more professional/B2B or realistic B2C names
    first_names = ['Amit', 'Rahul', 'Sanjay', 'Priya', 'Anjali', 'Vikram', 'Suresh', 'Meena', 'Deepak', 'Rohan',
                   'Neha', 'Kiran', 'Pooja', 'Arun', 'Manish', 'Sneha', 'Rajesh', 'Aarti', 'Gaurav', 'Divya']
    last_names = ['Sharma', 'Verma', 'Patil', 'Deshmukh', 'Joshi', 'Kulkarni', 'Singh', 'Nair', 'Iyer', 'Gupta',
                  'Mehta', 'Reddy', 'Bose', 'Chandra', 'Tiwari', 'Pandey', 'Mishra', 'Das', 'Rao', 'Kapoor']

    data = []
    today = datetime(2025, 5, 12)

    for i in range(num_rows):
        customer_id = f'C{i+1:04d}'
        customer_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        city = random.choice(cities)
        
        # Equipment info
        equipment_age_years = np.random.randint(1, 15)
        equipment_brand = random.choice(brands)
        equipment_type = np.random.choice(eq_types, p=[0.7, 0.2, 0.1])
        
        # Contract info — variable duration (6 months to 2 years)
        contract_tier = np.random.choice(tiers, p=[0.5, 0.3, 0.2])
        contract_duration_months = np.random.choice([6, 12, 18, 24], p=[0.1, 0.6, 0.15, 0.15])
        contract_duration_days = int(contract_duration_months * 30.44)
        
        # Standardize contract values to look like real AMC packages (rounded to 500)
        if contract_tier == 'Basic':
            contract_value_inr = int(np.round(np.random.randint(5000, 12000) / 500.0) * 500)
        elif contract_tier == 'Standard':
            contract_value_inr = int(np.round(np.random.randint(10000, 22000) / 500.0) * 500)
        else:
            contract_value_inr = int(np.round(np.random.randint(18000, 50000) / 500.0) * 500)

        # days_to_expiry: from -30 (already expired) to 365
        days_to_expiry = np.random.randint(-30, 365)
        
        contract_end_date = today + timedelta(days=int(days_to_expiry))
        contract_start_date = contract_end_date - timedelta(days=contract_duration_days)
        installation_date = today - timedelta(days=365 * int(equipment_age_years) + np.random.randint(0, 180))

        # Renewal history
        previous_renewals = np.random.randint(0, 10)
        
        # Service history (correlated with equipment age)
        base_calls = max(0, equipment_age_years - 2)
        total_service_calls = np.random.randint(0, base_calls * 3 + 3)
        
        if total_service_calls > 0:
            unresolved_complaints = np.random.randint(0, min(total_service_calls, 6) + 1)
            avg_resolution_time_days = round(np.random.uniform(0.5, 10.0), 1)
            service_recency = np.random.randint(5, 350)
            last_service_date = today - timedelta(days=service_recency)
            missed_scheduled_visits = np.random.randint(0, 4)
            repeat_complaints = bool(np.random.choice([True, False], p=[0.3, 0.7]))
        else:
            unresolved_complaints = 0
            avg_resolution_time_days = 0.0
            last_service_date = pd.NaT
            missed_scheduled_visits = 0
            repeat_complaints = False
            
        renewal_reminder_sent = bool(days_to_expiry < 30 and np.random.random() < 0.8)
        if previous_renewals > 0:
            last_renewal_delay_days = np.random.randint(-10, 45)
        else:
            last_renewal_delay_days = 0

        # --- Churn probability calculation ---
        churn_prob = 0.12  # base

        # High churn signals
        if equipment_age_years > 5:
            churn_prob += 0.12
        if equipment_age_years > 10:
            churn_prob += 0.08
        if unresolved_complaints > 2:
            churn_prob += 0.18
        if avg_resolution_time_days > 4:
            churn_prob += 0.10
        if days_to_expiry < 30:
            churn_prob += 0.08
        if previous_renewals == 0:
            churn_prob += 0.10
        if missed_scheduled_visits >= 2:
            churn_prob += 0.06
        if repeat_complaints:
            churn_prob += 0.05
            
        # Seasonality Rule: If it's April/May (peak cooling season) and they haven't renewed (days_to_expiry < 15), huge risk.
        if contract_end_date.month in [4, 5] and days_to_expiry < 15:
            churn_prob += 0.15
            
        # Low churn signals
        if previous_renewals > 2:
            churn_prob -= 0.12
        if previous_renewals > 5:
            churn_prob -= 0.05
        if unresolved_complaints == 0:
            churn_prob -= 0.08
        if contract_value_inr > 25000:
            churn_prob -= 0.08
        if contract_tier == 'Premium':
            churn_prob -= 0.05
            
        # Noise
        churn_prob += np.random.uniform(-0.08, 0.08)
        churn_prob = max(0.02, min(0.92, churn_prob))
        
        churn = 1 if random.random() < churn_prob else 0
        
        # --- Churn reason ---
        if churn == 1:
            weights = {
                'Service Quality': 1.0,
                'Price': 1.0,
                'Equipment Replaced': 1.0,
                'Competitor': 1.0
            }
            if unresolved_complaints > 2 or avg_resolution_time_days > 5 or missed_scheduled_visits > 0:
                weights['Service Quality'] += 4.0
            if contract_value_inr > 18000 and contract_tier == 'Basic':
                weights['Price'] += 3.0
            if equipment_age_years > 8:
                weights['Equipment Replaced'] += 4.0
            if days_to_expiry < 15 and previous_renewals <= 1:
                weights['Competitor'] += 2.0
            if contract_end_date.month in [4, 5]: # Peak season competitor poaching
                weights['Competitor'] += 2.0
            
            reason_list = list(weights.keys())
            reason_weights = np.array(list(weights.values()))
            reason_weights = reason_weights / reason_weights.sum()
            churn_reason = np.random.choice(reason_list, p=reason_weights)
            
            # estimated_churn_date within 60 days of contract end
            churn_date_offset = np.random.randint(-30, 60)
            estimated_churn_date = contract_end_date + timedelta(days=int(churn_date_offset))
        else:
            churn_reason = None
            estimated_churn_date = pd.NaT
        
        data.append({
            'customer_id': customer_id,
            'customer_name': customer_name,
            'city': city,
            'contract_start_date': contract_start_date,
            'contract_end_date': contract_end_date,
            'contract_value_inr': contract_value_inr,
            'contract_tier': contract_tier,
            'equipment_brand': equipment_brand,
            'equipment_age_years': equipment_age_years,
            'equipment_type': equipment_type,
            'installation_date': installation_date,
            'total_service_calls': total_service_calls,
            'avg_resolution_time_days': avg_resolution_time_days,
            'unresolved_complaints': unresolved_complaints,
            'last_service_date': last_service_date,
            'missed_scheduled_visits': missed_scheduled_visits,
            'repeat_complaints': repeat_complaints,
            'previous_renewals': previous_renewals,
            'days_to_expiry': days_to_expiry,
            'renewal_reminder_sent': renewal_reminder_sent,
            'last_renewal_delay_days': last_renewal_delay_days,
            'churn': churn,
            'churn_reason': churn_reason,
            'estimated_churn_date': estimated_churn_date
        })

    df = pd.DataFrame(data)
    
    # Format dates to string
    date_cols = ['contract_start_date', 'contract_end_date', 'installation_date', 'last_service_date', 'estimated_churn_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
        
    return df

if __name__ == '__main__':
    df = generate_amc_data(1200)
    print(f"Total Rows: {len(df)}")
    print(f"Churn Rate: {df['churn'].mean():.2%}")
    df.to_csv('amc_synthetic_data.csv', index=False)
    print("\nSaved to amc_synthetic_data.csv")
