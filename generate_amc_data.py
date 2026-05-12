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

    first_names = ['Amit', 'Rahul', 'Sanjay', 'Priya', 'Anjali', 'Vikram', 'Suresh', 'Meena', 'Deepak', 'Rohan']
    last_names = ['Sharma', 'Verma', 'Patil', 'Deshmukh', 'Joshi', 'Kulkarni', 'Singh', 'Nair', 'Iyer', 'Gupta']

    data = []

    current_date = datetime(2025, 5, 12)

    for i in range(num_rows):
        customer_id = f'C{i+1:04d}'
        customer_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        # Base features
        equipment_age_years = np.random.randint(1, 15)
        contract_tier = np.random.choice(tiers, p=[0.5, 0.3, 0.2])
        
        # Base values for contract
        if contract_tier == 'Basic':
            contract_value_inr = np.random.randint(5000, 10000)
        elif contract_tier == 'Standard':
            contract_value_inr = np.random.randint(10000, 20000)
        else:
            contract_value_inr = np.random.randint(20000, 50000)

        previous_renewals = np.random.randint(0, 10)
        
        # Derive days_to_expiry. Most AMC contracts are 1 year.
        # Let's say days_to_expiry ranges from -30 to 365
        days_to_expiry = np.random.randint(-30, 365)
        
        contract_end_date = current_date + timedelta(days=int(days_to_expiry))
        contract_start_date = contract_end_date - timedelta(days=365)
        installation_date = contract_end_date - timedelta(days=365 * int(equipment_age_years))
        
        # Service history based somewhat on age and random chance
        total_service_calls = np.random.randint(0, equipment_age_years * 3 + 2)
        
        if total_service_calls > 0:
            unresolved_complaints = np.random.randint(0, min(total_service_calls, 5) + 1)
            avg_resolution_time_days = round(np.random.uniform(0.5, 10.0), 1)
            last_service_date = contract_end_date - timedelta(days=int(np.random.randint(10, 300)))
            missed_scheduled_visits = np.random.randint(0, 3)
            repeat_complaints = np.random.choice([True, False], p=[0.3, 0.7])
        else:
            unresolved_complaints = 0
            avg_resolution_time_days = 0.0
            last_service_date = pd.NaT
            missed_scheduled_visits = 0
            repeat_complaints = False
            
        renewal_reminder_sent = bool(days_to_expiry < 30 and np.random.choice([True, False], p=[0.8, 0.2]))
        if previous_renewals > 0:
            last_renewal_delay_days = np.random.randint(-10, 30)
        else:
            last_renewal_delay_days = 0

        # Calculate churn probability based on rules
        churn_prob = 0.14 # base probability
        
        if equipment_age_years > 5:
            churn_prob += 0.15
        if unresolved_complaints > 2:
            churn_prob += 0.2
        if avg_resolution_time_days > 4:
            churn_prob += 0.1
        if days_to_expiry < 30:
            churn_prob += 0.1
        if previous_renewals == 0:
            churn_prob += 0.1
            
        if previous_renewals > 2:
            churn_prob -= 0.15
        if unresolved_complaints == 0:
            churn_prob -= 0.1
        if contract_value_inr > 25000:
            churn_prob -= 0.1
            
        # Add some noise
        churn_prob += np.random.uniform(-0.1, 0.1)
        
        # Cap probability between 0.01 and 0.95
        churn_prob = max(0.01, min(0.95, churn_prob))
        
        churn = 1 if random.random() < churn_prob else 0
        
        # Set churn_reason and estimated_churn_date
        if churn == 1:
            if unresolved_complaints > 2 or avg_resolution_time_days > 4:
                churn_reason = 'Service Quality'
            elif equipment_age_years > 8:
                churn_reason = 'Equipment Replaced'
            else:
                churn_reason = random.choice(['Price', 'Competitor', 'Service Quality'])
            
            # estimated_churn_date within 60 days of contract end
            churn_date_offset = np.random.randint(-60, 60)
            estimated_churn_date = contract_end_date + timedelta(days=int(churn_date_offset))
        else:
            churn_reason = None
            estimated_churn_date = pd.NaT
            
        equipment_brand = random.choice(brands)
        equipment_type = np.random.choice(eq_types, p=[0.7, 0.2, 0.1])
        city = random.choice(cities)
        
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
        # Replace NaT with None so it translates to empty/null in CSV gracefully if needed, or we can just leave it
        
    return df

if __name__ == '__main__':
    df = generate_amc_data(1200)
    print(f"Total Rows: {len(df)}")
    print(f"Churn Rate: {df['churn'].mean():.2%}")
    df.to_csv('amc_synthetic_data.csv', index=False)
    print("Saved to amc_synthetic_data.csv")
