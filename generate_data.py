import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
    'employee_id': np.arange(1, n_samples + 1),
    'age': np.random.randint(22, 65, n_samples),
    'years_experience': np.random.randint(0, 40, n_samples),
    'weekly_work_hours': np.random.randint(30, 70, n_samples),
    'meetings_per_week': np.random.randint(0, 30, n_samples),
    'emails_sent_per_day': np.random.randint(10, 200, n_samples),
    'projects_handled': np.random.randint(1, 10, n_samples),
    'remote_days_per_month': np.random.randint(0, 22, n_samples),
    'sleep_hours': np.random.uniform(4, 10, n_samples),
    'stress_level': np.random.randint(1, 10, n_samples),
    'exercise_hours_week': np.random.uniform(0, 10, n_samples),
    'sick_leaves_year': np.random.randint(0, 20, n_samples),
    'productivity_score': np.random.uniform(20, 100, n_samples),
}

# Generate burnout risk score with some correlation to features
burnout = (
    np.random.normal(40, 15, n_samples) +
    0.3 * data['weekly_work_hours'] -
    0.5 * data['sleep_hours'] * 5 -
    0.2 * data['exercise_hours_week'] * 5 +
    0.15 * data['stress_level'] * 5 +
    0.1 * data['meetings_per_week']
)
data['burnout_risk_score'] = np.clip(burnout, 0, 100)

# Add some zero inflation
zero_mask = np.random.random(n_samples) < 0.15
data['burnout_risk_score'][zero_mask] = 0

df = pd.DataFrame(data)
df.to_csv('Employee_Burnout.csv', index=False)
print(f'Created Employee_Burnout.csv with {len(df)} rows')
