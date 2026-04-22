import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load Data
df = pd.read_csv('heart.csv')

# 2. Rename Columns (Taaki web app aur model ke headers ekdam same hon)
df.rename(columns={
    'Age': 'age', 'Sex': 'sex', 'ChestPainType': 'cp',
    'RestingBP': 'trestbps', 'Cholesterol': 'chol', 'FastingBS': 'fbs',
    'RestingECG': 'restecg', 'MaxHR': 'thalach', 'ExerciseAngina': 'exang',
    'Oldpeak': 'oldpeak', 'ST_Slope': 'slope'
}, inplace=True)

# 3. Mapping (As per your logic)
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['cp'] = df['cp'].map({'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3})
df['restecg'] = df['restecg'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
df['exang'] = df['exang'].map({'N': 0, 'Y': 1})
df['slope'] = df['slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

# FastingBS ensure karein numeric hai
if df['fbs'].dtype == 'object':
    df['fbs'] = df['fbs'].map({'Normal': 0, 'High': 1})

# 4. Train-Test Split se pehle Features (X) aur Target (y) alag karein
X = df.drop('HeartDisease', axis=1) 
y = df['HeartDisease']

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 6. Save Model
joblib.dump(model, 'heart_model.pkl')
print("Model trained on 11 lowercase parameters successfully!")