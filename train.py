import pandas as pd
import torch
import os
import json
from pycaret.classification import setup, compare_models, save_model

# 1. Load and validate data
data = pd.read_csv("Student_performance_data _.csv")

# 2. Define value ranges and mappings (EXCLUDING GradeClass from features)
VALUE_RANGES = {
    # Features
    'Age': (15, 18),
    'Gender': {0: 'female', 1: 'male'},
    'Ethnicity': {0: 'white', 1: 'black', 2: 'asian', 3: 'hispanic'},
    'ParentalEducation': {
        0: 'none', 1: 'high_school', 2: 'college', 
        3: 'bachelor', 4: 'graduate'
    },
    'StudyTimeWeekly': (0.0, 20.0),
    'Absences': (0, 30),
    'Tutoring': {0: 'no', 1: 'yes'},
    'ParentalSupport': {
        0: 'none', 1: 'minimal', 2: 'moderate',
        3: 'strong', 4: 'very_strong'
    },
    'Extracurricular': {0: 'no', 1: 'yes'},
    'Sports': {0: 'no', 1: 'yes'},
    'Music': {0: 'no', 1: 'yes'},
    'Volunteering': {0: 'no', 1: 'yes'},
    
    # Target only
    'GradeClass_target': {0: 'fail', 1: 'pass', 2: 'good', 3: 'very_good', 4: 'excellent'}
}

# 3. Validate and convert data
def prepare_data(df):
    # Validate ranges for numeric features
    for col, (min_val, max_val) in {
        'Age': VALUE_RANGES['Age'],
        'StudyTimeWeekly': VALUE_RANGES['StudyTimeWeekly'],
        'Absences': VALUE_RANGES['Absences']
    }.items():
        df[col] = pd.to_numeric(df[col])
        if not df[col].between(min_val, max_val).all():
            invalid = df[~df[col].between(min_val, max_val)][col].unique()
            raise ValueError(f"Invalid {col} values: {invalid}. Must be between {min_val}-{max_val}")

    # Convert categorical features (EXCLUDING target)
    for col, mapping in VALUE_RANGES.items():
        if isinstance(mapping, dict) and col in df and col != 'GradeClass_target':
            df[col] = df[col].map(mapping).astype('category')
    
    # Convert target separately
    if 'GradeClass' in df:
        df['GradeClass'] = df['GradeClass'].map(VALUE_RANGES['GradeClass_target']).astype('category')
    
    return df

data = prepare_data(data)

# 4. Setup PyCaret with correct feature specification
clf = setup(
    data=data,
    target='GradeClass',  # This is our target, not a feature
    session_id=210,
    numeric_features=['Age', 'StudyTimeWeekly', 'Absences'],
    categorical_features=[k for k in VALUE_RANGES 
                         if isinstance(VALUE_RANGES[k], dict) 
                         and k != 'GradeClass_target'],
    ignore_features=['StudentID', 'GPA'],
    ordinal_features={
        'ParentalEducation': VALUE_RANGES['ParentalEducation'].values(),
        'ParentalSupport': VALUE_RANGES['ParentalSupport'].values()
        # GradeClass is handled automatically as target
    },
    use_gpu=True,
    verbose=False
)

# 5. Train and save model
best_model = compare_models()
save_model(best_model, 'student_performance_model')

# Save value ranges (without target for feature validation)
feature_ranges = {k: v for k, v in VALUE_RANGES.items() if k != 'GradeClass_target'}
with open('feature_ranges.json', 'w') as f:
    json.dump(feature_ranges, f)

print("Model trained successfully with proper feature/target separation")