import pandas as pd
import json
from pycaret.classification import load_model, predict_model

# 1. Load model and feature ranges
model = load_model('student_performance_model')
with open('feature_ranges.json') as f:
    FEATURE_RANGES = json.load(f)

# 2. Validate and prepare test data
def prepare_test_input(input_dict):
    test_data = pd.DataFrame([input_dict])
    
    # Numeric range validation
    for col in ['Age', 'StudyTimeWeekly', 'Absences']:
        val = test_data[col].iloc[0]
        min_val, max_val = FEATURE_RANGES[col]
        if not (min_val <= val <= max_val):
            raise ValueError(f"{col} must be between {min_val}-{max_val}, got {val}")
    
    # Categorical validation and conversion
    for col, mapping in FEATURE_RANGES.items():
        if isinstance(mapping, dict) and col in test_data:
            val = test_data[col].iloc[0]
            if val not in mapping:
                raise ValueError(f"Invalid {col}: {val}. Must be one of: {list(mapping.keys())}")
            test_data[col] = test_data[col].map(mapping)
    
    return test_data

# 3. Example test case
# test_input = {
#     'Age': 17,
#     'Gender': '1',
#     'Ethnicity': '0',
#     'ParentalEducation': '2',
#     'StudyTimeWeekly': 15.5,
#     'Absences': 3,
#     'Tutoring': '1',
#     'ParentalSupport': '2',
#     'Extracurricular': '0',
#     'Sports': '1',
#     'Music': '0',
#     'Volunteering': '1'
# }

# try:
#     test_data = prepare_test_input(test_input)
#     predictions = predict_model(model, data=test_data)
    
#     print("\nPrediction Successful!")
#     print("Input:", test_input)
#     print("Predicted Grade:", predictions['prediction_label'].iloc[0])
#     print("Confidence:", f"{predictions['prediction_score'].iloc[0]*100:.2f}%")
    
# except Exception as e:
#     print("\nPrediction Failed!")
#     print("Error:", str(e))











#    import pandas as pd

def prepare_test_input(input_dict: dict) -> pd.DataFrame:
    """
    Casts specific fields to correct types and returns a single-row DataFrame.

    Expected input format:
        - Age: int
        - Gender, Ethnicity, ParentalEducation, Tutoring, ParentalSupport,
          Extracurricular, Sports, Music, Volunteering: str
        - StudyTimeWeekly: float
        - Absences: int
    """
    try:
        processed = {
            'Age': int(input_dict['Age']),
            'Gender': str(input_dict['Gender']),
            'Ethnicity': str(input_dict['Ethnicity']),
            'ParentalEducation': str(input_dict['ParentalEducation']),
            'StudyTimeWeekly': float(input_dict['StudyTimeWeekly']),
            'Absences': int(input_dict['Absences']),
            'Tutoring': str(input_dict['Tutoring']),
            'ParentalSupport': str(input_dict['ParentalSupport']),
            'Extracurricular': str(input_dict['Extracurricular']),
            'Sports': str(input_dict['Sports']),
            'Music': str(input_dict['Music']),
            'Volunteering': str(input_dict['Volunteering']),
        }
        return pd.DataFrame([processed])
    
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid input or type conversion failed: {e}")
 


test_input = {
    'Age': 15,
    'Gender': 1,
    'Ethnicity': '0',
    'ParentalEducation': '2',
    'StudyTimeWeekly': 15.5,
    'Absences': 3,
    'Tutoring': '1',
    'ParentalSupport': '2',
    'Extracurricular': '0',
    'Sports': '1',
    'Music': '0',
    'Volunteering': '1'
}

try:
    test_data = prepare_test_input(test_input)
    predictions = predict_model(model, data=test_data)
    # print(predictions)
    print("\nPrediction Successful!")
    print("Input DataFrame:\n", test_data)
    print("Predicted Grade:", predictions['prediction_label'].iloc[0])
    print("Confidence:", f"{predictions['prediction_score'].iloc[0]*100:.2f}%")

except Exception as e:
    print("\nPrediction Failed!")
    print("Error:", str(e))
