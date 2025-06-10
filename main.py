from fastapi import FastAPI, Query, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Dict, Any, Optional
from pycaret.classification import *
from pydantic import BaseModel
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
from pycaret.classification import *
from pydantic import BaseModel
import pandas as pd
import json
import json
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

# Configuration
SECRET_KEY = "your-secret-key-here"  # Change this to a strong secret key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Mock user database (in production, use a real database)
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # password = "secret"
        "disabled": False,
    }
}

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()
model = load_model("student_performance_model")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class Student(BaseModel):
    id: int
    name: str
    grade: int

students = [
    Student(id=1, name="ali", grade=150), 
    Student(id=2, name="nesma", grade=120),
]

with open('feature_ranges.json') as f:
    FEATURE_RANGES = json.load(f)

GRADE_MAP = {
    0: "fail (0-59%)",
    1: "pass (60-69%)",
    2: "good (70-79%)",
    3: "very good (80-89%)",
    4: "excellent (90-100%)",
    -1: "unknown (error)",
    5: "exceptional (100+%)"
}

# Authentication utility functions
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Authentication endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# Your existing utility function
def prepare_test_input(input_dict: dict) -> pd.DataFrame:
    try:
        processed = {
            'Age': int(input_dict['Age']),
            'Gender': int(input_dict['Gender']),
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

# Protected endpoints
@app.get("/predict", response_model=Dict[str, Any])
async def predict_grade(
    current_user: User = Depends(get_current_active_user),
    Age: int = Query(..., ge=15, le=18),
    Gender: int = Query(..., ge=0, le=1),
    Ethnicity: str = Query(..., pattern="^[0-3]$"),
    ParentalEducation: str = Query(..., pattern="^[0-4]$"),
    StudyTimeWeekly: float = Query(..., ge=0.0, le=20.0),
    Absences: int = Query(..., ge=0, le=30),
    Tutoring: str = Query(..., pattern="^[0-1]$"),
    ParentalSupport: str = Query(..., pattern="^[0-4]$"),
    Extracurricular: str = Query(..., pattern="^[0-1]$"),
    Sports: str = Query(..., pattern="^[0-1]$"),
    Music: str = Query(..., pattern="^[0-1]$"),
    Volunteering: str = Query(..., pattern="^[0-1]$")
) -> JSONResponse:
    try:
        input_data = {
            'Age': Age,
            'Gender': Gender,
            'Ethnicity': Ethnicity,
            'ParentalEducation': ParentalEducation,
            'StudyTimeWeekly': StudyTimeWeekly,
            'Absences': Absences,
            'Tutoring': Tutoring,
            'ParentalSupport': ParentalSupport,
            'Extracurricular': Extracurricular,
            'Sports': Sports,
            'Music': Music,
            'Volunteering': Volunteering
        }

        test_data = prepare_test_input(input_data)
        predictions = predict_model(model, data=test_data)
        predicted_grade = predictions["prediction_label"].iloc[0] 
        response_data = {
            "expected_grade": predicted_grade,
            "confidence_score": float(predictions["prediction_score"].iloc[0]),
            "input_values": input_data
        }

        return JSONResponse(
            content=response_data,
            media_type="application/json",
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={
                "error": "Prediction failed",
                "detail": str(e),
                "suggestion": "Please check your input values against the expected ranges"
            },
            media_type="application/json",
            status_code=400
        )

@app.get("/students/")
def read_students(current_user: User = Depends(get_current_active_user)):
    return students

@app.post("/students/")
def create_student(new_student: Student, current_user: User = Depends(get_current_active_user)):
    students.append(new_student)
    return new_student

@app.put("/students/{id}")
def update_student(id: int, updated: Student, current_user: User = Depends(get_current_active_user)):
    for index, student in enumerate(students):
        if student.id == id:
            students[index] = updated
            return updated
    return {"Error":"Error 404: Student not found"}

@app.delete("/students/{id}")
def delete_student(id: int, current_user: User = Depends(get_current_active_user)):
    for index, student in enumerate(students):
        if student.id == id:
            del (students[index])
            return {"msg":"student deleted"}
    return {"Error":"Error 404: Student not found"}



###################################### Code without JWT #######################################


# from fastapi import FastAPI, Query, HTTPException
# from fastapi.responses import JSONResponse
# from typing import Dict, Any
# from pycaret.classification import *
# from pydantic import BaseModel
# import pandas as pd
# import json


# app = FastAPI()
# model = load_model("student_performance_model")

# class Student(BaseModel):
#     id: int
#     name: str
#     grade: int


# students = [
#     Student(id = 1, name= "ali", grade= 150), 
#     Student(id = 2, name= "nesma", grade= 120),
#             ]

# with open('feature_ranges.json') as f:
#     FEATURE_RANGES = json.load(f)

# # Replace your current GRADE_MAP with this complete version
# GRADE_MAP = {
#     0: "fail (0-59%)",
#     1: "pass (60-69%)",
#     2: "good (70-79%)",
#     3: "very good (80-89%)",
#     4: "excellent (90-100%)",
#     # Add fallbacks for any unexpected values
#     -1: "unknown (error)",
#     5: "exceptional (100+%)"
# }


# def prepare_test_input(input_dict: dict) -> pd.DataFrame:
#     """
#     Casts specific fields to correct types and returns a single-row DataFrame.

#     Expected input format:
#         - Age: int
#         - Gender, Ethnicity, ParentalEducation, Tutoring, ParentalSupport,
#           Extracurricular, Sports, Music, Volunteering: str
#         - StudyTimeWeekly: float
#         - Absences: int
#     """
#     try:
#         processed = {
#             'Age': int(input_dict['Age']),
#             'Gender': int(input_dict['Gender']),
#             'Ethnicity': str(input_dict['Ethnicity']),
#             'ParentalEducation': str(input_dict['ParentalEducation']),
#             'StudyTimeWeekly': float(input_dict['StudyTimeWeekly']),
#             'Absences': int(input_dict['Absences']),
#             'Tutoring': str(input_dict['Tutoring']),
#             'ParentalSupport': str(input_dict['ParentalSupport']),
#             'Extracurricular': str(input_dict['Extracurricular']),
#             'Sports': str(input_dict['Sports']),
#             'Music': str(input_dict['Music']),
#             'Volunteering': str(input_dict['Volunteering']),
#         }
#         return pd.DataFrame([processed])
    
#     except (KeyError, ValueError, TypeError) as e:
#         raise ValueError(f"Invalid input or type conversion failed: {e}")





# @app.get("/predict", response_model=Dict[str, Any])
# async def predict_grade(
#     Age: int = Query(..., ge=15, le=18),
#     Gender: int = Query(..., ge=0, le=1),
#     Ethnicity: str = Query(..., pattern="^[0-3]$"),
#     ParentalEducation: str = Query(..., pattern="^[0-4]$"),
#     StudyTimeWeekly: float = Query(..., ge=0.0, le=20.0),
#     Absences: int = Query(..., ge=0, le=30),
#     Tutoring: str = Query(..., pattern="^[0-1]$"),
#     ParentalSupport: str = Query(..., pattern="^[0-4]$"),
#     Extracurricular: str = Query(..., pattern="^[0-1]$"),
#     Sports: str = Query(..., pattern="^[0-1]$"),
#     Music: str = Query(..., pattern="^[0-1]$"),
#     Volunteering: str = Query(..., pattern="^[0-1]$")
# ) -> JSONResponse:
#     try:
#         # Construct input dictionary
#         input_data = {
#             'Age': Age,
#             'Gender': Gender,
#             'Ethnicity': Ethnicity,
#             'ParentalEducation': ParentalEducation,
#             'StudyTimeWeekly': StudyTimeWeekly,
#             'Absences': Absences,
#             'Tutoring': Tutoring,
#             'ParentalSupport': ParentalSupport,
#             'Extracurricular': Extracurricular,
#             'Sports': Sports,
#             'Music': Music,
#             'Volunteering': Volunteering
#         }

#         # Convert to DataFrame
#         # test_input = pd.DataFrame([input_data])
#         test_data = prepare_test_input(input_data)

#         # # Optional: apply any mapping
#         # for col, mapping in FEATURE_RANGES.items():
#         #     if isinstance(mapping, dict) and col in test_data:
#         #         test_data[col] = test_data[col].map(mapping)

#         # Predict
#         predictions = predict_model(model, data=test_data)
#         # raw_label = predictions["prediction_label"].iloc[0]
#         print(predictions)
#         # Ensure label is an integer to map from GRADE_MAP
#         # try:
#         #     predicted_grade = GRADE_MAP.get(int(raw_label), GRADE_MAP[-1])
#         # except ValueError:
#         #     predicted_grade = GRADE_MAP[-1]
#         predicted_grade = predictions["prediction_label"].iloc[0] 
#         response_data = {
#             "expected_grade": predicted_grade,
#             "confidence_score": float(predictions["prediction_score"].iloc[0]),
#             "input_values": input_data
#         }

#         return JSONResponse(
#             content=response_data,
#             media_type="application/json",
#             status_code=200
#         )

#     except Exception as e:
#         return JSONResponse(
#             content={
#                 "error": "Prediction failed",
#                 "detail": str(e),
#                 "suggestion": "Please check your input values against the expected ranges"
#             },
#             media_type="application/json",
#             status_code=400
#         )
    

# @app.get("/students/")
# def read_students():
#     return students


# @app.post("/students/")
# def create_student(new_student: Student):
#     students.append(new_student)
#     return new_student

# @app.put("/students/{id}")
# def update_student(id: int, updated: Student):
#     for index, student in enumerate(students):
#         if student.id == id:
#             students[index] = updated
#             return updated
#     return {"Error":"Error 404: Student not found"}

# @app.delete("/students/{id}")
# def delete_student(id: int):
#     for index, student in enumerate(students):
#         if student.id == id:
#             del (students[index])
#             return {"msg":"student deleted"}
#     return {"Error":"Error 404: Student not found"}




















