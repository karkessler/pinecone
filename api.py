from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
import joblib
import pandas as pd
from datetime import datetime
import os
from typing import List
from pinecone import Pinecone, ServerlessSpec
from fastapi.middleware.cors import CORSMiddleware


# Load environment variables
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI(
    title="Student Performance Prediction API",
    description="An API to predict student performance based on various factors",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://raspidev", "http://localhost"],  # Add specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Load the trained model and scaler
model = tf.keras.models.load_model("student_performance_model.keras")
scaler = joblib.load("scaler.pkl")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "student-performance"

# Check if the index exists; create it if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=scaler.feature_names_in_.shape[0] + 1,  # Match dimensions
        metric="euclidean",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"  # Replace with your preferred region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Define input schema for FastAPI
class StudentInput(BaseModel):
    gender: str = Field(..., description="Student's gender")
    race_ethnicity: str = Field(..., description="Student's race/ethnicity")
    parental_level_of_education: str = Field(..., description="Parent's education level")
    lunch: str = Field(..., description="Type of lunch (standard or free/reduced)")
    test_preparation_course: str = Field(..., description="Test preparation course (completed or none)")

    class Config:
        json_schema_extra  = {
            "example": {
                "gender": "female",
                "race_ethnicity": "group A",
                "parental_level_of_education": "bachelor's degree",
                "lunch": "standard",
                "test_preparation_course": "completed"
            }
        }


# Define output schema for FastAPI
class PredictionResponse(BaseModel):
    predicted_score: float
    timestamp: str


@app.post("/predict", response_model=PredictionResponse)
def predict_student_score(student: StudentInput):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([student.dict()])

        # One-hot encode categorical variables
        categorical_columns = [
            'gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course'
        ]
        input_encoded = pd.get_dummies(input_data, columns=categorical_columns)

        # Ensure all columns from training data are present
        for col in scaler.feature_names_in_:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder columns to match training data
        input_encoded = input_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # Scale the input data
        input_scaled = scaler.transform(input_encoded)

        # Predict score
        prediction = model.predict(input_scaled).flatten()[0]

        # Store prediction in Pinecone
        metadata = student.dict()
        metadata["predicted_math_score"] = float(prediction)
        vector = {
            "id": "student_" + str(datetime.now().timestamp()),
            "values": [float(val) for val in input_scaled.flatten()] + [float(prediction)], # Explicit float64 conversion
            "metadata": metadata
        }
        index.upsert([vector])

        # Return prediction
        return PredictionResponse(
            predicted_score=float(prediction),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
