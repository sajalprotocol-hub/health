import os
import pandas as pd
import datetime
import sqlalchemy
from sqlalchemy import create_engine, text, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Create a database connection
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///healthcare.db")
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Define database models
class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(20), unique=True, index=True)
    name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(10))
    blood_type = Column(String(5))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    medical_records = relationship("MedicalRecord", back_populates="patient")
    predictions = relationship("Prediction", back_populates="patient")

class MedicalRecord(Base):
    __tablename__ = 'medical_records'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(20), ForeignKey('patients.patient_id'))
    date = Column(DateTime, default=datetime.datetime.utcnow)
    symptoms = Column(Text)
    vital_signs = Column(JSON)
    test_results = Column(JSON)
    diagnosis = Column(String(100))
    treatment = Column(Text)
    doctor = Column(String(100))
    
    # Relationship
    patient = relationship("Patient", back_populates="medical_records")

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(20), ForeignKey('patients.patient_id'))
    prediction_date = Column(DateTime, default=datetime.datetime.utcnow)
    disease = Column(String(100))
    probability = Column(Float)
    risk_level = Column(String(20))
    model_used = Column(String(50))
    input_data = Column(JSON)
    
    # Relationship
    patient = relationship("Patient", back_populates="predictions")

class Hospital(Base):
    __tablename__ = 'hospitals'
    
    id = Column(Integer, primary_key=True)
    hospital_id = Column(String(20), unique=True, index=True)
    name = Column(String(100))
    address = Column(String(200))
    latitude = Column(Float)
    longitude = Column(Float)
    total_beds = Column(Integer)
    available_beds = Column(Integer)
    total_icu = Column(Integer)
    available_icu = Column(Integer)
    rating = Column(Float)
    specialties = Column(JSON)
    
    # Relationship
    doctors = relationship("Doctor", back_populates="hospital")

class Doctor(Base):
    __tablename__ = 'doctors'
    
    id = Column(Integer, primary_key=True)
    doctor_id = Column(String(20), unique=True, index=True)
    name = Column(String(100))
    specialty = Column(String(50))
    hospital_id = Column(String(20), ForeignKey('hospitals.hospital_id'))
    experience = Column(Integer)
    rating = Column(Float)
    gender = Column(String(10))
    available_today = Column(Boolean)
    languages = Column(JSON)
    profile_image = Column(String(200))
    education = Column(String(100))
    phone = Column(String(20))
    email = Column(String(100))
    
    # Relationship
    hospital = relationship("Hospital", back_populates="doctors")

# Initialize database
def init_db():
    """Create tables in the database"""
    Base.metadata.create_all(engine)

# Database operations
def add_patient(patient_data):
    """Add a new patient to the database"""
    with Session() as session:
        patient = Patient(**patient_data)
        session.add(patient)
        session.commit()
        return patient

def get_patient(patient_id):
    """Get a patient by ID"""
    with Session() as session:
        return session.query(Patient).filter_by(patient_id=patient_id).first()

def add_prediction(prediction_data):
    """Add a prediction result to the database"""
    with Session() as session:
        prediction = Prediction(**prediction_data)
        session.add(prediction)
        session.commit()
        return prediction

def get_patient_predictions(patient_id):
    """Get all predictions for a patient"""
    with Session() as session:
        return session.query(Prediction).filter_by(patient_id=patient_id).all()

def add_medical_record(record_data):
    """Add a medical record to the database"""
    with Session() as session:
        record = MedicalRecord(**record_data)
        session.add(record)
        session.commit()
        return record

def add_hospital(hospital_data):
    """Add a hospital to the database"""
    with Session() as session:
        hospital = Hospital(**hospital_data)
        session.add(hospital)
        session.commit()
        return hospital

def add_doctor(doctor_data):
    """Add a doctor to the database"""
    with Session() as session:
        doctor = Doctor(**doctor_data)
        session.add(doctor)
        session.commit()
        return doctor

def get_hospitals():
    """Get all hospitals from the database"""
    with Session() as session:
        return session.query(Hospital).all()
        
def get_doctors_by_specialty(specialty):
    """Get doctors by specialty"""
    with Session() as session:
        return session.query(Doctor).filter_by(specialty=specialty).all()

def get_hospital_bed_status():
    """Get bed status for all hospitals"""
    with Session() as session:
        hospitals = session.query(Hospital).all()
        return [{
            'hospital_name': h.name,
            'available_beds': h.available_beds,
            'total_beds': h.total_beds,
            'occupied_beds': h.total_beds - h.available_beds,
            'available_icu': h.available_icu,
            'total_icu': h.total_icu,
            'occupied_icu': h.total_icu - h.available_icu
        } for h in hospitals]

def update_hospital_beds(hospital_id, beds_used=1, icu_used=0):
    """Update hospital bed counts after a booking"""
    with Session() as session:
        hospital = session.query(Hospital).filter_by(hospital_id=hospital_id).first()
        if hospital:
            hospital.available_beds = max(0, hospital.available_beds - beds_used)
            hospital.available_icu = max(0, hospital.available_icu - icu_used)
            session.commit()
            return True
        return False

# Import functions for dataset management
def load_dataset_from_csv(file_path):
    """Load a dataset from a CSV file into the database"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def import_hospital_data(file_path):
    """Import hospital data from CSV to database"""
    df = load_dataset_from_csv(file_path)
    if df is not None:
        hospitals = []
        for _, row in df.iterrows():
            hospital_data = row.to_dict()
            # Convert specialties to list if it's a string
            if 'specialties' in hospital_data and isinstance(hospital_data['specialties'], str):
                hospital_data['specialties'] = hospital_data['specialties'].split(',')
            hospitals.append(add_hospital(hospital_data))
        return len(hospitals)
    return 0

def import_patient_data(file_path):
    """Import patient data from CSV to database"""
    df = load_dataset_from_csv(file_path)
    if df is not None:
        patients = []
        for _, row in df.iterrows():
            patient_data = row.to_dict()
            patients.append(add_patient(patient_data))
        return len(patients)
    return 0
