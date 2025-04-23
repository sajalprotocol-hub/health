import pandas as pd
import numpy as np
import random
import uuid
import datetime
from data.hospital_data import get_hospital_data

def get_hospital_bed_status(hospitals=None):
    """
    Get the current status of hospital beds.
    
    Args:
        hospitals (list, optional): List of hospital dictionaries. If None, will load from data module.
        
    Returns:
        pandas.DataFrame: DataFrame containing bed status information
    """
    if hospitals is None:
        hospitals = get_hospital_data()
    
    # Create a DataFrame with bed status information
    bed_status = pd.DataFrame({
        'hospital_name': [h['name'] for h in hospitals],
        'available_beds': [h['available_beds'] for h in hospitals],
        'total_beds': [h['total_beds'] for h in hospitals]
    })
    
    # Calculate occupied beds
    bed_status['occupied_beds'] = bed_status['total_beds'] - bed_status['available_beds']
    
    # Calculate occupancy rate
    bed_status['occupancy_rate'] = (bed_status['occupied_beds'] / bed_status['total_beds'] * 100).round(1)
    
    return bed_status

def recommend_hospitals(hospitals, disease):
    """
    Recommend hospitals based on predicted disease and available beds.
    
    Args:
        hospitals (list): List of hospital dictionaries
        disease (str): Predicted disease
        
    Returns:
        list: List of recommended hospital dictionaries, sorted by relevance
    """
    # Filter hospitals with available beds
    available_hospitals = [h for h in hospitals if h['available_beds'] > 0]
    
    if not available_hospitals:
        return []
    
    # Disease to specialty mapping
    disease_specialty_map = {
        "Pneumonia": ["Pulmonology", "Infectious Disease"],
        "Diabetes": ["Endocrinology", "Internal Medicine"],
        "Heart Disease": ["Cardiology"],
        "Liver Disease": ["Gastroenterology", "Hepatology"],
        "Kidney Disease": ["Nephrology"],
        "Influenza": ["Infectious Disease", "Internal Medicine"],
        "Hypertension": ["Cardiology", "Internal Medicine"],
        "Asthma": ["Pulmonology", "Allergy and Immunology"],
        "COVID-19": ["Infectious Disease", "Pulmonology", "Critical Care"]
    }
    
    # Get relevant specialties for the disease
    relevant_specialties = disease_specialty_map.get(disease, ["General Medicine"])
    
    # Score hospitals based on specialty match, bed availability, and rating
    scored_hospitals = []
    for hospital in available_hospitals:
        specialties = hospital.get('specialties', [])
        
        # Calculate specialty match score
        specialty_score = sum(1 for specialty in relevant_specialties if specialty in specialties)
        specialty_score = specialty_score / len(relevant_specialties) if relevant_specialties else 0
        
        # Calculate bed availability score (0-1)
        bed_score = min(1.0, hospital['available_beds'] / 20)  # Cap at 20 beds for scoring
        
        # Get rating score (0-1)
        rating_score = hospital['rating'] / 5
        
        # Calculate distance factor (closer is better)
        distance = hospital.get('distance', 20)  # Default to 20 km if not provided
        distance_score = max(0, 1 - (distance / 50))  # 0 km -> 1, 50+ km -> 0
        
        # Calculate total score with weights
        total_score = (
            specialty_score * 0.4 +  # Specialty match is important
            bed_score * 0.3 +        # Bed availability is important
            rating_score * 0.2 +     # Rating matters somewhat
            distance_score * 0.1     # Distance is least important factor
        )
        
        # Add to scored hospitals
        scored_hospitals.append({
            **hospital,
            'score': total_score
        })
    
    # Sort hospitals by score (descending)
    recommended_hospitals = sorted(scored_hospitals, key=lambda h: h['score'], reverse=True)
    
    # Return top 5 hospitals
    return recommended_hospitals[:5]

def book_hospital_bed(booking_data, hospital_data):
    """
    Book a hospital bed and update hospital availability.
    
    Args:
        booking_data (dict): Dictionary containing booking information
        hospital_data (dict): Dictionary containing hospital information
        
    Returns:
        dict: Dictionary with booking result information
    """
    # Check if there are available beds
    if hospital_data['available_beds'] <= 0:
        return {
            'success': False,
            'error': 'No beds available at this hospital'
        }
    
    # Check if the requested bed type is available
    bed_type = booking_data.get('bed_type', 'General Ward')
    if bed_type == 'ICU' and hospital_data.get('available_icu', 0) <= 0:
        return {
            'success': False,
            'error': 'No ICU beds available at this hospital'
        }
    
    try:
        # Generate a booking ID
        booking_id = str(uuid.uuid4())[:8].upper()
        
        # In a real application, you would update the database here
        # For simulation purposes, we'll just return success
        
        # In a real application, this would update hospital bed counts in the database
        
        # Return success with booking information
        return {
            'success': True,
            'booking_id': booking_id,
            'hospital': hospital_data['name'],
            'bed_type': bed_type,
            'booking_date': booking_data.get('booking_date', datetime.datetime.now().date())
        }
    
    except Exception as e:
        # Handle any exceptions during booking
        return {
            'success': False,
            'error': f'Booking failed: {str(e)}'
        }

def get_doctors_by_specialty(specialty):
    """
    Get a list of doctors filtered by specialty.
    
    Args:
        specialty (str): Medical specialty to filter by
        
    Returns:
        list: List of doctor dictionaries matching the specialty
    """
    # Get hospital data
    hospitals = get_hospital_data()
    
    # Create a list to store doctors
    all_doctors = []
    
    # Names for generating Indian doctor data
    first_names = ["Rajesh", "Amit", "Sunil", "Vikram", "Rahul", "Sanjay", "Anil", "Mukesh", "Deepak", "Priya", 
                   "Neha", "Sunita", "Anita", "Kavita", "Pooja", "Anjali", "Meera", "Shikha", "Divya", "Ritu"]
    
    last_names = ["Sharma", "Patel", "Singh", "Kumar", "Verma", "Agarwal", "Gupta", "Chopra", "Reddy", "Joshi", 
                  "Malhotra", "Mehta", "Bhatia", "Desai", "Nair", "Kapoor", "Chauhan", "Saxena", "Rao", "Khanna"]
    
    # Languages for doctors in India
    languages = ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"]
    
    # Generate doctors for each hospital
    for hospital in hospitals:
        # Determine how many doctors to generate for this hospital
        num_doctors = random.randint(3, 8)
        
        for _ in range(num_doctors):
            # Randomly generate doctor details
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            name = f"Dr. {first_name} {last_name}"
            
            # Experience between 1 and 35 years
            experience = random.randint(1, 35)
            
            # Rating between 3.0 and 5.0
            rating = round(random.uniform(3.0, 5.0), 1)
            
            # Gender
            gender = random.choice(["Male", "Female"])
            
            # Available today (70% chance)
            available_today = random.random() < 0.7
            
            # Languages (always English plus 0-2 random others)
            doctor_languages = ["English"]
            for _ in range(random.randint(0, 2)):
                lang = random.choice([l for l in languages if l != "English"])
                if lang not in doctor_languages:
                    doctor_languages.append(lang)
            
            # Add profile image URL - using consistent placeholder images
            if gender == "Male":
                profile_image = f"https://randomuser.me/api/portraits/men/{random.randint(1, 99)}.jpg"
            else:
                profile_image = f"https://randomuser.me/api/portraits/women/{random.randint(1, 99)}.jpg"
            
            # Add contact and education information with Indian medical institutions
            education = random.choice(["MBBS, AIIMS Delhi", 
                                      "MD, Christian Medical College Vellore", 
                                      "MBBS, KEM Hospital Mumbai", 
                                      "MD, Maulana Azad Medical College",
                                      "MS, PGIMER Chandigarh",
                                      "DNB, Manipal Academy of Higher Education",
                                      "MBBS, Seth GS Medical College",
                                      "MD, JIPMER Puducherry"])
            
            # Indian mobile numbers start with +91 and have 10 digits
            phone = f"+91 {random.randint(7, 9)}{random.randint(1000, 9999)}{random.randint(10000, 99999)}"[0:14]
            # Indian domains for healthcare email addresses
            domains = ["healthcare.in", "hospital.co.in", "medcenter.in", "health.in", "care.co.in"]
            email = f"{first_name.lower()}.{last_name.lower()}@{hospital['name'].lower().replace(' ', '').replace('-', '')}.{random.choice(domains)}"
            
            # Create doctor dictionary with enhanced profile data
            doctor = {
                'id': f"D{random.randint(1000, 9999)}",
                'name': name,
                'specialty': specialty,
                'experience': experience,
                'rating': rating,
                'gender': gender,
                'hospital': hospital['name'],
                'hospital_id': hospital['id'],
                'available_today': available_today,
                'languages': doctor_languages,
                'profile_image': profile_image,
                'education': education,
                'phone': phone,
                'email': email,
                'accepting_new_patients': random.choice([True, False, True, True])  # 75% chance of accepting new patients
            }
            
            all_doctors.append(doctor)
    
    # Shuffle the list to randomize order
    random.shuffle(all_doctors)
    
    return all_doctors

def get_hospital_ward_status(hospital_id=None):
    """
    Get detailed ward status for a specific hospital or all hospitals.
    
    Args:
        hospital_id (str, optional): Hospital ID to filter by. If None, returns data for all hospitals.
        
    Returns:
        dict: Dictionary containing ward status information
    """
    hospitals = get_hospital_data()
    
    # Filter by hospital ID if provided
    if hospital_id:
        hospitals = [h for h in hospitals if h.get('id') == hospital_id]
        if not hospitals:
            return {"error": "Hospital not found"}
    
    # Ward types
    ward_types = ["General", "ICU", "Pediatric", "Emergency", "Surgical", "Maternity"]
    
    # Generate ward status for each hospital
    ward_status = []
    
    for hospital in hospitals:
        hospital_wards = []
        
        for ward_type in ward_types:
            # Skip some wards randomly to simulate different hospitals having different wards
            if random.random() < 0.2 and ward_type not in ["General", "Emergency"]:
                continue
                
            # Generate ward data
            ward = {
                "ward_name": f"{ward_type} Ward",
                "total_beds": random.randint(10, 50),
                "staff_count": random.randint(5, 20),
                "equipment_status": random.choice(["Optimal", "Good", "Needs Maintenance"])
            }
            
            # Calculate occupied and available beds
            occupied = random.randint(0, ward["total_beds"])
            ward["occupied_beds"] = occupied
            ward["available_beds"] = ward["total_beds"] - occupied
            
            # Calculate occupancy rate
            ward["occupancy_rate"] = round((occupied / ward["total_beds"]) * 100, 1)
            
            hospital_wards.append(ward)
        
        ward_status.append({
            "hospital_name": hospital["name"],
            "hospital_id": hospital.get("id", ""),
            "wards": hospital_wards
        })
    
    return ward_status

def get_hospital_staff_schedule(hospital_id=None, date=None):
    """
    Get staff schedule for a specific hospital on a specific date.
    
    Args:
        hospital_id (str, optional): Hospital ID to filter by. If None, returns data for first hospital.
        date (datetime.date, optional): Date to get schedule for. If None, uses current date.
        
    Returns:
        dict: Dictionary containing staff schedule information
    """
    # Get hospitals
    hospitals = get_hospital_data()
    
    # Set default date to today if not provided
    if date is None:
        date = datetime.datetime.now().date()
    
    # Filter by hospital ID if provided
    if hospital_id:
        hospital = next((h for h in hospitals if h.get('id') == hospital_id), None)
        if not hospital:
            return {"error": "Hospital not found"}
    else:
        # Use first hospital if ID not provided
        hospital = hospitals[0]
    
    # Staff roles
    staff_roles = {
        "Doctor": ["Morning", "Afternoon", "Night"],
        "Nurse": ["Morning", "Afternoon", "Night"],
        "Technician": ["Morning", "Afternoon"],
        "Administrative": ["Morning", "Afternoon"]
    }
    
    # Indian Staff names
    first_names = ["Rajesh", "Amit", "Sunil", "Vikram", "Rahul", "Sanjay", "Anil", "Mukesh", "Deepak", "Priya", 
                   "Neha", "Sunita", "Anita", "Kavita", "Pooja", "Anjali", "Meera", "Shikha", "Divya", "Ritu"]
    
    last_names = ["Sharma", "Patel", "Singh", "Kumar", "Verma", "Agarwal", "Gupta", "Chopra", "Reddy", "Joshi", 
                  "Malhotra", "Mehta", "Bhatia", "Desai", "Nair", "Kapoor", "Chauhan", "Saxena", "Rao", "Khanna"]
    
    # Generate staff schedule
    schedule = []
    
    for role, shifts in staff_roles.items():
        # Determine number of staff for this role
        num_staff = random.randint(3, 10) if role == "Doctor" else random.randint(5, 15)
        
        for i in range(num_staff):
            # Generate staff name
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            # Assign a shift
            shift = random.choice(shifts)
            
            # Determine start and end times based on shift
            if shift == "Morning":
                start_time = "07:00"
                end_time = "15:00"
            elif shift == "Afternoon":
                start_time = "15:00"
                end_time = "23:00"
            else:  # Night
                start_time = "23:00"
                end_time = "07:00"
            
            # Generate department
            department = random.choice(["General", "Emergency", "Surgery", "Pediatrics", "Obstetrics", "Radiology"])
            
            # Add to schedule
            schedule.append({
                "name": name,
                "role": role,
                "shift": shift,
                "start_time": start_time,
                "end_time": end_time,
                "department": department
            })
    
    # Sort schedule by role and shift
    schedule = sorted(schedule, key=lambda x: (x["role"], x["shift"]))
    
    return {
        "hospital_name": hospital["name"],
        "date": date.strftime("%Y-%m-%d"),
        "schedule": schedule
    }
