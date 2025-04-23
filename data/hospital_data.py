import numpy as np
import random

def get_hospital_data():
    """
    Get hospital data for the application.
    In a real application, this would fetch data from a database or API.
    For demonstration purposes, we generate synthetic hospital data.
    
    Returns:
        list: List of hospital dictionaries containing hospital information
    """
    # Define hospital names (Indian hospitals)
    hospital_names = [
        "AIIMS Delhi",
        "Fortis Hospital",
        "Apollo Hospitals",
        "Medanta - The Medicity",
        "Max Super Speciality Hospital",
        "Manipal Hospital",
        "Kokilaben Dhirubhai Ambani Hospital",
        "Christian Medical College",
        "Lilavati Hospital",
        "Tata Memorial Hospital"
    ]
    
    # Define specialties
    specialties_pool = [
        "General Medicine", 
        "Cardiology", 
        "Neurology", 
        "Orthopedics", 
        "Pediatrics", 
        "Obstetrics & Gynecology",
        "Oncology", 
        "Pulmonology", 
        "Gastroenterology", 
        "Nephrology",
        "Urology", 
        "Endocrinology", 
        "Dermatology", 
        "Infectious Disease", 
        "Hematology",
        "Psychiatry", 
        "Ophthalmology", 
        "ENT", 
        "Rheumatology", 
        "Allergy and Immunology",
        "Internal Medicine", 
        "Emergency Medicine", 
        "Critical Care", 
        "Hepatology"
    ]
    
    # Generate hospitals with randomized properties
    hospitals = []
    
    for i, name in enumerate(hospital_names):
        # Generate random coordinates (for location-based services)
        # These are roughly centered around Delhi, India
        # with some variation to spread hospitals around
        lat_base = 28.6139  # Delhi latitude
        lon_base = 77.2090  # Delhi longitude
        
        latitude = lat_base + (random.random() - 0.5) * 0.1  # +/- 0.05 degrees
        longitude = lon_base + (random.random() - 0.5) * 0.1  # +/- 0.05 degrees
        
        # Calculate distance from center (in km)
        # This is a simplified flat-earth approximation
        dx = (longitude - lon_base) * 111.32 * np.cos(lat_base * np.pi/180)
        dy = (latitude - lat_base) * 110.574
        distance = np.sqrt(dx**2 + dy**2)
        
        # Generate random bed counts
        total_beds = random.randint(150, 800)  # Indian hospitals often have higher bed counts
        available_beds = random.randint(int(total_beds * 0.1), int(total_beds * 0.5))
        
        # Generate random ICU beds
        total_icu = random.randint(30, 100)
        available_icu = random.randint(int(total_icu * 0.1), int(total_icu * 0.5))
        
        # Generate random rating
        rating = round(random.uniform(3.5, 5.0), 1)
        
        # Generate random specialties (5-10 specialties per hospital)
        num_specialties = random.randint(5, 10)
        specialties = random.sample(specialties_pool, num_specialties)
        
        # Generate random staff counts
        doctors = random.randint(75, 250)
        nurses = random.randint(150, 500)
        staff_availability = random.randint(70, 98)
        
        # Create hospital address with Indian streets
        streets = ["Rajpath Road", "MG Road", "Nehru Place", "Connaught Place", "Chandni Chowk", 
                  "Saket District", "Karol Bagh", "Vasant Vihar", "Dwarka Sector", "Greater Kailash"]
        localities = ["New Delhi", "Gurgaon", "Noida", "Faridabad", "Ghaziabad", "South Delhi", "North Delhi"]
        address = f"{random.choice(streets)}, {random.choice(localities)}"
        
        # Create hospital dictionary
        hospital = {
            'id': f"H{i+1:03d}",
            'name': name,
            'latitude': latitude,
            'longitude': longitude,
            'distance': round(distance, 2),
            'address': address,
            'total_beds': total_beds,
            'available_beds': available_beds,
            'total_icu': total_icu,
            'available_icu': available_icu,
            'rating': rating,
            'specialties': specialties,
            'doctors': doctors,
            'nurses': nurses,
            'staff_availability': staff_availability
        }
        
        hospitals.append(hospital)
    
    return hospitals

def get_hospital_by_id(hospital_id):
    """
    Get hospital data for a specific hospital ID.
    
    Args:
        hospital_id (str): Hospital ID to retrieve
        
    Returns:
        dict: Hospital data dictionary or None if not found
    """
    hospitals = get_hospital_data()
    
    for hospital in hospitals:
        if hospital['id'] == hospital_id:
            return hospital
    
    return None

def get_hospitals_by_specialty(specialty):
    """
    Get list of hospitals that offer a specific specialty.
    
    Args:
        specialty (str): Medical specialty to filter by
        
    Returns:
        list: List of hospital dictionaries with the specified specialty
    """
    hospitals = get_hospital_data()
    
    # Filter hospitals by specialty
    matching_hospitals = [h for h in hospitals if specialty in h['specialties']]
    
    return matching_hospitals

def update_hospital_beds(hospital_id, beds_used=1, icu_used=0):
    """
    Update hospital bed counts after a booking.
    In a real application, this would update a database.
    
    Args:
        hospital_id (str): ID of the hospital to update
        beds_used (int): Number of regular beds being used
        icu_used (int): Number of ICU beds being used
        
    Returns:
        bool: True if update successful, False otherwise
    """
    # In a real application, this would modify a database
    # For this demo, we just return True to simulate success
    return True

def get_hospital_bed_history(hospital_id, days=30):
    """
    Get bed occupancy history for a specific hospital.
    For demo purposes, this generates synthetic history.
    
    Args:
        hospital_id (str): Hospital ID to get history for
        days (int): Number of days of history to return
        
    Returns:
        dict: Dictionary containing bed occupancy history
    """
    hospital = get_hospital_by_id(hospital_id)
    
    if not hospital:
        return {"error": "Hospital not found"}
    
    # Generate synthetic history data
    import datetime
    
    history = []
    total_beds = hospital['total_beds']
    
    # Set random seed based on hospital_id for consistent results
    random.seed(int(hospital_id[1:]))
    
    # Start with random occupancy between 40-70%
    occupancy = random.randint(40, 70)
    
    for i in range(days):
        # Calculate date
        date = (datetime.datetime.now() - datetime.timedelta(days=days-i-1)).strftime('%Y-%m-%d')
        
        # Adjust occupancy with some random variation
        # Keeping it between 30% and 90%
        change = random.randint(-5, 5)
        occupancy = max(30, min(90, occupancy + change))
        
        # Calculate occupied and available beds
        occupied_beds = int(total_beds * occupancy / 100)
        available_beds = total_beds - occupied_beds
        
        # Add entry to history
        history.append({
            'date': date,
            'occupancy_rate': occupancy,
            'occupied_beds': occupied_beds,
            'available_beds': available_beds
        })
    
    return {
        'hospital_id': hospital_id,
        'hospital_name': hospital['name'],
        'total_beds': total_beds,
        'history': history
    }
