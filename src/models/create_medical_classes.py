import os

def create_medical_classes_file():
    # Path to save the file
    output_file = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
    
    # List of medical conditions that might be visually identifiable in images
    medical_conditions = [
        # Skin conditions
        "acne",
        "eczema",
        "psoriasis",
        "rosacea",
        "dermatitis",
        "melanoma",
        "basal_cell_carcinoma",
        "squamous_cell_carcinoma",
        "hives",
        "shingles",
        "ringworm",
        "impetigo",
        "cellulitis",
        "warts",
        "vitiligo",
        "scabies",
        "measles_rash",
        "chickenpox",
        
        # Eye conditions
        "conjunctivitis",
        "cataract",
        "glaucoma",
        "stye",
        "subconjunctival_hemorrhage",
        "macular_degeneration",
        "diabetic_retinopathy",
        "corneal_ulcer",
        
        # Oral conditions
        "oral_thrush",
        "gingivitis",
        "periodontitis",
        "oral_herpes",
        "oral_cancer",
        "tonsillitis",
        "dental_abscess",
        "angular_cheilitis",
        
        # Wounds and injuries
        "laceration",
        "abrasion",
        "burn_first_degree",
        "burn_second_degree",
        "burn_third_degree",
        "bruise",
        "fracture",
        "sprain",
        "dislocation",
        
        # Infectious diseases with visible symptoms
        "influenza",
        "common_cold",
        "covid19_pneumonia",
        "tuberculosis",
        "pneumonia",
        "bronchitis",
        
        # Medical imaging findings
        "lung_nodule",
        "pulmonary_edema",
        "pleural_effusion",
        "pneumothorax",
        "bone_fracture",
        "osteoarthritis",
        "rheumatoid_arthritis",
        "brain_tumor",
        "brain_hemorrhage",
        "stroke",
        "coronary_artery_disease",
        "appendicitis",
        "kidney_stone",
        "gallstone",
        
        # Other visible conditions
        "jaundice",
        "anemia",
        "edema",
        "cyanosis",
        "clubbing",
        "lymphadenopathy",
        "goiter",
        "hair_loss",
        "nail_fungus",
        
        # Common medical equipment/scenarios (for context)
        "x_ray",
        "mri_scan",
        "ct_scan",
        "ultrasound",
        "ecg",
        "blood_test",
        "stethoscope",
        "surgical_procedure",
        "hospital_room",
        "medical_prescription",
        
        # Normal/healthy state
        "normal_skin",
        "normal_eye",
        "normal_oral_cavity",
        "normal_lung",
        "normal_heart",
        "normal_brain",
        "normal_bone",
        "normal_joint"
    ]
    
    # Write the medical conditions to the file
    with open(output_file, "w") as f:
        f.write("\n".join(medical_conditions))
    
    print(f"Successfully created {output_file} with {len(medical_conditions)} medical conditions")
    print("Note: This is a custom list for medical image classification.")
    print("For production use, consider consulting with medical professionals to refine this list.")

if __name__ == "__main__":
    create_medical_classes_file()
