from src.models.image_classifier import ImageClassifier
from src.nlp.diagnosis_chain import build_diagnosis_chain
# from src.utils.config import OPENAI_API_KEY
from src.utils.settings import settings

if (settings.MODEL_API_KEY == "openai" ):
    API_KEY = settings.OPENAI_API_KEY
OPENAI_API_KEY = settings.OPENAI_API_KEY
def main():
    # Step 1: Image classification
    classifier = ImageClassifier()
    # Replace with a real image file path. Ensure the file exists.
    image_path = "sample_image.jpg"
    try:
        image_result = classifier.classify_image(image_path)
        print("Image Classification Result:", image_result)
    except Exception as e:
        print("Error during image classification:", e)
        image_result = "unknown"

    # Step 2: Get user-reported symptoms (hardcoded for demonstration)
    symptoms = "Patient reports severe headache, fever, and joint pain."

    # Step 3: Build and run the diagnosis chain using LangChain
    diagnosis_chain = build_diagnosis_chain()
    diagnosis = diagnosis_chain.run({"symptoms": symptoms, "image_result": image_result})
    print("Final Diagnosis:", diagnosis)

if __name__ == "__main__":
    main()