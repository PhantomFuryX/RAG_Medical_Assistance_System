# from src.models.image_classifier import ImageClassifier
from src.retrieval.document_retriever import MedicalDocumentRetriever
from src.nlp.diagnosis_chain import build_diagnosis_chain
from typing import List, Optional
from langchain.chains import LLMChain
from src.retrieval.document_retriever import MedicalDocumentRetriever
from src.nlp.medical_entity_extractor import MedicalEntityExtractor
from src.models.symptom_classifier import SymptomClassifier
from src.main.core.llm_engine import generate_response
from src.utils.registry import registry
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger("pipeline")

class DiagnosisPipeline:
    """
    Pipeline for processing medical symptoms and images to provide a diagnosis.
    Combines document retrieval with LLM-based diagnosis.
    """
    
    def __init__(self):
        """
        Initialize the diagnosis pipeline.
        
        Args:
            retriever: Component for retrieving relevant medical documents
            diagnosis_chain: LLM chain for generating diagnoses
            image_classifier: Optional component for classifying medical images
        """
        if registry.get("retriever") is None:
            self.retriever = MedicalDocumentRetriever(lazy_loading=True)
        else:
            # Use the existing retriever from the registry
            logger.info("Using existing retriever from registry")
            self.retriever = registry.get("retriever")
        # self.retriever = MedicalDocumentRetriever()
        self.entity_extractor = MedicalEntityExtractor()
        self.symptom_classifier = SymptomClassifier()
        self.diagnosis_chain = build_diagnosis_chain(settings.MODEL_API)
        self.image_classifier = None  # Placeholder for image classifier
    
    def process_input(self, symptoms: str, image_path: str):
        """
        Process user symptoms and optional image to generate a diagnosis.
        
        Args:
            symptoms: String describing patient symptoms
            image_path: Optional path to a medical image for analysis
            
        Returns:
            String containing the diagnosis based on symptoms and retrieved information
        """
        image_result = None
        if image_path and self.image_classifier:
            image_result = self.image_classifier.classify_image(image_path)
            print(f"Image Classification Result: {image_result}")
        
        retrieved_docs = self.retriever.retrieve(symptoms)
        print("Relevant Documents Retrieved:")
        for doc in retrieved_docs:
            print(doc.page_content[:200], "...")  # Print summary
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Run the diagnosis chain with symptoms and retrieved context
        response = self.diagnosis_chain.run(
            symptoms=symptoms, 
            context=context,
            image_result=image_result if image_result else "No image provided"
        )
        
        return response
    
    def get_retrieved_documents(self, symptoms: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant documents for the given symptoms.
        
        Args:
            symptoms: String describing patient symptoms
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved document contents
        """
        docs = self.retriever.retrieve(symptoms, k=k)
        return [doc.page_content for doc in docs]
    
    def process(self, user_input: str) -> dict:
        """Process user input through the diagnosis pipeline."""
        # Extract medical entities
        entities = self.entity_extractor.extract_entities(user_input)
        
        # Classify symptoms
        symptom_classification = self.symptom_classifier.classify(user_input)
        
        # Retrieve relevant medical information
        retrieved_docs = []
        if self.retriever.index is not None:
            retrieved_docs = self.retriever.retrieve(user_input, k=3)
        
        # Format context for LLM
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate diagnosis with LLM
        prompt = f"""
        You are a medical assistant AI. Based on the following information, provide a possible diagnosis:
        
        USER SYMPTOMS: {user_input}
        
        EXTRACTED MEDICAL ENTITIES: {entities}
        
        SYMPTOM CLASSIFICATION: {symptom_classification}
        
        RELEVANT MEDICAL INFORMATION:
        {context}
        
        Provide a thoughtful analysis of the possible conditions, their likelihood, and what further information would be helpful. Always recommend consulting a healthcare professional.
        """
        
        diagnosis = generate_response(user_question=prompt)
        
        return {
            "user_input": user_input,
            "entities": entities,
            "symptom_classification": symptom_classification,
            "retrieved_documents": [doc.page_content for doc in retrieved_docs],
            "diagnosis": diagnosis
        }
        
    def process_image(self, image_path: str) -> dict:
        """Process an image through the diagnosis pipeline."""
        # Classify the image
        image_result = self.image_classifier.classify_image(image_path)

        # Retrieve relevant medical information
        retrieved_docs = self.retriever.retrieve(image_result, k=3)

        # Format context for LLM
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate diagnosis with LLM
        prompt = f"""
        You are a medical assistant AI. Based on the following information, provide a possible diagnosis:

        IMAGE CLASSIFICATION RESULT: {image_result}

        RELEVANT MEDICAL INFORMATION:
        {context}

        Provide a thoughtful analysis of the possible conditions, their likelihood, and what further information would be helpful. Always recommend consulting a healthcare professional.
        """

        diagnosis = generate_response(user_question=prompt)

        return {
            "image_classification": image_result,
            "retrieved_documents": [doc.page_content for doc in retrieved_docs],
            "diagnosis": diagnosis
        }
        
# Example Usage
if __name__ == "__main__":
    assistant = DiagnosisPipeline()
    user_symptoms = "I have a fever, sore throat, and body ache."
    response = assistant.process_input(user_symptoms)
    print("Diagnosis:", response)
