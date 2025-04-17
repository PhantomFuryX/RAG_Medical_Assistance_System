from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class SymptomClassifier:
    def __init__(self, model_name="bvanaken/clinical-assertion-negation-bert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def classify(self, text: str) -> dict:
        """Classify if symptoms are present, absent, or uncertain."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predictions = predictions.cpu().numpy()
        
        # Classes: present, absent, possible, conditional, associated_with_someone_else, hypothetical
        labels = ["present", "absent", "possible", "conditional", "associated_with_someone_else", "hypothetical"]
        result = {label: float(pred) for label, pred in zip(labels, predictions[0])}
        
        return result
