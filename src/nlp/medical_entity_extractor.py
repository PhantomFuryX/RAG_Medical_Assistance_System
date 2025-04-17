import spacy
from typing import List, Dict

class MedicalEntityExtractor:
    def __init__(self, model="en_core_web_md"):  # Scientific/medical NER model
        self.nlp = spacy.load(model)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract medical entities from text."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
