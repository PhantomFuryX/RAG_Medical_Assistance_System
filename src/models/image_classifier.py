import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os
import torch.nn as nn

class MedicalImageClassifier:
    def __init__(self, device=None, model_path=None):
        # Determine device: use GPU if available
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load a pre-trained model (ResNet18 as base)
        self.model = models.resnet18(pretrained=True)
        
        # Modify the final layer to match our number of medical classes
        # First, load the classes to determine the number of output classes
        classes_file = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
        with open(classes_file) as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        num_classes = len(self.classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Load fine-tuned weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded fine-tuned model from {model_path}")
        else:
            print("Using base model without medical fine-tuning.")
            print("For better results, fine-tune the model on medical images.")
        
        self.model.eval()
        self.model.to(self.device)
        
        # Define transformations to match model input requirements
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def classify_image(self, image_path: str, top_k: int = 3) -> list:
        """
        Classify the image and return the top k predicted classes with probabilities.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with class names and probabilities
        """
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to list of dictionaries
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            predictions.append({
                "rank": i + 1,
                "class": self.classes[idx],
                "probability": float(prob)
            })
        
        return predictions
    
    def get_medical_advice(self, predictions: list) -> str:
        """
        Generate basic medical advice based on the top prediction.
        This is a placeholder - in a real system, this would be more sophisticated.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Basic medical advice string
        """
        top_condition = predictions[0]["class"]
        probability = predictions[0]["probability"]
        
        # Very basic advice - in a real system, this would come from a medical knowledge base
        advice = f"The image appears to show signs of {top_condition} (confidence: {probability:.2f}).\n\n"
        
        # Add disclaimer
        advice += "IMPORTANT DISCLAIMER: This is an automated analysis and should not be considered a diagnosis. "
        advice += "Please consult with a healthcare professional for proper medical advice and treatment."
        
        return advice

# For quick testing
if __name__ == "__main__":
    classifier = MedicalImageClassifier()
    # Replace 'sample_medical_image.jpg' with an actual image file path
    try:
        result = classifier.classify_image("sample_medical_image.jpg")
        print("Top Predictions:")
        for pred in result:
            print(f"{pred['rank']}. {pred['class']} - {pred['probability']:.4f}")
        
        advice = classifier.get_medical_advice(result)
        print("\nMedical Advice:")
        print(advice)
    except Exception as e:
        print(f"Error during classification: {e}")
