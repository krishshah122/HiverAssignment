from sentence_transformers import SentenceTransformer, util
import torch

class IntentClassifier:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the zero-shot embedding-based intent classifier.
        Downloads the model on first run (very lightweight, ~80MB).
        """
        self.model = SentenceTransformer(model_name)
        
        # Define the distinct intents for AppleSupport based on EDA
        self.intents = {
            "ios_update_issue": "Problems, bugs, or slowness after updating iOS to a new version.",
            "battery_drain": "Battery dying quickly or draining too fast.",
            "app_crash": "Applications are crashing, freezing, or failing to load.",
            "wifi_connectivity": "Wi-Fi is disconnecting frequently or failing to connect.",
            "general_inquiry": "A general question about an Apple product, feature, or service."
        }
        
        # Precompute embeddings for the intents
        self.intent_keys = list(self.intents.keys())
        intent_descriptions = list(self.intents.values())
        self.intent_embeddings = self.model.encode(intent_descriptions, convert_to_tensor=True)

    def classify(self, text: str):
        """
        Given a user tweet, returns the most likely intent and its confidence score.
        """
        # Embed the query
        query_embedding = self.model.encode(text, convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = util.cos_sim(query_embedding, self.intent_embeddings)[0]
        
        # Find the highest score
        best_idx = torch.argmax(cosine_scores).item()
        best_intent = self.intent_keys[best_idx]
        confidence = cosine_scores[best_idx].item()
        
        return best_intent, confidence

if __name__ == "__main__":
    classifier = IntentClassifier()
    test_text = "Okay @AppleSupport I used my phone for 2 minutes and it drains it down 8 percent"
    intent, conf = classifier.classify(test_text)
    print(f"Text: {test_text}")
    print(f"Predicted Intent: {intent} (Confidence: {conf:.2f})")
