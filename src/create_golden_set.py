import pandas as pd
import os
from intent_classifier import IntentClassifier

def create_golden_set(pairs_csv: str, output_csv: str, sample_size: int = 200):
    """
    Bootstraps the golden evaluation set by labeling the customer's intent
    using the embedding-based IntentClassifier.
    Samples 200 examples to meet the assignment requirement.
    """
    if not os.path.exists(pairs_csv):
        print(f"Error: {pairs_csv} not found.")
        return
        
    df = pd.read_csv(pairs_csv)
    print(f"Loaded {len(df)} pairs from {pairs_csv}.")
    
    # Sample exactly 200 rows for the Golden Set
    df = df.sample(n=sample_size, random_state=42).copy()
    print(f"Sampled {sample_size} pairs for the Golden Evaluation Set.")
    
    classifier = IntentClassifier()
    
    intents = []
    confidences = []
    
    for idx, row in df.iterrows():
        customer_text = str(row['text_customer'])
        intent, conf = classifier.classify(customer_text)
        intents.append(intent)
        confidences.append(conf)
        
    df['intent'] = intents
    df['intent_confidence'] = confidences
    
    # We rename columns to be clearer for our evaluation harness later
    golden_df = df[['tweet_id_customer', 'author_id_customer', 'text_customer', 
                    'intent', 'intent_confidence', 'text_brand']].copy()
    
    golden_df.rename(columns={
        'tweet_id_customer': 'user_tweet_id',
        'author_id_customer': 'user_id',
        'text_customer': 'user_query',
        'text_brand': 'ideal_response'
    }, inplace=True)
    
    golden_df.to_csv(output_csv, index=False)
    print(f"Successfully created golden set with {len(golden_df)} examples at {output_csv}")
    
if __name__ == "__main__":
    create_golden_set("data/AppleSupport_massive_pairs.csv", "data/golden_set.csv")
