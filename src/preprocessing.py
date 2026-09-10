import pandas as pd
import os

def filter_massive_dataset(csv_path: str, target_brand: str = 'AppleSupport', chunksize: int = 100000) -> pd.DataFrame:
    """
    Reads the massive twcs.csv in chunks to avoid memory crashes.
    Extracts all replies from the target brand, and then extracts the 
    corresponding customer queries.
    """
    print(f"Scanning massive dataset for {target_brand} replies in chunks...")
    
    brand_replies_list = []
    
    # 1. First Pass: Find all replies made by the brand
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        # Outbound: brand to customer
        brand_chunk = chunk[(chunk['author_id'] == target_brand) & (chunk['inbound'] == False)]
        brand_replies_list.append(brand_chunk)
        
    brand_replies_df = pd.concat(brand_replies_list)
    print(f"Found {len(brand_replies_df)} total replies by {target_brand}.")
    
    # Get the IDs of the tweets the brand was responding to
    target_customer_tweet_ids = set(brand_replies_df['in_response_to_tweet_id'].dropna().unique())
    
    # 2. Second Pass: Find all the customer queries that match those IDs
    print(f"Scanning for the {len(target_customer_tweet_ids)} matching customer queries...")
    customer_queries_list = []
    
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        # Inbound: customer to brand
        customer_chunk = chunk[(chunk['inbound'] == True) & (chunk['tweet_id'].isin(target_customer_tweet_ids))]
        customer_queries_list.append(customer_chunk)
        
    customer_queries_df = pd.concat(customer_queries_list)
    print(f"Found {len(customer_queries_df)} matching customer queries.")
    
    # 3. Merge them into pairs
    pairs = pd.merge(
        brand_replies_df, 
        customer_queries_df, 
        left_on='in_response_to_tweet_id', 
        right_on='tweet_id',
        suffixes=('_brand', '_customer')
    )
    
    return pairs

def filter_substantive_replies(pairs: pd.DataFrame, min_length: int = 50) -> pd.DataFrame:
    """
    Filters out very short replies which are often just "Please DM us".
    """
    substantive = pairs[pairs['text_brand'].str.len() > min_length]
    return substantive

if __name__ == "__main__":
    # Change this to the path of your massive twcs.csv file
    massive_csv_file = "data/raw/twcs.csv"
    
    if not os.path.exists(massive_csv_file):
        print(f"File not found: {massive_csv_file}. Please place the massive twcs.csv here.")
    else:
        # 1. Process massive dataset
        pairs = filter_massive_dataset(massive_csv_file, target_brand='AppleSupport')
        
        # 2. Filter for substantive responses
        substantive_pairs = filter_substantive_replies(pairs)
        print(f"\nFinal substantive pairs for AppleSupport: {len(substantive_pairs)}")
        
        # 3. Save the result
        output_file = "data/AppleSupport_massive_pairs.csv"
        substantive_pairs.to_csv(output_file, index=False)
        print(f"Saved massive pairs to {output_file}!")
