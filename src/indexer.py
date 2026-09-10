import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import shutil

DM_KEYWORDS = ['send us a dm', 'dm us', 'reply in dm', 'join us in dm',
               'let us know in dm', 'dm your', 'https://t.co/']


def is_dm_redirect(text: str) -> bool:
    """Returns True if the reply is just a 'DM us' redirect with no real content."""
    t = str(text).lower()
    return any(k in t for k in DM_KEYWORDS)


def build_vector_db(
    massive_csv_path: str = "data/AppleSupport_massive_pairs.csv",
    db_path: str = "data/chroma_db",
    index_size: int = 5000,
):
    """
    Builds the ChromaDB vector index from ONLY substantive replies.
    
    Step 1: Load all 105K AppleSupport pairs.
    Step 2: REMOVE all 'DM us' redirect replies (76% of the data).
    Step 3: Sample from the remaining 24% substantive replies.
    Step 4: Index into ChromaDB.
    """
    if not os.path.exists(massive_csv_path):
        print(f"Error: {massive_csv_path} not found.")
        return

    df = pd.read_csv(massive_csv_path)
    print(f"Loaded {len(df)} total pairs.")

    # CRITICAL: Remove DM-redirect replies
    df['is_dm'] = df['text_brand'].apply(is_dm_redirect)
    dm_count = df['is_dm'].sum()
    df = df[~df['is_dm']].copy()
    print(f"Removed {dm_count} DM-redirect replies.")
    print(f"Remaining substantive replies: {len(df)}")

    # Sample
    sample_n = min(index_size, len(df))
    df = df.sample(n=sample_n, random_state=100).reset_index(drop=True)

    # Delete old DB to start fresh
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print("Cleared old ChromaDB.")

    # Build fresh DB
    client = chromadb.PersistentClient(path=db_path)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="apple_support_resolutions",
        embedding_function=sentence_transformer_ef,
    )

    print(f"Indexing {len(df)} substantive resolutions into ChromaDB...")

    documents = df['text_customer'].astype(str).tolist()
    metadatas = [{"ideal_response": str(resp)} for resp in df['text_brand']]
    ids = [str(tid) for tid in df['tweet_id_customer']]

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )

    print(f"Successfully built Vector DB with {len(df)} substantive-only entries!")


if __name__ == "__main__":
    build_vector_db()
