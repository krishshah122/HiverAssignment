"""Quick script to analyze how many AppleSupport replies are just 'DM us' redirects."""
import pandas as pd

df = pd.read_csv('data/AppleSupport_massive_pairs.csv')

dm_keywords = ['send us a dm', 'dm us', 'reply in dm', 'join us in dm', 
               'let us know in dm', 'dm your', 'https://t.co/']

def is_dm_redirect(text):
    t = str(text).lower()
    return any(k in t for k in dm_keywords)

df['is_dm'] = df['text_brand'].apply(is_dm_redirect)

dm_count = df['is_dm'].sum()
substantive = df[~df['is_dm']]

print(f"Total pairs: {len(df)}")
print(f"DM-redirect replies: {dm_count} ({dm_count/len(df)*100:.1f}%)")
print(f"Substantive replies: {len(substantive)} ({len(substantive)/len(df)*100:.1f}%)")
print()
print("Sample SUBSTANTIVE replies (actual troubleshooting):")
for i, t in enumerate(substantive['text_brand'].head(10).values):
    print(f"  [{i+1}] {t[:200]}")
print()
print("Sample DM-REDIRECT replies (what we want to REMOVE):")
for i, t in enumerate(df[df['is_dm']]['text_brand'].head(5).values):
    print(f"  [{i+1}] {t[:200]}")
