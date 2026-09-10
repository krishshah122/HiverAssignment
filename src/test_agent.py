"""
Interactive test script: feed sample tweets, see what the agent produces, compare against ideal.

Usage:
    python src/test_agent.py
"""

import json
import sys
import io
import pandas as pd

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from agent import SupportAgent


def run_side_by_side_test():
    """
    Takes sample inputs from the golden set, runs the agent,
    and prints the generated reply vs the ideal reply side by side.
    """
    df = pd.read_csv("data/golden_set.csv")
    samples = df.sample(n=5, random_state=7).reset_index(drop=True)

    agent = SupportAgent()

    print("=" * 70)
    print("  SIDE-BY-SIDE TEST: Agent Output vs Ideal Response")
    print("=" * 70)

    for i, row in samples.iterrows():
        user_id = f"test_{row['user_tweet_id']}"
        query = str(row['user_query'])
        ideal = str(row['ideal_response'])
        expected_intent = str(row['intent'])

        result = agent.process_tweet(user_id, query)

        print(f"\n{'-' * 70}")
        print(f"SAMPLE {i+1}")
        print(f"{'-' * 70}")
        print(f"CUSTOMER INPUT:")
        print(f"   {query[:200]}")
        print()
        print(f"PREDICTED INTENT:  {result.get('intent')} (conf: {result.get('intent_confidence', 'N/A')})")
        print(f"EXPECTED INTENT:   {expected_intent}")
        print()

        if result['status'] == 'escalated':
            print(f"STATUS: ESCALATED -> {result['reason']}")
            print(f"   (Agent decided this needs a human)")
        else:
            print(f"STATUS: AUTO-HANDLED")
            print()
            print(f"AGENT DRAFT:")
            print(f"   {result['draft'][:280]}")
            print()
            print(f"IDEAL RESPONSE:")
            print(f"   {ideal[:280]}")
            print()
            if 'retrieved_cases' in result:
                print(f"RETRIEVED HISTORICAL CASES USED:")
                for j, case in enumerate(result['retrieved_cases']):
                    print(f"   Case {j+1} (score: {case['score']:.4f})")
                    print(f"     Past query: {case['past_query']}")
                    print(f"     Past reply: {case['past_reply']}")

    print(f"\n{'=' * 70}")
    print("  TEST COMPLETE")
    print(f"{'=' * 70}")


def run_custom_test():
    """
    Test with your own custom tweets.
    """
    agent = SupportAgent()

    custom_tweets = [
        "My iPhone battery dies in 30 minutes after the new update @AppleSupport",
        "Hey @AppleSupport none of my apps are opening they just crash immediately",
        "@AppleSupport wifi keeps disconnecting every 5 minutes on my iPad",
        "Can I get a replacement phone? Mine is completely broken @AppleSupport",
        "Love the new iOS! But my phone gets really hot. Is that normal? @AppleSupport",
    ]

    print("=" * 70)
    print("  CUSTOM INPUT TEST: See Agent Behavior on New Tweets")
    print("=" * 70)

    for i, tweet in enumerate(custom_tweets):
        user_id = f"custom_{i}"
        result = agent.process_tweet(user_id, tweet)

        print(f"\n{'-' * 70}")
        print(f"INPUT:  {tweet}")
        print(f"INTENT: {result.get('intent')} (conf: {result.get('intent_confidence', 'N/A')})")
        print(f"STATUS: {result['status'].upper()}")

        if result['status'] == 'auto-handled':
            print(f"REPLY:  {result['draft'][:280]}")
        else:
            print(f"REASON: {result['reason']}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    print("\n[1] Running side-by-side test (Golden Set samples)...\n")
    run_side_by_side_test()

    print("\n\n[2] Running custom input test...\n")
    run_custom_test()
