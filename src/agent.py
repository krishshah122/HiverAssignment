"""
Production AI Support Agent for AppleSupport (Groq LLM)

Pipeline:
  New Customer Tweet
    -> Intent Classifier
    -> Risk / Confidence Gate
    -> Hybrid Retrieval (BM25 + Vector -> RRF -> Reranker)
    -> Response LLM (Groq, grounded in top 3 historical cases)
    -> Grounding / Policy Check
    -> AUTO REPLY  or  ESCALATE TO HUMAN
"""

import os
import sys
import re
import sqlite3
import json
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from intent_classifier import IntentClassifier
from hybrid_retriever import HybridRetriever

import time

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def llm_generate(prompt: str, model: str = "qwen/qwen3.8-27b") -> str:
    """Call Groq LLM. Raises on failure."""
    for attempt in range(5):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "Rate limit" in str(e):
                wait_time = 12 * (attempt + 1)
                print(f"  [Agent API Rate Limited] Sleeping for {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    return "Rate limit exceeded. Try again later."


class SupportAgent:
    def __init__(
        self,
        sqlite_path: str = "data/threads.sqlite",
        intent_threshold: float = 0.10,  # Lowered from 0.25 to prevent unnecessary escalations
        reranker_score_threshold: float = -10.0, # Lowered from -5.0 to allow more retrievals
    ):
        # 1. Intent Classifier
        self.intent_classifier = IntentClassifier()
        self.intent_threshold = intent_threshold

        # 2. Hybrid Retriever (BM25 + Vector + RRF + Reranker)
        self.retriever = HybridRetriever()
        self.reranker_score_threshold = reranker_score_threshold

        # 3. SQLite per-user thread state
        self.sqlite_conn = sqlite3.connect(sqlite_path)
        self._init_sqlite()

    # ================================================================
    # SQLite helpers
    # ================================================================
    def _init_sqlite(self):
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_threads (
                user_id TEXT,
                message TEXT,
                role TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.sqlite_conn.commit()

    def _get_thread_context(self, user_id: str) -> str:
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            'SELECT role, message FROM active_threads WHERE user_id = ? ORDER BY timestamp ASC LIMIT 5',
            (user_id,),
        )
        rows = cursor.fetchall()
        if not rows:
            return ""
        context = "Previous conversation:\n"
        for role, msg in rows:
            context += f"  [{role}]: {msg}\n"
        return context

    def _save_to_thread(self, user_id: str, message: str, role: str):
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            'INSERT INTO active_threads (user_id, message, role) VALUES (?, ?, ?)',
            (user_id, message, role),
        )
        self.sqlite_conn.commit()

    # ================================================================
    # Grounding / Policy Check (strengthened)
    # ================================================================
    def _grounding_check(self, draft: str, retrieved_cases: list) -> Dict:
        """
        Multi-layer safety and grounding validation.
        Returns {"grounded": bool, "reason": str, "details": dict}
        """
        details = {}

        # Layer 1: Basic validity
        if not draft or len(draft.strip()) < 10:
            return {"grounded": False, "reason": "Draft too short or empty", "details": {}}

        # Layer 2: Policy violation (forbidden promises)
        forbidden = [
            "refund guaranteed", "we will replace", "free of charge",
            "lawsuit", "legal action", "compensation", "we apologize for the inconvenience",
            "100% guarantee", "money back",
        ]
        for phrase in forbidden:
            if phrase.lower() in draft.lower():
                return {"grounded": False, "reason": f"Policy violation: '{phrase}'", "details": {}}

        # Layer 3: Hallucination detection (fabricated contact info)
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', draft):
            return {"grounded": False, "reason": "Hallucinated phone number", "details": {}}
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', draft):
            return {"grounded": False, "reason": "Hallucinated email address", "details": {}}
        # Check for fabricated URLs (not from retrieved cases)
        draft_urls = set(re.findall(r'https?://\S+', draft))
        retrieved_urls = set()
        for case in retrieved_cases:
            retrieved_urls.update(re.findall(r'https?://\S+', case.get('brand_response', '')))
        fabricated_urls = draft_urls - retrieved_urls
        if fabricated_urls:
            details['fabricated_urls'] = list(fabricated_urls)
            return {"grounded": False, "reason": f"Hallucinated URL: {fabricated_urls.pop()}", "details": details}

        # Layer 4: Semantic grounding check
        # Verify the draft is actually related to the retrieved cases
        # by checking keyword overlap between draft and retrieved brand responses
        draft_words = set(draft.lower().split())
        retrieved_words = set()
        for case in retrieved_cases:
            retrieved_words.update(case.get('brand_response', '').lower().split())
            retrieved_words.update(case.get('customer_query', '').lower().split())
        overlap = draft_words & retrieved_words
        overlap_ratio = len(overlap) / len(draft_words) if draft_words else 0
        details['grounding_overlap'] = round(overlap_ratio, 3)

        if overlap_ratio < 0.15:
            return {"grounded": False, "reason": f"Low grounding overlap ({overlap_ratio:.1%})", "details": details}

        # Layer 5: Length sanity (Twitter = 280 chars)
        if len(draft) > 300:
            details['draft_length'] = len(draft)
            return {"grounded": False, "reason": f"Draft too long ({len(draft)} chars)", "details": details}

        return {"grounded": True, "reason": "Passed all 5 checks", "details": details}

    # ================================================================
    # Main pipeline
    # ================================================================
    def process_tweet(self, user_id: str, tweet_text: str) -> Dict:
        # Step 1: Save incoming message
        self._save_to_thread(user_id, tweet_text, "Customer")
        full_context = self._get_thread_context(user_id)

        # Step 2: Intent Classification
        intent, intent_conf = self.intent_classifier.classify(tweet_text)

        # Step 3: Risk / Confidence Gate
        if intent_conf < self.intent_threshold:
            self._save_to_thread(user_id, f"ESCALATED: Low confidence ({intent_conf:.2f})", "System")
            return {
                "status": "escalated",
                "reason": f"Low Intent Confidence ({intent_conf:.2f})",
                "intent": intent,
                "intent_confidence": round(intent_conf, 4),
            }

        # Step 4: Hybrid Retrieval
        retrieved_cases = self.retriever.retrieve(tweet_text, final_top_k=3)

        if not retrieved_cases:
            self._save_to_thread(user_id, "ESCALATED: No historical cases found", "System")
            return {
                "status": "escalated",
                "reason": "No historical cases found",
                "intent": intent,
                "intent_confidence": round(intent_conf, 4),
            }

        best_score = retrieved_cases[0]['reranker_score']
        if best_score < self.reranker_score_threshold:
            self._save_to_thread(user_id, f"ESCALATED: Reranker score too low ({best_score:.2f})", "System")
            return {
                "status": "escalated",
                "reason": f"Reranker score too low ({best_score:.2f})",
                "intent": intent,
                "intent_confidence": round(intent_conf, 4),
            }

        # Step 5: Response LLM grounded in historical cases
        cases_block = ""
        for i, case in enumerate(retrieved_cases):
            cases_block += f"\nCase {i+1} (relevance: {case['reranker_score']:.2f}):\n"
            cases_block += f"  Customer said: \"{case['customer_query'][:200]}\"\n"
            cases_block += f"  Brand replied: \"{case['brand_response'][:200]}\"\n"

        prompt = f"""You are a highly empathetic, human customer support agent for AppleSupport on Twitter.
Your goal is to draft a helpful, friendly, and concise reply (under 280 characters).

{full_context}

Here are the top historical cases where AppleSupport resolved similar issues:
{cases_block}

RULES FOR BRAND VOICE:
- Sound like a real, caring human being, not a robotic automated system.
- Match AppleSupport's professional, warm, and conversational tone.
- Acknowledge the user's frustration gracefully if they are upset.
- Ground your troubleshooting steps ONLY in the historical cases above.
- Do NOT invent policies, phone numbers, emails, or guarantees.
- If you need more info to help, ask for it naturally.
- Output ONLY the draft reply text, nothing else."""

        draft = llm_generate(prompt)

        # Step 6: Grounding / Policy Check
        grounding = self._grounding_check(draft, retrieved_cases)

        if not grounding['grounded']:
            self._save_to_thread(user_id, f"ESCALATED: {grounding['reason']}", "System")
            return {
                "status": "escalated",
                "reason": f"Grounding check failed: {grounding['reason']}",
                "intent": intent,
                "intent_confidence": round(intent_conf, 4),
                "failed_draft": draft,
            }

        # Step 7: Auto-reply
        self._save_to_thread(user_id, draft, "Brand")

        return {
            "status": "auto-handled",
            "intent": intent,
            "intent_confidence": round(intent_conf, 4),
            "draft": draft,
            "retrieved_cases": [
                {
                    "past_query": c['customer_query'][:100],
                    "past_reply": c['brand_response'][:100],
                    "score": round(c['reranker_score'], 4),
                }
                for c in retrieved_cases
            ],
            "grounding_check": grounding,
        }


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    agent = SupportAgent()

    test_cases = [
        ("user_001", "My battery drains so fast after the iOS update, phone dies in 2 hours @AppleSupport"),
        ("user_002", "@AppleSupport my apps keep crashing since I updated to iOS 11"),
        ("user_003", "How do I get a refund for my broken charger? @AppleSupport"),
        ("user_001", "I already tried restarting, still drains fast!"),  # 2nd msg from user_001
    ]

    for user_id, tweet in test_cases:
        print(f"\n{'='*60}")
        print(f"USER [{user_id}]: {tweet}")
        print(f"{'='*60}")
        result = agent.process_tweet(user_id, tweet)
        print(json.dumps(result, indent=2, ensure_ascii=False))
