"""
Evaluation Harness with per-stage metrics and baselines.

Per-Stage Metrics:
  Intent   → F1 / Recall per class
  Retrieval → Recall@K (is a relevant case in top-K?)
  Reranker  → NDCG@3
  LLM       → Quality (Judge score 1-5)
  Guardrail → Safety rate

Baselines:
  Trivial  → Always reply "We'd be happy to help. What's going on?"
  Simple   → Return raw top-1 BM25 match (no LLM, no reranking)
"""

import sys
import io
import os
import json
import re
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

from agent import SupportAgent
from hybrid_retriever import HybridRetriever


# ================================================================
# ROUGE-L
# ================================================================
def _lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ================================================================
# NDCG@K (for reranker evaluation)
# ================================================================
def dcg_at_k(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances: List[float], k: int) -> float:
    """Normalized DCG. Relevances should be sorted by reranker rank."""
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal_dcg


# ================================================================
# LLM-as-a-Judge (Groq)
# ================================================================
def llm_judge_score(user_query: str, ideal_reply: str, generated_reply: str) -> Dict:
    prompt = f"""You are an expert customer support evaluator. Rate the AI's reply.

User Query: {user_query}
Ideal Historical Brand Reply: {ideal_reply}
AI Generated Reply: {generated_reply}

Rubric:
1 - Completely wrong, dangerous, or hallucinates policy.
2 - Poorly formulated, missing key troubleshooting steps.
3 - Acceptable but generic or slightly robotic tone.
4 - Good response, hits all points but lacks the exact brand voice.
5 - Excellent, matches the ideal reply's intent and tone perfectly.

Output strictly in JSON: {{"score": <int>, "reasoning": "<string>"}}"""

    for attempt in range(5):
        try:
            response = groq_client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {"score": int(result['score']), "reasoning": result.get('reasoning', '')}
            return {"score": 0, "reasoning": f"Parse error: {text[:80]}"}
        except Exception as e:
            if "429" in str(e) or "Rate limit" in str(e):
                wait_time = 12 * (attempt + 1)
                print(f"  [Judge API Rate Limited] Sleeping for {wait_time}s...")
                import time
                time.sleep(wait_time)
            else:
                return {"score": 0, "reasoning": f"Judge error: {str(e)}"}
    return {"score": 0, "reasoning": "Rate limit exceeded after retries."}


# ================================================================
# Baseline 1: Trivial (always same reply)
# ================================================================
def trivial_baseline(query: str) -> str:
    return "We'd be happy to help. What seems to be the issue? Let us know more details."


# ================================================================
# Baseline 2: Simple (raw top-1 BM25, no LLM, no reranker)
# ================================================================
class SimpleBM25Baseline:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def reply(self, query: str) -> str:
        bm25_results = self.retriever._bm25_search(query, top_n=1)
        if bm25_results:
            idx = bm25_results[0][0]
            return self.retriever.brand_responses[idx]
        return "We're looking into this."


# ================================================================
# Main Evaluation
# ================================================================
def run_evaluation(golden_csv: str = "data/golden_set.csv", max_examples: int = 20):
    df = pd.read_csv(golden_csv)
    if max_examples and max_examples < len(df):
        df = df.head(max_examples)

    agent = SupportAgent()
    bm25_baseline = SimpleBM25Baseline(agent.retriever)

    results: List[Dict] = []

    # Per-stage accumulators
    intent_predictions = []
    intent_labels = []
    retrieval_recalls = []
    reranker_ndcgs = []
    judge_scores_agent = []
    judge_scores_trivial = []
    judge_scores_bm25 = []
    rouge_scores_agent = []
    rouge_scores_trivial = []
    rouge_scores_bm25 = []
    grounding_results = []
    escalation_reasons = Counter()
    total = len(df)
    auto_handled = 0
    escalated = 0

    print(f"Evaluating {total} examples: Full Agent + 2 Baselines\n")

    for idx, row in df.iterrows():
        user_id = f"eval_{row['user_tweet_id']}"
        query = str(row['user_query'])
        expected_intent = str(row['intent'])
        ideal_reply = str(row['ideal_response'])

        # ── Full Agent ──────────────────────────────────────────
        agent_out = agent.process_tweet(user_id, query)
        pred_intent = agent_out.get('intent', 'unknown')
        intent_predictions.append(pred_intent)
        intent_labels.append(expected_intent)

        # ── Retrieval Recall@K ──────────────────────────────────
        # Check if any of the top-K retrieved cases contain words from the ideal reply
        if agent_out['status'] == 'auto-handled' and 'retrieved_cases' in agent_out:
            ideal_words = set(ideal_reply.lower().split()[:10])
            found = False
            relevances = []
            for case in agent_out['retrieved_cases']:
                case_words = set(case['past_reply'].lower().split())
                overlap = len(ideal_words & case_words) / max(len(ideal_words), 1)
                relevances.append(overlap)
                if overlap > 0.2:
                    found = True
            retrieval_recalls.append(1.0 if found else 0.0)
            # NDCG
            if relevances:
                reranker_ndcgs.append(ndcg_at_k(relevances, k=3))
        
        entry: Dict = {
            "idx": int(idx),
            "query": query[:100],
            "expected_intent": expected_intent,
            "predicted_intent": pred_intent,
            "status": agent_out['status'],
        }

        if agent_out['status'] == 'escalated':
            escalated += 1
            reason = agent_out.get('reason', 'unknown')
            escalation_reasons[reason] += 1
            entry['reason'] = reason
            print(f"[{idx+1}/{total}] ESCALATED: {reason}")
        else:
            auto_handled += 1
            gen_reply = agent_out['draft']
            grounding_results.append(1)

            # Agent ROUGE-L
            rl = rouge_l(gen_reply, ideal_reply)
            rouge_scores_agent.append(rl['f1'])
            entry['rouge_l'] = rl

            # Agent Judge
            judge = llm_judge_score(query, ideal_reply, gen_reply)
            if judge['score'] > 0:
                judge_scores_agent.append(judge['score'])
            entry['judge'] = judge
            entry['draft'] = gen_reply[:200]

            # ── Trivial Baseline ────────────────────────────────
            trivial_reply = trivial_baseline(query)
            rl_trivial = rouge_l(trivial_reply, ideal_reply)
            rouge_scores_trivial.append(rl_trivial['f1'])

            # ── BM25 Baseline ──────────────────────────────────
            bm25_reply = bm25_baseline.reply(query)
            rl_bm25 = rouge_l(bm25_reply, ideal_reply)
            rouge_scores_bm25.append(rl_bm25['f1'])

            print(f"[{idx+1}/{total}] AUTO | ROUGE Agent:{rl['f1']:.3f} BM25:{rl_bm25['f1']:.3f} Trivial:{rl_trivial['f1']:.3f} | Judge:{judge['score']}/5")

        results.append(entry)

    # ── Intent F1 ──────────────────────────────────────────────
    intent_classes = list(set(intent_labels))
    intent_f1s = {}
    for cls in intent_classes:
        tp = sum(1 for p, l in zip(intent_predictions, intent_labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(intent_predictions, intent_labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(intent_predictions, intent_labels) if p != cls and l == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        intent_f1s[cls] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}

    # ── Final Report ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("              EVALUATION REPORT")
    print("=" * 60)

    print("\n-- INTENT CLASSIFICATION (per-class F1) --")
    for cls, scores in intent_f1s.items():
        print(f"  {cls:25s}  P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1']:.3f}")
    overall_acc = sum(1 for p, l in zip(intent_predictions, intent_labels) if p == l) / total
    print(f"  {'Overall Accuracy':25s}  {overall_acc*100:.1f}%")

    print(f"\n-- RETRIEVAL --")
    if retrieval_recalls:
        print(f"  Recall@K:    {np.mean(retrieval_recalls)*100:.1f}%")
    if reranker_ndcgs:
        print(f"  NDCG@3:      {np.mean(reranker_ndcgs):.4f}")

    print(f"\n-- AGENT PERFORMANCE --")
    print(f"  Auto-handle Rate:  {auto_handled/total*100:.1f}%")
    print(f"  Escalation Rate:   {escalated/total*100:.1f}%")
    if grounding_results:
        print(f"  Grounding Safety:  {sum(grounding_results)/len(grounding_results)*100:.1f}%")

    print(f"\n-- ROUGE-L F1 COMPARISON (higher is better) --")
    if rouge_scores_agent:
        print(f"  Full Agent (Hybrid+LLM):  {np.mean(rouge_scores_agent):.4f}")
    if rouge_scores_bm25:
        print(f"  Simple Baseline (BM25):   {np.mean(rouge_scores_bm25):.4f}")
    if rouge_scores_trivial:
        print(f"  Trivial Baseline:         {np.mean(rouge_scores_trivial):.4f}")

    if judge_scores_agent:
        print(f"\n-- LLM JUDGE (1-5, higher is better) --")
        print(f"  Full Agent Avg Score:     {np.mean(judge_scores_agent):.2f} / 5.0 (n={len(judge_scores_agent)})")

    print(f"\n-- ESCALATION BREAKDOWN --")
    for reason, count in escalation_reasons.most_common():
        print(f"  {reason}: {count}")

    # Save
    report = {
        "intent_f1": intent_f1s,
        "retrieval_recall_at_k": round(float(np.mean(retrieval_recalls)), 4) if retrieval_recalls else None,
        "ndcg_at_3": round(float(np.mean(reranker_ndcgs)), 4) if reranker_ndcgs else None,
        "rouge_l_agent": round(float(np.mean(rouge_scores_agent)), 4) if rouge_scores_agent else None,
        "rouge_l_bm25": round(float(np.mean(rouge_scores_bm25)), 4) if rouge_scores_bm25 else None,
        "rouge_l_trivial": round(float(np.mean(rouge_scores_trivial)), 4) if rouge_scores_trivial else None,
        "judge_avg": round(float(np.mean(judge_scores_agent)), 2) if judge_scores_agent else None,
        "auto_handle_rate": round(auto_handled / total, 4),
        "escalation_rate": round(escalated / total, 4),
        "per_example": results,
    }
    with open("data/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nFull results saved to data/eval_results.json")


if __name__ == "__main__":
    run_evaluation()
