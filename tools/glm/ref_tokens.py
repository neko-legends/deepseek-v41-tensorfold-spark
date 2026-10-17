#!/usr/bin/env python3
"""Greedy tokens of GLM-5.3-Flash: the reference engine's recorded once, then tensorfold's compared with them.

  ref_tokens.py record URL OUT.json      the reference (vLLM): each prompt's ids (/tokenize, thinking off), its greedy
                                         completion over those ids with 5 top logprobs a position (token ids kept)
  ref_tokens.py compare URL OUT.json     ours (tensorfold): the same ids, greedy; per prompt the matching prefix in
                                         tokens, and at the first difference whether ours was the reference's runner-up
                                         within 0.5 nats (a near tie, not a fault on its own)

The reference must be serving when recording; only short requests (MAX_TOKENS each).
"""
import json, sys, urllib.request

MAX_TOKENS = 48
PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a haiku about the sea.",
    "List the first ten prime numbers, comma separated.",
    "Explain in two sentences why the sky is blue.",
    "Translate to German: The weather is nice today and we are going for a walk.",
    "def fib(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    # a long prompt (several prompt chunks): a text repeated, then a question about it
    ("The lighthouse keeper logged the weather every hour: wind, swell, visibility and the ships that passed. " * 120)
    + "\nHow often did the keeper log the weather? One short sentence.",
]


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def model(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def tid(s):
    return int(s.split(":", 1)[1]) if isinstance(s, str) and s.startswith("token_id:") else s


def record(base, out):
    m = model(base)
    rows = []
    for p in PROMPTS:
        ids = post(base + "/tokenize", {"model": m, "messages": [{"role": "user", "content": p}],
                                         "chat_template_kwargs": {"enable_thinking": False}})["tokens"]
        r = post(base + "/v1/completions", {"model": m, "prompt": ids, "max_tokens": MAX_TOKENS, "temperature": 0,
                                            "logprobs": 5, "return_tokens_as_token_ids": True, "skip_special_tokens": False})
        c = r["choices"][0]
        lp = c["logprobs"]
        rows.append({"prompt": p[:80], "ids": ids, "text": c["text"], "finish": c["finish_reason"],
                     "tokens": [tid(t) for t in lp["tokens"]],
                     "top": [{str(tid(k)): v for k, v in (d or {}).items()} for d in lp["top_logprobs"]]})
        print(f"{len(ids):6d} prompt ids -> {len(rows[-1]['tokens'])} tokens: {c['text'][:60]!r}")
    json.dump({"model": m, "max_tokens": MAX_TOKENS, "rows": rows}, open(out, "w"), indent=1, ensure_ascii=False)


def compare(base, path):
    ref = json.load(open(path))
    m = model(base)
    bad = 0
    for row in ref["rows"]:
        r = post(base + "/v1/completions", {"model": m, "prompt": row["ids"], "max_tokens": ref["max_tokens"], "temperature": 0})
        text = r["choices"][0]["text"]
        mine = post(base + "/tokenize", {"model": m, "prompt": text, "add_special_tokens": False})["tokens"]
        want = row["tokens"]
        n = 0
        while n < min(len(mine), len(want)) and mine[n] == want[n]:
            n += 1
        note = "equal" if text == row["text"] or (n == len(want) and n <= len(mine)) else ""
        if not note and n < len(mine) and n < len(want):
            top = row["top"][n]
            first = top.get(str(want[n]))
            ours = top.get(str(mine[n]))
            if ours is not None and first is not None and first - ours <= 0.5:
                note = f"near tie at token {n} (ref {first:.3f} vs ours {ours:.3f})"
            else:
                note = f"DIFFER at token {n} (ours {mine[n]} logprob {ours} in ref top-5; ref {want[n]} {first})"
                bad += 1
        elif not note:
            note = f"DIFFER in length ({len(mine)} vs {len(want)} tokens)"
            bad += 1
        print(f"{len(row['ids']):6d} prompt ids: {n}/{len(want)} tokens match; {note}; ours {text[:60]!r}")
    print("PASS" if bad == 0 else f"FAIL: {bad} prompt(s) differ beyond a near tie")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("record", "compare"):
        sys.exit(__doc__)
    sys.exit(record(sys.argv[2], sys.argv[3]) if sys.argv[1] == "record" else compare(sys.argv[2], sys.argv[3]))
