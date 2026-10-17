#!/usr/bin/env python3
"""OpenAI API parity for GLM-5.3-Flash: engine/zig-glm's server against the reference engine (vLLM, glm45 reasoning
parser, glm47 tool parser), with no GPU on our side.

Our side is the real server on `fake_serve` (zig build fake-serve), the scripted engine, over the checkpoint's own
tokenizer and chat template (TF_FAKE_TOKENIZER=DIR with tokenizer.json, tokenizer_config.json, chat_template.jinja,
config.json, generation_config.json). Two kinds of checks:

  prompts  every request's rendered prompt ids: ours (/tokenize) == the reference's (/tokenize), so thinking on/off,
           efforts, system/tool turns and tool specs render identically (the template is the reference's).
  replies  the reference answers a request greedily (chat), and the same prompt raw (/v1/completions over its ids,
           special tokens kept); that raw text becomes our engine's script, and our server's parsed answer must equal
           the reference's: content, reasoning, tool calls (name, arguments as JSON), finish reason, prompt tokens;
           streamed too (the deltas joined, the usage chunk with stream_options.include_usage).

  api_parity.py --ref URL --fake BIN --tok DIR [--out DIR]     (starts fake_serve itself, twice: prompts, then replies)

The reference must be serving the checkpoint (greedy requests only, a few hundred tokens each).
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request

MODEL = None  # the reference's served id, read from /v1/models
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]},
                          "days": {"type": "integer"}}, "required": ["city"]}}}
ADD = {"type": "function", "function": {"name": "add", "description": "Add two integers",
       "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
OFF = {"enable_thinking": False}


def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def stream(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    chunks = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    return chunks


def user(text):
    return {"role": "user", "content": text}


# name -> request fields (messages, tools, kwargs...); the prompt check renders each, the reply check runs those with "reply"
CASES = {
    "plain_default": {"messages": [user("Say hi.")]},
    "off_enable": {"messages": [user("Say hi.")], "chat_template_kwargs": {"enable_thinking": False}},
    "off_thinking": {"messages": [user("Say hi.")], "chat_template_kwargs": {"thinking": False}},
    "on_enable": {"messages": [user("Say hi.")], "chat_template_kwargs": {"enable_thinking": True}},
    "mixed_on": {"messages": [user("Say hi.")], "chat_template_kwargs": {"enable_thinking": False, "thinking": True}},
    "mixed_on2": {"messages": [user("Say hi.")], "chat_template_kwargs": {"enable_thinking": True, "thinking": False}},
    "both_off": {"messages": [user("Say hi.")], "chat_template_kwargs": {"enable_thinking": False, "thinking": False}},
    "effort_low": {"messages": [user("Say hi.")], "chat_template_kwargs": {"reasoning_effort": "low"}},
    "effort_high": {"messages": [user("Say hi.")], "chat_template_kwargs": {"reasoning_effort": "high"}},
    "effort_max": {"messages": [user("Say hi.")], "chat_template_kwargs": {"reasoning_effort": "max"}},
    "effort_medium": {"messages": [user("Say hi.")], "chat_template_kwargs": {"reasoning_effort": "medium"}},
    "system": {"messages": [{"role": "system", "content": "You are terse."}, user("Say hi.")], "chat_template_kwargs": OFF},
    "history": {"messages": [user("Pick a color."), {"role": "assistant", "content": "Blue.", "reasoning": "Any color works."},
                             user("Another one?")], "chat_template_kwargs": OFF},
    "history_rc": {"messages": [user("Pick a color."), {"role": "assistant", "content": "Blue.", "reasoning_content": "Any color works."},
                                user("Another one?")], "chat_template_kwargs": OFF},
    "tools_spec": {"messages": [user("Weather in Tokyo?")], "tools": [WEATHER, ADD], "chat_template_kwargs": OFF},
    "tool_turn": {"messages": [user("Weather in Tokyo?"),
                               {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\", \"days\": 2}"}}]},
                               {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 21}"}],
                  "tools": [WEATHER], "chat_template_kwargs": OFF},
    # replies (the reference generates; these also render as prompts)
    "r_plain_off": {"messages": [user("What is the capital of France? One word.")], "chat_template_kwargs": OFF, "reply": 40},
    "r_think": {"messages": [user("What is 17 + 25? Reply with just the number.")], "chat_template_kwargs": {"reasoning_effort": "low"}, "reply": 600},
    "r_tool": {"messages": [user("What is the weather in Tokyo in celsius for the next 3 days? Use the tool.")], "tools": [WEATHER],
               "chat_template_kwargs": OFF, "reply": 120},
    "r_tool_two": {"messages": [user("Use the add tool twice, in one reply: 2+3 and 10+32.")], "tools": [ADD],
                   "chat_template_kwargs": OFF, "reply": 160},
    "r_tool_think": {"messages": [user("What is the weather in Paris? Use the tool.")], "tools": [WEATHER],
                     "chat_template_kwargs": {"reasoning_effort": "low"}, "reply": 600},
}


# differences we keep on purpose: reported, not counted
KNOWN = {
    "effort_medium": "ours coerces an effort the template does not name to its nearest level (medium -> High); "
                     "the reference passes it through and the template falls back to Max",
    "history_rc": "the reference drops an earlier reply's reasoning_content (it reads only `reasoning`); ours renders "
                  "both names, as the template keeps history reasoning (clear_thinking false)",
}


def tag(name, req):
    """The script name rides in the last user message (both servers render the same prompt)."""
    r = json.loads(json.dumps(req))
    r.pop("reply", None)
    for m in reversed(r["messages"]):
        if m["role"] == "user":
            m["content"] += f" @script={name}"
            break
    return r


def pieces(text):
    """Script chunks: special markers whole, other text in pieces of at most 6 characters."""
    out = []
    for part in re.split(r"(<[/|]?[a-z_]+\|?>)", text):
        if not part:
            continue
        if re.fullmatch(r"<[/|]?[a-z_]+\|?>", part):
            out.append([part])
        else:
            out.extend([part[i:i + 6]] for i in range(0, len(part), 6))
    return out


class Fake:
    def __init__(self, binary, tok, scripts, port):
        env = dict(os.environ, TF_FAKE_TOKENIZER=tok, TF_FAKE_SCRIPTS=scripts, TF_FAKE_CONTEXT="0", TENSORFOLD_REASONING_FIELDS="both")
        fixtures = os.path.dirname(scripts)
        self.p = subprocess.Popen([binary, "serve", fixtures, "--host", "127.0.0.1", "--port", str(port), "--name", MODEL],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                get(self.url + "/v1/models")
                return
            except Exception:
                time.sleep(0.1)
        raise SystemExit("fake_serve did not start")

    def close(self):
        self.p.terminate()
        self.p.wait(10)


def message_of(resp):
    m = resp["choices"][0]["message"]
    calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in (m.get("tool_calls") or [])]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "calls": calls, "finish": resp["choices"][0]["finish_reason"]}


def joined(chunks):
    content, reasoning, calls, usage, finish = "", "", {}, None, None
    for c in chunks:
        if c.get("usage") and not c.get("choices"):
            usage = c["usage"]
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            content += d.get("content") or ""
            reasoning += d.get("reasoning") or d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                f = tc.get("function") or {}
                slot["name"] += f.get("name") or ""
                slot["arguments"] += f.get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    out = [(v["name"], json.loads(v["arguments"]) if v["arguments"] else {}) for _, v in sorted(calls.items())]
    return {"content": content, "reasoning": reasoning, "calls": out, "finish": finish}, usage


def main():
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--port", type=int, default=18990)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    MODEL = get(args.ref + "/v1/models")["data"][0]["id"]
    names = [n for n in CASES if not args.only or n in args.only.split(",")]
    record, bad = {"prompts": {}, "replies": {}}, 0
    raw_scripts = {"default": {"chunks": [["<|user|>"]]}}
    ref_replies = {}
    # the reference's prompts and replies first (all its requests in one go)
    for n in names:
        req = tag(n, CASES[n])
        tok = {k: v for k, v in req.items() if k in ("messages", "tools", "chat_template_kwargs")}
        tok["model"] = MODEL
        ids = post(args.ref + "/tokenize", tok)["tokens"]
        record["prompts"][n] = {"ref": ids}
        if "reply" in CASES[n]:
            body = dict(req, model=MODEL, max_tokens=CASES[n]["reply"], temperature=0)
            chat = post(args.ref + "/v1/chat/completions", body)
            raw = post(args.ref + "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": CASES[n]["reply"], "temperature": 0,
                                                      "skip_special_tokens": False})
            text = raw["choices"][0]["text"]
            if raw["choices"][0]["finish_reason"] == "stop":
                text += "<|user|>"
            sref = stream(args.ref + "/v1/chat/completions", dict(body, stream=True, stream_options={"include_usage": True}))
            ref_replies[n] = {"chat": chat, "raw": text, "stream": sref}
            raw_scripts[n] = {"chunks": pieces(text)}
    scripts = os.path.join(args.out, "scripts.json")
    json.dump(raw_scripts, open(scripts, "w"), indent=1)
    # fake_serve reads FIXTURES/vocab.json (its own fake tokenizer, unused beside the real one): the tests' copy
    fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../engine/zig-glm/zig/tests/server/fixtures/vocab.json")
    with open(fixtures) as f, open(os.path.join(args.out, "vocab.json"), "w") as g:
        g.write(f.read())
    ours = Fake(args.fake, args.tok, scripts, args.port)
    try:
        for n in names:
            req = tag(n, CASES[n])
            tok = {k: v for k, v in req.items() if k in ("messages", "tools", "chat_template_kwargs")}
            tok["model"] = MODEL
            mine = post(ours.url + "/tokenize", tok)["tokens"]
            same = mine == record["prompts"][n]["ref"]
            record["prompts"][n]["ours"] = mine
            record["prompts"][n]["equal"] = same
            known = not same and n in KNOWN
            bad += not same and not known
            print(f"prompt {n:16s} {'equal' if same else 'known difference' if known else 'DIFFER'} ({len(mine)} vs {len(record['prompts'][n]['ref'])} ids)"
                  + (f": {KNOWN[n]}" if known else ""))
            if n not in ref_replies:
                continue
            body = dict(req, model=MODEL, max_tokens=CASES[n]["reply"], temperature=0)
            ref = ref_replies[n]
            want = message_of(ref["chat"])
            got = message_of(post(ours.url + "/v1/chat/completions", body))
            gs, gu = joined(stream(ours.url + "/v1/chat/completions", dict(body, stream=True, stream_options={"include_usage": True})))
            ws, wu = joined(ref["stream"])
            checks = {
                "message": got == want,
                "stream": gs == ws,
                "stream=message": gs == got,
                "prompt_tokens": ref["chat"]["usage"]["prompt_tokens"] == (gu or {}).get("prompt_tokens"),
                "usage chunk": gu is not None and wu is not None,
            }
            record["replies"][n] = {"ref": want, "ours": got, "ref_stream": ws, "ours_stream": gs, "raw": ref["raw"], "checks": checks}
            bad += not all(checks.values())
            print(f"reply  {n:16s} " + " ".join(f"{k}={'ok' if v else 'NO'}" for k, v in checks.items()))
            if not checks["message"]:
                print(f"   ref  {json.dumps(want, ensure_ascii=False)[:400]}\n   ours {json.dumps(got, ensure_ascii=False)[:400]}")
        models = get(ours.url + "/v1/models")
        record["models"] = {"ours": models, "ref": get(args.ref + "/v1/models")}
        print("models ids: ours", [m["id"] for m in models["data"]], "ref", [m["id"] for m in record["models"]["ref"]["data"]])
    finally:
        ours.close()
    json.dump(record, open(os.path.join(args.out, "api_parity.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{'PASS' if bad == 0 else 'FAIL'}: {bad} differing check(s); record in {args.out}/api_parity.json")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
