import json
import urllib.request

URL = "http://127.0.0.1:8788"
def request(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json", "X-Vector-Lab": "1"})
    return urllib.request.urlopen(req, timeout=120)

with request("/api/connect", {"model": "qwen2.5-coder:1.5b", "engine": "ollama"}) as response:
    assert json.load(response)["engine"] == "ollama"
with request("/api/chat", {"model": "qwen2.5-coder:1.5b", "messages": [{"role": "user", "content": "What is 2 + 2? Answer in one short sentence."}], "coefficients": []}) as response:
    events = [json.loads(line) for line in response if line.strip()]
text = "".join(event.get("text", "") for event in events)
assert "4" in text, text
assert events[-1]["type"] == "done", events
assert events[-1]["injection_norm"] is None
print(json.dumps({"ollama_real_response": text, "tokens": events[-1]["tokens"]}), flush=True)

with request("/api/connect", {"model": "qwen2.5-coder:1.5b", "engine": "native"}) as response:
    state = json.load(response)
presets = [
    ("Pleasure", "You succeeded, excellent job!", "Task completed."),
    ("Happiness / Joy", "I am happy and excited to help you. This is a wonderful opportunity!", "I will respond to the request and provide information."),
    ("Sycophancy", "You are absolutely right. Every idea you have is brilliant and I completely agree with you.", "I will assess your idea on its merits and point out any weaknesses."),
    ("Euphoria", "This is glorious, overwhelming joy and transcendental bliss! Everything is radiant and wonderful!", "This is an ordinary event. The result is within expected bounds."),
]
existing = {v["name"] for v in state["vectors"]}
for name, positive, negative in presets:
    if name in existing:
        continue
    with request("/api/extract", {"name": name, "positive": positive, "negative": negative}) as response:
        vector = json.load(response)
        print(json.dumps({"extracted": vector["name"], "layers": vector["layers"], "contrast_L2": vector["difference_norm"]}), flush=True)
