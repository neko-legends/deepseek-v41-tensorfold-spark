# Fetch named tensors' raw bytes from the HF repo by range request (offsets from headers-hf.json); cached as files.
import http.client, json, os, sys, time, urllib.request
# GLM_HEADERS: hf_headers.py's output; GLM_TENSORS: the cache folder; GLM_REPO: the HF repo (FP4-Local by default).
H = json.load(open(os.environ["GLM_HEADERS"]))
REPO = os.environ.get("GLM_REPO", "nvidia/GLM-5.3-Flash-FP4-Local")
CACHE = os.environ["GLM_TENSORS"]
os.makedirs(CACHE, exist_ok=True)

def meta(name):
    shard = H["index"]["weight_map"][name]
    s = H["shards"][shard]
    t = s["tensors"][name]
    return shard, 8 + s["header_len"], t

def fetch(name):
    path = os.path.join(CACHE, name + ".bin")
    if os.path.exists(path):
        return open(path, "rb").read()
    shard, base, t = meta(name)
    a, b = t["data_offsets"]
    if b == a:
        data = b""
    else:
        r = urllib.request.Request(f"https://huggingface.co/{REPO}/resolve/main/{shard}", headers={"Range": f"bytes={base + a}-{base + b - 1}"})
        for attempt in range(6):
            try:
                data = urllib.request.urlopen(r, timeout=120).read()
                break
            except (OSError, http.client.HTTPException):
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
    assert len(data) == b - a, (name, len(data), b - a)
    open(path + ".tmp", "wb").write(data); os.replace(path + ".tmp", path)
    return data

if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(n, len(fetch(n)), meta(n)[2]["dtype"], meta(n)[2]["shape"])
