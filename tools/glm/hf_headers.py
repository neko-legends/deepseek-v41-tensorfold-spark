# Fetch safetensors headers of every shard via HTTP range requests (header bytes only).
import json, struct, sys, urllib.request
repo = sys.argv[1]; base = f"https://huggingface.co/{repo}/resolve/main/"
def get(name, a, b):
    r = urllib.request.Request(base + name, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(r, timeout=60).read()
idx = json.loads(urllib.request.urlopen(base + "model.safetensors.index.json", timeout=60).read())
out = {"index": idx, "shards": {}}
for s in sorted(set(idx["weight_map"].values())):
    n = struct.unpack("<Q", get(s, 0, 7))[0]
    out["shards"][s] = {"header_len": n, "tensors": json.loads(get(s, 8, 8 + n - 1))}
    print(s, n, file=sys.stderr)
json.dump(out, sys.stdout)
