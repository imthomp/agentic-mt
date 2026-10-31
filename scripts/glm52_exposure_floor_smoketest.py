"""GLM-5.2 exposure-floor smoke test -- replicates Phase 6's methodology
(same sentences, same seed, same 5 pairs) but against a frontier-scale
model (GLM-5.2, 753B MoE, served locally via vLLM) instead of BLOOMZ (7B).

Motivation: every finding in this project that depends on model generation
quality (the exposure floor, the CoT confound, joint-necessity) comes from
small/weak models (InkubaLM 0.4B, BLOOMZ 7B, Aya-101 13B). "A small model
trained on a curated corpus doesn't know Zulu" isn't a discovery on its
own -- this checks whether the exposure floor Phase 6 found is a real
phenomenon (frontier model hits it too) or a BLOOMZ-specific artifact
(frontier model clears it), which the project currently can't distinguish.

Talks to a GLM-5.2 server already running via serve_glm52.sbatch (same
pattern as /nobackup/autodelete/usr/it238/glm52/glm_harness.py: poll
server_node.txt, wait for /v1/models to respond, POST /v1/chat/completions).
Reuses this project's own data loaders (sample_test_and_tm_corpus) so the
exact same 10 sentences per pair as Phase 6 are used -- output is directly
comparable, not just similarly-shaped.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from agentic_mt.pipelines.data import sample_test_and_tm_corpus

SERVER_NODE_FILE = Path("/nobackup/autodelete/usr/it238/glm52/server_node.txt")
PORT = 8000
N_TEST = 10
SEED = 42
PAIRS = {"en-fr": "French", "en-sw": "Swahili", "en-yo": "Yoruba", "en-xh": "Xhosa", "en-zu": "Zulu"}

OUT_DIR = Path("results/glm52_exposure_floor_smoketest")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server() -> str:
    # Re-reads server_node.txt on every retry (rather than caching it once)
    # since the file may hold a stale node from a previous, now-dead server
    # run until the fresh serve job overwrites it -- this must not lock
    # onto an address before the current run's node has actually been
    # published, and must not require deleting/touching the file ourselves.
    print("Waiting for server_node.txt and a responding server...", flush=True)
    waited = 0
    last_node = None
    while True:
        node = SERVER_NODE_FILE.read_text().strip() if SERVER_NODE_FILE.exists() else ""
        if node:
            if node != last_node:
                print(f"Trying server node: {node}", flush=True)
                last_node = node
            base_url = f"http://{node}:{PORT}"
            try:
                req = urllib.request.Request(f"{base_url}/v1/models")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        print(f"Server ready after {waited}s.", flush=True)
                        return base_url
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                pass
        time.sleep(30)
        waited += 30
        if waited % 600 == 0:
            print(f"Still waiting for server ({waited}s elapsed)...", flush=True)


def translate(base_url: str, source_text: str, tgt_lang: str) -> str:
    messages = [
        {"role": "system", "content": "You are a professional translator. Output only the translation, nothing else."},
        {"role": "user", "content": f"Translate the following English text into {tgt_lang}:\n\n{source_text}"},
    ]
    payload = {
        "model": "glm-5.2",
        "messages": messages,
        "n": 1,
        "temperature": 0,
        "max_tokens": 512,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def main():
    base_url = wait_for_server()
    rows = []
    for pair, tgt_lang in PAIRS.items():
        test_df, _ = sample_test_and_tm_corpus(pair, N_TEST, SEED)
        for _, row in test_df.iterrows():
            output = translate(base_url, row["source"], tgt_lang)
            rows.append({"pair": pair, "source": row["source"], "output": output})
            print(f"{pair}: done", flush=True)

    out_path = OUT_DIR / "smoketest_outputs.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
