"""What the harness's history summaries keep of the task statement, from an official run's request log (log_requests).
A summariser request is a logged request whose system turn is empty and whose user turn begins with ADK's summarising instruction.
First summaries see the original task statement ("Problem Statement:" ... "## Task Budget"); later ones see only the previous summary.
For first summaries: the share of the statement's lines (20+ characters) and of its back-quoted identifiers that appear verbatim in
the summary. Usage: python tools/analysis/summary_loss.py <run folder>"""
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import served_prompts as SP  # noqa: E402
from sne.runner import tokenizer  # noqa: E402

run = sys.argv[1]
reqs = SP.read_log(os.path.join(run, "results", "vllm_requests.log.gz"))
tok = tokenizer()
first, later, rows = 0, 0, []
for k, v in reqs.items():
    p = v.get("prompt_text") or ""
    if "The following is a conversation history between a user and an AI" not in p[:400]:
        continue
    out = tok.decode(v["output_ids"]) if v.get("output_ids") else (v.get("output_text") or "")
    summary = out.split("<channel|>")[-1]
    m = re.search(r"Problem Statement:\n(.*?)\n## Task Budget", p, re.S)
    if not m:
        later += 1
        continue
    first += 1
    st = m.group(1)
    lines = [l.strip() for l in st.splitlines() if len(l.strip()) >= 20]
    idents = set(re.findall(r"`([^`\n]{2,60})`", st))
    rows.append(dict(statement_chars=len(st), summary_chars=len(summary), lines=len(lines), lines_kept=sum(l in summary for l in lines),
                     idents=len(idents), idents_kept=sum(i in summary for i in idents), prompt_tokens=len(v["prompt_ids"])))
print(f"summariser requests: {first + later} (first summaries of a run: {first}; later ones, which no longer see the statement: {later})")
if rows:
    tl, kl = sum(r["lines"] for r in rows), sum(r["lines_kept"] for r in rows)
    ti, ki = sum(r["idents"] for r in rows), sum(r["idents_kept"] for r in rows)
    print(f"task statement: median {int(statistics.median(r['statement_chars'] for r in rows))} characters; summary: median {int(statistics.median(r['summary_chars'] for r in rows))} characters")
    print(f"statement lines of 20+ characters kept verbatim in the summary: {kl} of {tl} ({100 * kl / max(1, tl):.0f}%)")
    print(f"back-quoted identifiers of the statement kept in the summary: {ki} of {ti} ({100 * ki / max(1, ti):.0f}%)")
    print(f"first summaries that keep no statement line verbatim: {sum(r['lines_kept'] == 0 and r['lines'] > 0 for r in rows)} of {sum(r['lines'] > 0 for r in rows)}")
