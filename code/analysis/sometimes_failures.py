"""How E2B's runs on the four 'sometimes' tasks end, solved against failed: steps, edits, last prompt size, whether a patch was submitted."""
import json, glob, collections, os, statistics as st
off = collections.defaultdict(dict)
for f in glob.glob('artifacts_local/kaggle_verify/*/verify_results.json'):
    for x in json.load(open(f, encoding='utf-8')):
        off[x['run']][x['iid']] = bool(x['resolved'])
cfg = {}
for f in glob.glob('experiments/colab_configs/*.json'):
    try:
        for e in json.load(open(f, encoding='utf-8')):
            if isinstance(e, dict) and 'id' in e: cfg[e['id'].upper()] = e
    except Exception: pass
runs = [r for r in off if cfg.get(r.upper(), {}).get('model', '').startswith('gemma-4-e2b') and len(off[r]) >= 56]
SOME = ['fastapi_14301', 'rich_3006', 'rich_3454', 'fastapi_9425']
rows = collections.defaultdict(list)
for r in runs:
    d = next((os.path.join(b, r.lower(), 'results', 'traces') for b in ('artifacts_local/colab', 'artifacts_local/kaggle') if os.path.isdir(os.path.join(b, r.lower(), 'results', 'traces'))), None)
    if not d: continue
    for iid in SOME:
        p = os.path.join(d, f'trace_{iid}.json')
        if not os.path.exists(p) or iid not in off[r]: continue
        t = json.load(open(p, encoding='utf-8'))
        ag = [s for s in t['steps'] if s.get('source') == 'agent']
        calls = [c.get('function_name') for s in ag for c in (s.get('tool_calls') or [])]
        last_prompt = max([(s.get('metrics') or {}).get('prompt_tokens', 0) for s in ag] or [0])
        rows[off[r][iid]].append(dict(steps=len(ag), edits=calls.count('edit_file'), submitted='submit_patch' in calls, last_prompt=last_prompt))
for ok in (True, False):
    R = rows[ok]
    if not R: continue
    print('SOLVED' if ok else 'FAILED', len(R), '| median steps', st.median(x['steps'] for x in R), '| median edits', st.median(x['edits'] for x in R),
          '| submitted', sum(x['submitted'] for x in R), '| last prompt >= 28k tokens', sum(x['last_prompt'] >= 28000 for x in R), '| median last prompt', st.median(x['last_prompt'] for x in R))
