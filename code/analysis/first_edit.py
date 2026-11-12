import json, glob, collections, os
from _edits import applied
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
SOME = {'fastapi_14301', 'rich_3006', 'rich_3454', 'fastapi_9425', 'fastapi_14786', 'rich_2943', 'requests_6644'}
tab = collections.Counter()
for r in runs:
    d = next((os.path.join(b, r.lower(), 'results', 'traces') for b in ('artifacts_local/colab', 'artifacts_local/kaggle') if os.path.isdir(os.path.join(b, r.lower(), 'results', 'traces'))), None)
    if not d: continue
    for iid, ok in off[r].items():
        if iid not in SOME: continue
        f = os.path.join(d, f'trace_{iid}.json')
        if not os.path.exists(f): continue
        t = json.load(open(f, encoding='utf-8'))
        first = None
        for s in t['steps']:
            if s.get('source') != 'agent': continue
            for c in s.get('tool_calls') or []:
                if c.get('function_name') == 'edit_file' and first is None:
                    o = s.get('observation') or {}
                    txt = o.get('content') if isinstance(o, dict) else str(o)
                    first = 'applied' if applied(o) else 'failed'
        tab[(first or 'never edited', ok)] += 1
for k in ('applied', 'failed', 'never edited'):
    s, f_ = tab[(k, True)], tab[(k, False)]
    print(f'{k:13s} runs {s + f_:3d} | resolved {s:3d} ({100 * s / max(1, s + f_):.0f}%)')
