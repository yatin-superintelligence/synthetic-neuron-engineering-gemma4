import json, glob, collections, os
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
def tdir(r):
    for b in ('artifacts_local/colab', 'artifacts_local/kaggle'):
        p = os.path.join(b, r.lower(), 'results', 'traces')
        if os.path.isdir(p): return p
SOME = ['fastapi_14301', 'rich_3006', 'rich_3454', 'fastapi_9425', 'fastapi_14786', 'rich_2943', 'requests_6644']
ABBR = {'read_file': 'R', 'edit_file': 'E', 'run_command': 'C', 'search_similar_code': 'S', 'submit_patch': 'P', 'write_file': 'W',
        'get_status': 'G', 'code_analyzer_agent': 'A', 'get_code_neighbors': 'N', 'get_code_subgraph': 'U'}
for iid in SOME:
    print('=====', iid)
    for ok_want in (True, False):
        shown = 0
        for r in runs:
            if off[r].get(iid) != ok_want or tdir(r) is None: continue
            f = os.path.join(tdir(r), f'trace_{iid}.json')
            if not os.path.exists(f): continue
            t = json.load(open(f, encoding='utf-8'))
            seq = ''
            for s in t['steps']:
                if s.get('source') != 'agent': continue
                for c in s.get('tool_calls') or []:
                    nm = c.get('function_name'); a = c.get('arguments') or {}
                    o = s.get('observation') or {}
                    txt = o.get('content') if isinstance(o, dict) else str(o)
                    err = '"status": "error"' in (txt or '')[:200]
                    ch = ABBR.get(nm, '?')
                    if nm == 'run_command' and 'pytest' in str(a.get('command', '')): ch = 'T'
                    seq += ch.lower() if err else ch
            print(('OK  ' if ok_want else 'FAIL'), r, seq[:70])
            shown += 1
            if shown >= 4: break
