import json, re, collections, difflib
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|$)", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)
S = json.load(open('artifacts_local/EXP-092/states.json', encoding='utf-8'))
S = {s['iid']: s for s in (S if isinstance(S, list) else S['states'])}
G = json.load(open('artifacts_local/EXP-092/base_greedy.json', encoding='utf-8'))
cat = collections.Counter(); ex = collections.defaultdict(list)
for iid, g in G.items():
    st = S.get(iid); t = g.get('text') or ''
    if st is None: continue
    exc = st['excerpt']; gold = st['gold']
    m = CALL.search(t)
    if not m:
        c = 'no tool call' + (' (thought never closed)' if '<channel|>' not in t else '')
    elif m.group(1) != 'edit_file':
        c = 'other tool: ' + m.group(1)
    else:
        a = dict(ARG.findall(m.group(2))); old = a.get('old_string') or ''; new = a.get('new_string')
        n = exc.count(old) if old else 0
        if g['correct']: c = 'CORRECT'
        elif g['applies']:
            # applied but wrong: did it touch the right line?
            c = 'applies, wrong fix: right line' if gold['old_string'].strip() in old or old.strip() in gold['old_string'] else 'applies, wrong fix: other line'
        elif not old: c = 'empty old_string'
        elif n > 1: c = 'old_string not unique'
        elif '\\"' in old and '\\"' not in exc: c = 'escape \\" in old_string'
        else:
            lines = [l for l in old.split('\n')]
            ok_strip = all(l.strip() in exc for l in lines if l.strip())
            th = t[: t.find('<|tool_call>')]
            bad = [l.strip() for l in lines if l.strip() and l.strip() not in exc]
            if ok_strip and bad == []:
                c = 'whitespace/indent mismatch only'
            elif bad and all(l in th for l in bad):
                c = 'line copied from own thought (self-copy)'
            else:
                # fuzzy: closest line in file
                fl = [l.strip() for l in exc.split('\n') if l.strip()]
                r = max((difflib.SequenceMatcher(None, b, f).ratio() for b in bad[:1] for f in fl), default=0)
                c = 'line altered (near match %.1f+)' % (0.9 if r>=0.9 else 0.7 if r>=0.7 else 0.0)
    cat[c] += 1; ex[c].append(iid)
tot = sum(cat.values())
for c, n in cat.most_common(): print(f'{n:4d} {100*n/tot:5.1f}%  {c}')
print('total', tot)
json.dump(ex, open('scratch/analysis/fail_taxonomy_ids.json','w'))
