import json, re, difflib, statistics as stt
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|$)", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)
S = json.load(open('artifacts_local/EXP-092/states.json', encoding='utf-8'))
S = {s['iid']: s for s in (S if isinstance(S, list) else S['states'])}
G = json.load(open('artifacts_local/EXP-092/base_greedy.json', encoding='utf-8'))
F = json.load(open('artifacts_local/EXP-092/fit_samples.json', encoding='utf-8'))
def dsize(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2-i1, j2-j1) for op,i1,i2,j1,j2 in sm.get_opcodes() if op != 'equal')
def rows(items):
    out = {'correct': [], 'wrong': []}
    for iid, t, ok, ap in items:
        st = S.get(iid)
        if not st or not ap: continue
        m = CALL.search(t)
        if not m: continue
        a = dict(ARG.findall(m.group(2)))
        if "old_string" not in a or "new_string" not in a: continue
        out['correct' if ok else 'wrong'].append((dsize(a['old_string'], a['new_string']), len(a['old_string'])))
    return out
gi = [(i, g['text'], g['correct'], g['applies']) for i, g in G.items() if g.get('text')]
fi = [(a['iid'], a['text'], a['correct'], a.get('applies', True)) for a in F if a.get('text')]
gold = [dsize(s['gold']['old_string'], s['gold']['new_string']) for s in S.values()]
print('gold diff chars: median', stt.median(gold), 'mean %.1f' % stt.mean(gold))
for name, it in (('greedy', gi), ('sampled T=1', fi)):
    r = rows(it)
    for k, v in r.items():
        d = [x[0] for x in v]; L = [x[1] for x in v]
        if d: print(f"{name:12s} {k:8s} n={len(d):4d} diff chars median {stt.median(d):5.1f} mean {stt.mean(d):6.1f} | share diff<=6: {sum(x<=6 for x in d)/len(d):.2f} | old_string len median {stt.median(L)}")
print('keys in fit_samples row:', list(F[0].keys()))
