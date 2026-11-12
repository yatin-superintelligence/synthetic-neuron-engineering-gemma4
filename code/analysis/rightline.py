import json, re, collections
CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|$)", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)
S = json.load(open('artifacts_local/EXP-092/states.json', encoding='utf-8'))
S = {s['iid']: s for s in (S if isinstance(S, list) else S['states'])}
G = json.load(open('artifacts_local/EXP-092/base_greedy.json', encoding='utf-8'))
E = json.load(open('scratch/analysis/fail_taxonomy_ids.json'))
byop = collections.defaultdict(collections.Counter)
inv = {i: c for c, ids in E.items() for i in ids}
for iid, st in S.items():
    if iid in inv: byop[st['operator']][inv[iid]] += 1
for op, c in sorted(byop.items(), key=lambda x: -sum(x[1].values())):
    n = sum(c.values()); print(f"{op:28s} n={n:3d} correct={c['CORRECT']:3d} rightline-wrong={c['applies, wrong fix: right line']:3d} otherline={c['applies, wrong fix: other line']:3d} read={c['other tool: read_file']:3d} notool={c['no tool call (thought never closed)']+c['no tool call']:3d}")
print()
for iid in E['applies, wrong fix: right line'][:10]:
    st = S[iid]; t = G[iid]['text']; m = CALL.search(t); a = dict(ARG.findall(m.group(2)))
    print('==', st['operator'], '| gold:', repr(st['gold']['old_string'][:120]), '->', repr(st['gold']['new_string'][:120]))
    print('   model:', repr(a.get('old_string','')[:120]), '->', repr(a.get('new_string','')[:120]))
