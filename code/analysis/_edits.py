"""One rule for whether an edit_file call applied: the harness answered it with status ok.

A call the harness refused before running the tool (a mandatory parameter was missing: the answer has an "error" key and no status)
did not apply. Until 2026-11-11 the analyses counted every answer without '"status": "error"' as applied, which counted those refused
calls as applied (7 of 117 edit calls in EXP-044d, unedited E2B; 61 of 222 in EXP-074c, unedited 31B)."""
import json
import re

_OK = re.compile(r"status[\\\"'\s:]+ok")


def applied(observation):
    o = observation if isinstance(observation, str) else json.dumps(observation or {})
    return _OK.search(o) is not None
