# EXP-117: quote-trigger neuron (pre-registered, not run)

Pre-registration: D-131 (DECISIONS.md). Cancelled: D-133.

Plan: one closed-form decision neuron at layer 34 that makes E2B open its thought with a code fence ('```', token 2717), fitted on
348 dev thought-start states (tools/analysis/build_thought_starts.py), tested on FEB.

Why it was not run: EXP-116's fence arm, which forces exactly what this neuron would write (the thought opening with a python code
fence), was inert on FEB (145 first edits applying against 153 unedited), while the full commitment sentence raised them to 221. A
faithful implementation of an inert intervention would only confirm the null, so the chained launch on EXP-116's A100 was cancelled
at 02:21 UTC on 2026-11-10 and the runtime stopped. The sentence itself needs 28 tokens; EXP-118 tried to write it with one neuron per
token, EXP-120 to compile its effect.
