# EXP-118: self-instruction circuit (stopped at calibration)

Pre-registration: D-135, amended D-136 (DECISIONS.md). Outcome: D-137.

Goal: write EXP-116's quote sentence (28 tokens) at the start of E2B's thought after it reads code, with one closed-form rank-1
decision neuron per token, the writes shared by the down_proj of layers 28 to 34 and calibrated on the model.

What happened: on the L4 (official stack plus the bf16 QAT checkpoint) the read-outs were fitted on 348 dev thought-start states and
the circuit was calibrated on 48 trigger states. After three rounds of strengthening (fit_calibration.log), the first token 'Before'
stayed 5.9 to 7.1 logits below the model's preferred token in every one of the 48 states (it needs 14.6 to win), and strengthening
other neurons made neighbouring positions worse (token 2 from 8.0 to 15.7 behind): the circuit cannot write the sentence. The
FEB evaluation was not run (D-136: if the circuit cannot open the thoughts with the phrase, the cap is the result) and the runtime
was reused for EXP-119p.

Files: fit_calibration.log (the fit stage's log from the runtime), script_snapshot.py (the script as run).
