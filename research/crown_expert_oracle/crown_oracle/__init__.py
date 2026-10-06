"""Phase 5A2: a CROWN / auto_LiRPA oracle of AWPMI inside routed experts (decision 0010).

Runs in the isolated verifier environment of `research/crown_expert_oracle` (Python 3.11, torch 2.11, auto_LiRPA at a
pinned commit), never in the Splinterflow runtime environment. It reads the artifact `export.py` writes there and
imports nothing of `awpmi`.

Splinterflow code here is limited to forward code and bookkeeping: the verification graph (`graph`), the weight sets
built from what a runtime could have read (`sets`), the certificate's finite-precision assembly from the exported Phase
5A terms (`certify`), adversarial realizations (`attack`, validation only) and the driver. Every bound on the graph is
computed by auto_LiRPA.
"""
