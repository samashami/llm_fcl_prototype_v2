# FedQCL-DPP adaptation: stop note

**Date:** 30 September 2026  
**Status:** Stopped after one completed development run; excluded from comparative claims.

We adapted FedQCL-DPP to the shared-head controlled Data-IL pipeline and fixed a duplicate-domain memory-admission error. The corrected seed-41 development run completed all 14 rounds. With the configured \(V=200\) and \(\delta=1\), all 168 per-client/domain queue violations were negative (range: \(-1.35143\) to \(-0.75992\)). Every queue therefore remained zero, so the queue-weighted penalty never affected optimization. The large displayed training losses were the current cross-entropy multiplied by \(V\) in the objective; they did not indicate a useful active queue penalty.

We stopped because this configuration never activated the mechanism being evaluated. We did not tune \(\delta\), change the method post hoc, or run more seeds. This single adapted development run does **not** show that FedQCL itself is ineffective; it only shows that this adaptation/configuration did not provide an active FedQCL-DPP comparator.
