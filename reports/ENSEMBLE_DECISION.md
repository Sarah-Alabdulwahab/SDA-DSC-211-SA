# قرار التجميع

KEEP SINGLE — Logistic

Logistic Regression achieved the highest mean AP of 0.39166, with a fold SD of 0.02981, Brier score of 0.06327, and ECE of 0.01882. Equal averaging, weighted averaging, and stacking achieved AP scores of 0.37170, 0.38942, and 0.38314, respectively. None satisfied the required AP improvement greater than the reference fold SD. The worth-it gate also limited Brier and ECE deterioration to 0.005 and 0.01, respectively. Consequently, Logistic Regression was retained because additional ensemble complexity was not justified. This gate is a selection heuristic, not a statistical significance test.

Nested forward OOF validation addresses leakage risks through chronological splits, 90-day label maturity, customer separation, and inner splits for ensemble fitting. The experiment generated 2,155 OOF predictions across three validation periods, covering 32.77% of the fit-and-selection pool and 21.55% of the complete dataset. Limited historical coverage and reuse of OOF evidence for model and threshold selection mean that these results remain development evidence rather than an independent final-test estimate or guarantee of future generalization.

الدليل: artifacts/ensemble_comparison.csv وday5_ensemble_gate.json. SD وصفي، وليس اختبار دلالة.
