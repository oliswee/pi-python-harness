# Evidence ledger

Use these claims with their measurement scope. The Python base is pinned in the [M1 audit](baseline-gap-audit.md).

| Claim | Verifier and evidence | Implementation revision |
| --- | --- | --- |
| A deterministic Pi agent reads and writes through Mini-Nanobot | [M1 E2E cases](../tests/e2e/test_baseline_agent.py) and [baseline audit](baseline-gap-audit.md) | `1ed7f56` |
| Four process-crash windows verified, with uncertain writes held | [Fault worker](../tests/e2e/durable_run_worker.py), [recovery tests](../tests/e2e/test_durable_run.py), and B03/B04 in the [case report](../evidence/m6/report.json) | `44d9d51` |
| Controlled reads shortened by 67.3% in the paired median | Eight pairs of 34 seeded reads with a 20 ms simulated delay. Native warm-cache median shortened 11.7% and p95 rose. [Raw CSVs and protocol](tool-policy.md#reproduce-the-read-benchmark) | `57ad0fd` |
| Eight post-revocation requests excluded the revoked revision | Eight fresh **scripted** provider requests. B08 repeats the checks against a reopened SQLite store. [M4 verifier](../tests/e2e/test_memory_lifecycle.py) | `6cf25fe` |
| Eight live DeepSeek requests excluded the revoked revision | All eight `deepseek-flash` responses returned the retained canary, and the revoked canary was absent from every fresh context and response. [Redacted live report](../evidence/m4/live-revocation-deepseek.json) | `54b755d` |
| One verified request shrank from 13,738 to 2,268 estimated units including reserve | Seeded facts and follow-up read verified with the default byte estimator. [M5 trace extract](../evidence/m5/context-budget.jsonl) | `e19e021` |
| Twelve scripted cases passed three independent trials each | [Case manifest](../benchmarks/cases.json), [36 case outcomes](../evidence/m6/report.json), and [trial artifacts](../evidence/m6/trial-artifacts.zip) | `9a8112d` |
| Four runtime feature controls showed their expected ON and OFF behavior | Three pairs per feature. [24 observations](../evidence/m6/ablations.json) and [protocol](eval-protocol.md) | `2b9b40f` |
| Fresh Windows checkout passed locked setup, core tests, and evaluation smoke | [Verification record](../evidence/m7/verification.json) and [CI workflow](../.github/workflows/mini-nanobot.yml) | `9636826` |

Distribution readiness remains pending confirmation of the Python port license. [Release readiness](release-readiness.md)
