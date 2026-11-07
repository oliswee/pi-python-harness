# Release readiness

## Security and operating boundaries

- The Mini-Nanobot adapter registers workspace read and write. Its path policy resolves targets, rejects protected paths and Windows device names, and requires state and memory storage outside the tool workspace. The workspace and local filesystem are trusted. For untrusted workloads, use OS-level isolation and a separate working directory with minimal permissions.
- The inherited Pi CLI has its own tools, including shell execution. Use the Mini-Nanobot entry point for the controls described here. Treat prompts, tool output, recalled text, and summaries as untrusted data.
- Keep provider credentials in the process environment or an ignored local credential store. The offline CI and evaluation fixtures use injected scripted providers. The opt-in Gemini verifier sends only seeded canaries and saves response flags and reported usage.
- Diagnostic traces use an allowlist of metadata. The journal, checkpoint, and SQLite memory can retain raw text, tool arguments, and responses. Use private storage, scope access at the caller boundary, and apply a retention policy. Revocation governs fresh context assembly and historical replay.
- Durable execution was tested against process termination on a trusted local filesystem. An uncertain write needs operator inspection. Whole-request token figures are estimates at the Pi provider boundary.

## Attribution and dependency inventory

The Python base is [openxjarvis/pi-mono-python at `5e41e26`](https://github.com/openxjarvis/pi-mono-python/tree/5e41e26552d2e8ed29b7934f126bae0ee8031761), accessed through [oliswee/pi-python-harness](https://github.com/oliswee/pi-python-harness). The pinned Chinese README declares MIT and references a `LICENSE` file. Both the pinned tree and the referenced upstream URL lack that file. Confirm the Python port's full licensing and attribution with its maintainer before packaging or distributing the derivative. [Pinned upstream README](https://github.com/openxjarvis/pi-mono-python/blob/5e41e26552d2e8ed29b7934f126bae0ee8031761/README_CN.md)

The related TypeScript Pi project carries Mario Zechner's MIT notice. The verbatim reference notice is preserved in [LICENSES/pi-typescript-MIT.txt](../LICENSES/pi-typescript-MIT.txt). [Upstream TypeScript license](https://github.com/earendil-works/pi/blob/085c84e98e7f4c950f2be606466fff4f75d8fe7c/LICENSE)

The [runtime component inventory](../evidence/m7/runtime-components.json) lists 69 CycloneDX components exported from the locked runtime workspace. Development dependencies remain in `uv.lock`. Generate the full dependency graph and distribution hashes with the pinned uv version.

```powershell
uv export --locked --all-packages --no-dev --format cyclonedx1.5 --output-file runtime-sbom.cdx.json
```

## Verification gates

- The workflow installs from the lock in uv copy mode and runs Pi CLI help, the core E2E suite, and an evaluation-runner smoke test on Windows and Ubuntu. A fresh Windows clone passed locked installation, CLI help, all 73 focused tests, and both evaluation smoke cases. An earlier installation attempt hit a Windows cloud-filesystem hardlink error, which copy mode avoided. Hosted status becomes available when the workflow runs on GitHub. [Windows verification record](../evidence/m7/verification.json)
- The broad inherited offline suite had 651 passes and the same 18 known upstream failures documented in the [M1 audit](baseline-gap-audit.md). The CI job selects the Mini-Nanobot and Pi Agent suites.
- M4's eight offline request checks are reproducible. Eight successful live probes require an authorized Gemini credential and a usable model from the pinned Pi registry.
- Distribution depends on resolving the Python port licensing gap. The [evidence ledger](evidence-ledger.md) distinguishes offline claims, measurements, and outstanding gates.
