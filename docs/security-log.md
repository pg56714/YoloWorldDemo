# Security Log

English | [繁體中文](security-log.zh-TW.md)

- Date: 2026-10-25
- Branch: `fix/security-upgrades`
- Scope: `requirements.txt`

## Summary

gradio and inference were upgraded to patched, installable releases. pillow, filelock and wheel still have open advisories because gradio and inference cap their versions.

## Changes

| Item | Before | After | Reason |
|---|---|---|---|
| gradio | 4.19.2 | 6.20.0 | File read, SSRF and path traversal advisories (fixed in 6.16) |
| inference[yolo-world] | 0.9.13 | 1.8.2 | 0.9.13 can no longer be installed: its inference-cli dependency was yanked from PyPI |

## What you need to know

- gradio is pinned to 6.20.0: newer 6.x releases conflict with the huggingface-hub version that inference 1.8.2 requires.
- inference 1.8.2 is a newer model package than 0.9.13; detection results for `yolo_world/l` may differ a little.

## Remaining issues

### Dependency versions held back upstream

| Package | Version | Advisories | Why it is not fixed | Impact and mitigation |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 (fixed in 13.0.0) | gradio >= 6.16 requires `pillow<13`; no gradio release allows Pillow 13 yet. | Crafted image files (for example TIFF or PDF) can crash the process or read out-of-bounds memory when decoded. Run the Gradio app locally only (no `share=True`, no public port) and open only images you trust. Upgrade once gradio allows Pillow 13. |
| filelock | 3.32.7 | 3 (fixed in 4.2.1) | inference 1.8.2 requires `filelock<=4.0.0`. | Lock-handling flaws in `SoftFileLock` on Windows and in the SQLite-based `ReadWriteLock`. Here it only guards the Hugging Face model cache, so the risk is low for single-user local use. |
| wheel | 0.45.1 | 2 (fixed in 0.46.2) | inference 1.8.2 requires `wheel<=0.45.1`. | Path traversal and file-permission changes when a crafted wheel is unpacked with `wheel unpack`. The app never unpacks wheels at runtime. |

## How it was verified

- osv-scanner on the resolved `requirements.txt`; only the packages listed above remain.
- The Gradio UI builds and serves with the model modules replaced by stubs.
- The annotation code runs with supervision 0.30.9.

## Not verified

- Inference on a GPU with the real model weights (the test environment has no GPU and no weights).
