# Third-party notices

Original code: MIT, Copyright (c) 2026 Nicolas Coquelet. No model weights or
third-party source are vendored. No code has been copied from MakeMarketBench.
The local recorder is a read-only source and verification tool, MIT by Nicolas
Coquelet; it is not a package dependency or modified here.

## Market data

Binance USDⓈ-M and Hyperliquid public L2/trades are exchange data, **not MIT**.
[Binance terms](https://www.binance.com/en/terms),
[Binance API rules](https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info),
[Hyperliquid terms](https://app.hyperliquid.xyz/terms) and
[Hyperliquid API rules](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits)
apply. No general redistribution licence is asserted. Raw recordings and exact
local snapshot copies stay outside git. The public export contains author-computed
aggregates only; row-level derived data and the research archive's history remain
private. See [RELEASE_POLICY.md](RELEASE_POLICY.md).
The download/record procedure is the sibling recorder's public-stream CLI, not
an invented static historical download endpoint.

## Models and services

TypeSafe Jev is a proprietary hosted service, governed by
[TypeSafe legal documents](https://docs.typesafe.ai/legal). No weight licence is
claimed. The client uses the [Vercel Decision API](https://vercel.com/docs/ai-gateway/modalities/decision)
with a typed boolean veto and its probability. Historical TypeSafe-compatible
receipts remain access evidence only. Neither vendor latency nor calibration
claims are empirical results here. The gateway remains
subject to [Vercel terms](https://vercel.com/legal/terms).
The catalogue's evaluation model `typesafe-ai/jev` supports specification v4;
its proprietary service has no asserted open-weight licence.

NVIDIA NIM access is governed separately by
[NVIDIA API Trial terms](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf).
For hosted DeepSeek-V4.1-Flash, NVIDIA also applies the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)
([official model terms](https://docs.api.nvidia.com/nim/reference/nvidia-deepseek-v4_1-flash)).
Model weight licences do not supersede these service/model terms. API outputs and
caches remain private. The paper and author-computed aggregate measurements may
be exported under [RELEASE_POLICY.md](RELEASE_POLICY.md). Model-card licence
metadata, pinned revisions and upstream hashes are in
[results/licenses.json](results/licenses.json). No licence is inferred from a similar older model.

| Exact model card | Verified licence | Card revision | Required notice |
|---|---|---|---|
| zai-org/GLM-5.3-Flash | MIT | 46a050bec13002ad29ee795c16e61cf4fa3c4a9c | Preserve Z.ai copyright and MIT licence when redistributing weights/code |
| zai-org/GLM-5.3 | Custom GLM-5.3 License (`other` on model card) | ba32f8c19c8738e2c57c71798c601d477a7bf844 | Preserve Z.AI copyright and permission text; includes a security-review condition for very large model-as-a-service businesses |
| deepseek-ai/DeepSeek-V4.1-Flash | MIT | 314113f8a7dadb388215396ba5fbee2c610a04d6 | Preserve DeepSeek copyright and MIT licence |
| openai/gpt-oss-20b | Apache-2.0 | 6cee5e81ee83917806bbde320786a8fb61efebee | Retain licence, applicable notices and modification notices on redistribution |
| nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | OpenMDW-1.1 (`other` on card) | 77df655d5e9f8362164ed14dd8b48f8bce657498 | Retain the agreement and applicable copyright/origin notices when distributing model materials |

Additional framing-extension cards, verified in [results/tilt-model-licenses.json](results/tilt-model-licenses.json):

| Exact model card | Verified licence | Card revision | Required notice |
|---|---|---|---|
| MiniMaxAI/MiniMax-M3.1-Flash | MiniMax Community License (`other`; non-commercial default, conditional commercial use) | 262ae6830934f3512211081b8a3f16425b93a239 | Retain MiniMax copyright and licence; commercial use has attribution/notification or authorisation conditions |
| google/gemma-4-31B-it | Apache-2.0 (model-card SPDX) | 842da3794eaa0b77d5f08bae87a17459d91ff475 | Retain Apache licence and applicable notices |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | OpenMDW-1.1 (`other`) | a9904d24bcc1d289a1950fa9d2b978c47cf903b9 | Retain agreement and applicable copyright/origin notices |

Hosted aliases do not expose immutable deployment revisions. These card revisions
verify weight licences; they do not prove which weights served an API request.

OpenCode's free models are used only through the official CLI/server, subject to
[OpenCode terms](https://opencode.ai/legal/terms-of-service) and provider terms.
No free endpoint client impersonation or account creation is used.

## Runtime and tools

Resolved versions are pinned in `uv.lock`. Dependencies are installed, not vendored;
their full notices remain in their distributions. Upstream licence evidence is
recorded in `results/licenses.json`.

| Component | Licence | Attribution / upstream |
|---|---|---|
| CPython | PSF-2.0 and incorporated notices | Python Software Foundation; https://github.com/python/cpython/blob/3.12/LICENSE |
| NumPy | BSD-3-Clause | NumPy Developers; https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| SciPy | BSD-3-Clause and bundled notices | SciPy Developers; https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| scikit-learn | BSD-3-Clause | scikit-learn developers; https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| PyArrow | Apache-2.0 and bundled notices | Apache Software Foundation; https://github.com/apache/arrow/blob/main/LICENSE.txt and NOTICE.txt |
| HTTPX | BSD-3-Clause | Encode OSS Ltd; https://github.com/encode/httpx/blob/master/LICENSE.md |
| websockets | BSD-3-Clause | Aymeric Augustin and contributors; https://github.com/python-websockets/websockets/blob/main/LICENSE |
| Matplotlib | PSF-based Matplotlib licence and incorporated notices | Matplotlib Development Team; full upstream text in the licence receipt |
| uv | MIT OR Apache-2.0 | Astral; https://github.com/astral-sh/uv/tree/main |
| ruff | MIT | Astral; https://github.com/astral-sh/ruff/blob/main/LICENSE |
| pytest | MIT | pytest developers; https://github.com/pytest-dev/pytest/blob/main/LICENSE |
| hatchling | MIT | Ofek Lev; https://github.com/pypa/hatch/blob/master/LICENSE.txt |

BSD notices require preservation on redistribution; Apache distributions retain
LICENSE/NOTICE and change notices where applicable. Transitive dependencies retain
their installed licence files too; the code licence does not relicense them.

Paper build tool: Tectonic 0.15.0 (final compiler) and 0.17.0 (diagnostic) are MIT,
with additional notices for the TeX components from which they are derived; see
the [0.15.0 LICENSE](https://raw.githubusercontent.com/tectonic-typesetting/tectonic/tectonic@0.15.0/LICENSE)
and the diagnostic release's
[pinned upstream LICENSE](https://raw.githubusercontent.com/tectonic-typesetting/tectonic/tectonic@0.17.0/LICENSE).
The compiler archive/binary receipts remain in the private research archive. The binary is
installed only under ignored `runs/tools`, not vendored or exported. Installed
TeX packages/fonts retain their upstream licences; the code licence does not
relicense them. No third-party paper prose or figures are copied into the draft.
