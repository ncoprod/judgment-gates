# Publication boundary (2026-11-01)

The owner authorises publication of original code, synthetic tests, dependency
pins, the paper (TeX, precomputed tables/figures and PDF), and author-computed
aggregate measurements. The public repository is
[ncoprod/judgment-gates](https://github.com/ncoprod/judgment-gates).
The separate research archive, ncoprod/judgment-gates-private, and its entire
history remain private. Neither repository's visibility is changed.

`scripts/export_public.py` has an explicit file/directory allowlist and projects
named aggregate fields into five result files. It reads committed bytes, refuses
an existing target, checks repository-local document links, and emits source/file
hashes in `EXPORT.json`. Export into a fresh directory, inspect and scan it, then
add one normal commit to the existing public history. No private Git objects are
copied. Deleting material at HEAD would not sanitise private history.

## Included and excluded

Included measurements: calibration/reliability bins, replay totals and markout
intervals, latency quantiles and deadline counts, verification counts, forward
activity/P&L decomposition, framing shifts and their descriptive bootstrap CIs,
and upstream licence metadata. These are our computed statistics, not verbatim
provider responses or individual market observations. Failed routes and null CIs
remain visible; public results do not establish profitable trading or invariance.

Excluded: raw books/trades, row-level exchange-derived states/features/labels,
individual orders/fills/equity, trained policies, all raw API/CLI transcripts
(`*-calls.json`), reasoning, caches, per-call timing tables, local paths, secrets,
personal metadata and private history. Public aggregate files cannot reconstruct
historical replay or rescore individual provider calls. No model weights or
third-party source are redistributed.

## Official terms and rationale

- [Binance terms](https://www.binance.com/en/terms) and
  [Hyperliquid terms](https://app.hyperliquid.xyz/terms): public-stream access is
  not a redistribution licence. These dynamic pages did not expose a reviewed
  grant to redistribute underlying data. Retain those records privately; publish
  only the study's aggregate statistics.
- [NVIDIA API Trial terms](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf),
  sections 1.2/1.4, 4.2 and 6.3: trial/usage and distribution restrictions apply
  separately from conditional ownership. Publish no generated response content,
  reasoning or transcript. A model-weight licence does not confer hosted-service
  output rights.
- [OpenCode terms](https://opencode.ai/legal/terms-of-service): output ownership
  does not remove usage restrictions, including programmatic extraction. No new
  automated OpenCode research calls are authorised; historical caches stay private.
- [Vercel terms](https://vercel.com/legal/terms) and
  [TypeSafe legal documents](https://docs.typesafe.ai/legal) govern Jev separately.
  Its published preflight summary reports access status and timing only; there
  was no successful Jev inference in this study.

This is a scoped publication decision for author-computed statistics, not a
blanket redistribution grant or a legal opinion. The MIT code licence does not
relicense market data, model weights or hosted outputs. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The paper is a research draft,
not an arXiv submission; prospective results remain pending.
