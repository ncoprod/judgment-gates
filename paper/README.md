# Paper build and evidence

The [paper PDF](main.pdf), [TeX source](main.tex), [bibliography](refs.bib), and
precomputed tables/figures in `generated/` are included in the public export.
From the repository root, install [Tectonic](https://tectonic-typesetting.github.io/)
and run:

```sh
uv sync --frozen
uv run pytest -q
tectonic -X compile paper/main.tex
```

This compiles the supplied artifacts and runs synthetic code tests. It does not
reconstruct the underlying experiment. The original generator in the private
research archive reads committed result receipts containing excluded observations
and provider responses; it is not part of the public export. Tables and figures
are generated there, never typed by hand.

Published evidence includes [calibration, latency, replay and verification totals](../results/research-summary.json),
[framing shifts and descriptive CIs](../results/framing-summary.json), and
[forward activity and P&L decomposition](../results/forward-diagnostics.json).
The public [export manifest](../EXPORT.json) pins its private source commit and
file hashes. `generated/manifest.json` pins the public aggregate inputs, table/figure
hashes and the hash of the original private generation manifest. In the private
checkout, that original manifest instead records the full original input identities.
Neither version implies that the public aggregates suffice to regenerate every
table or the raw experiment.

Reference DOI/arXiv identifiers are in the bibliography; private resolver receipts
retain the verification evidence. No unverified reference was added for publication.
Tectonic 0.15 can emit the known
[bibliography rerun warning](https://github.com/tectonic-typesetting/tectonic/discussions/1186).
Render the final PDF with `pdftoppm` and inspect every page after layout changes.

Raw exchange data, provider transcripts, per-call timing tables and private history
remain excluded under the [publication boundary](../RELEASE_POLICY.md). This is
a research draft, not an arXiv submission; prospective results remain pending.
