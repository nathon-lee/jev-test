# Official held-out rerun

This work combines the explicit source partitions from PR #19 with the corrected,
checkpointed Noul runner from PR #20. Results are separate from the historical-row
parity benchmarks. Official partition membership alone does not establish a model
has never seen these rows: clean training exports and overlap audits remain separate
work in ENG-10199.

Current classification scope: HateCheck test (3,728), AEGIS2 test (1,964), and
Jailbreak Classification test (262), both v1/v2, primary threshold 0.5. Each provider
makes 11,908 requests. ToxiGen is excluded by request: its official test has 940 rows but none match
the legacy generated prompt_label lookup. No replacement label rule is assumed.
Detect Jailbreak and JailbreakHub remain excluded pending the joint held-out split.

PII is a separate extraction task: Nano validation (90) and Mini validation (990),
using `fastino/gliner2.5-multi-v1`. These are not GLiDE results. Reports distinguish
typed span F1, boundary F1, exact document matches and request failures. The fixed
entity schema is recorded; unsupported types and boundaries count as misses.

## Credentials

Use the ignored local `.env` file. GLiDE/GLiNER need `FASTINO_API_KEY`, Laya needs
`LAYA_API_KEY`, and TypeSafe/Jev needs `TYPESAFE_API_KEY`. Optional base URLs use
`FASTINO_BASE_URL`, `LAYA_BASE_URL`, `TYPESAFE_BASE_URL`. Do not commit keys.
The repo registry identifies these additional services without needing a new provider list:

| Provider | Repo model | Required local configuration | Readiness |
|---|---|---|---|
| TypeSafe/Jev | jev-latest | TYPESAFE_API_KEY | Hosted API; checkpointed held-out runner ready |
| OpenJev | openjev | OPENJEV_API_KEY, OPENJEV_BASE_URL | Local/self-hosted; repo default is localhost:8000 |
| Galileo | galileo-standard | GALILEO_API_KEY, GALILEO_BASE_URL | Repo contains an example.com placeholder, not a real deployment |
| Microsoft Decision-1 | deployment alias from MICROSOFT_DECISION_1_MODEL | MICROSOFT_DECISION_1_API_KEY, MICROSOFT_DECISION_1_BASE_URL, MICROSOFT_DECISION_1_MODEL | Foundry decision API configured; strict checkpointed runner ready |

Fastino, Laya, Jev and Microsoft are configured locally. OpenJev/Galileo have generic
chat-completion adapters in the legacy suite, but need reachable deployments and a
validated numeric score contract before inclusion in this strict checkpointed run.
Keys alone cannot identify a private deployment. The local .env has every registry
key/endpoint field, with existing values preserved and no invented internal URLs.

The registry was checked on the active `feat/summary_testdata` branch as well as
`type_safe_http`; the default `main` branch has no provider_suite.py. The older HTTP
branch uses typesafe-standard; the active branch uses jev-latest. Dynamo CPU/A100/L4,
Mistral and Warden occur only in supplied aggregate references: the repo provides
no runnable adapter/deployment contract for those configurations. OpenJev 4B/32B
reference labels also do not identify a deployed endpoint/model alias.

## Reproduce or resume

```bash
python -m scripts.prepare_datasets \
  --datasets hatecheck,aegis2,jailbreak-classification \
  --output-dir data/heldout_2026-10-28
python -m scripts.rerun_classification --provider fastino.ai \
  --datasets jailbreak-classification,hatecheck,aegis2 \
  --data-dir data/heldout_2026-10-28 --require-heldout \
  --output-dir outputs/heldout_2026-10-28/classification --env-file .env --workers 32
```

Use the same frozen prepared files for every provider. For Laya use
`--provider laya-ai.com --workers 16`; for Jev use `--provider typesafe.ai`.
The TypeSafe REST path/body follows the locally installed SDK's System One contract.
Scores are strictly `answers.score.noul`, not confidence. Source hashes and provenance
are checked before any requests. CSV outputs retain source row/revision/split identity.
Thresholds 0.5–0.8 are derived offline and are not independent repeat inferences.
Errors remain errors; resumption retries failed rows and reuses successful checkpoints.

```bash
python -m scripts.pii_extraction prepare --dataset ai4privacy/pii-masking-nano-1k \
  --output data/heldout_2026-10-28/pii_nano.jsonl
python -m scripts.rerun_pii_extraction --input data/heldout_2026-10-28/pii_nano.jsonl \
  --output outputs/heldout_2026-10-28/pii_extraction/nano.jsonl --env-file .env
python -m scripts.pii_extraction prepare --dataset ai4privacy/pii-masking-mini-10k \
  --output data/heldout_2026-10-28/pii_mini.jsonl
python -m scripts.rerun_pii_extraction --input data/heldout_2026-10-28/pii_mini.jsonl \
  --output outputs/heldout_2026-10-28/pii_extraction/mini.jsonl --env-file .env
python -m scripts.report_heldout
```

Rerun reporting as provider suites complete. A provider must finish all declared
configurations before inclusion. No historical predictions or supplied aggregate
reference scores enter this report. Cross-provider ranking additionally requires
matching successful input rows; per-provider error rates and coverage are retained.

## Access follow-up

Galileo is excluded from this rerun: its placeholder entry does not identify a
deployed classifier, and an observability/evaluation account is not required for
this harness. OpenJev's published hosted site and API landing page both currently
show "This deployment is temporarily paused"; signup could not be completed.
The repo's localhost entry alone does not identify the hosted model or establish
equivalence with the old 4B/32B aggregate references.

Microsoft Decision-1 uses the actual Foundry decision contract, not the legacy
generic chat adapter: `api-key` authentication, deployment name in `model`, and
`<resource>/providers/microsoft/v1/systemone`. Existing credentials are kept only
in the ignored .env. A test deployment was created in an existing non-production
resource, model version 1 / GlobalStandard. Initial live requests succeeded.

```bash
python -m scripts.rerun_classification --provider microsoft-decision-1 \
  --datasets jailbreak-classification,hatecheck,aegis2 \
  --data-dir data/heldout_2026-10-28 --require-heldout \
  --output-dir outputs/heldout_2026-10-28/classification --env-file .env \
  --workers 4 --requests-per-second 4
```

The shared cap applies to retries as well as initial requests, and is recorded in
the manifests. Timing includes rate-limit waits and must not be interpreted as
unconstrained model throughput. Foundry deployment instructions:
https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision

## HTML and figure publication

`python -m scripts.report_heldout` now rebuilds the Markdown/JSON report, HTML
comparison, gallery, standalone three-dataset page, machine-readable metric CSVs,
and eight PNG/SVG/PDF figures with a combined PDF. No hosted inference calls are
made. The report checks prediction sample IDs/text/labels against the pinned source
files and requires identical successful rows across complete providers before
publishing comparable quality charts. Metrics are recomputed from saved scores.

The v1 summary retains the familiar `fig13_three_dataset_averages.png` filename in
this **held-out** directory. Its scope is HateCheck, AEGIS2 and Jailbreak
Classification, equally weighted; it is not the ToxiGen/HateCheck/AEGIS2 scope in
PR #20. v2 is a separate chart. Timing includes retries/pacing and is shown
separately with worker counts and rate caps, not as a deployment-speed ranking.
PII extraction remains a separate GLiNER chart. Each figure's files and SHA-256
hashes are recorded in `figures/figure_manifest.json`, tied to the source summary.
