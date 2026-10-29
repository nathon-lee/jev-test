# Evaluation partition policy

Dataset families are not interchangeable evaluation sets. The suite previously
combined ToxiGen annotated train and test, used train for the jailbreak loaders
and PII, and used test for HateCheck/AEGIS. The policy now selects **test first,
then validation/valid/dev**, from the dataset's declared splits at a committed
Hugging Face commit SHA. It never concatenates partitions or accepts split slices
or unions as an implicit evaluation set.

| Dataset | Default evaluation partition | Change from historical preparation |
|---|---|---|
| ToxiGen (`annotated`) | test | Stops combining annotated train and test |
| HateCheck | test | Same partition, now pinned and recorded |
| AEGIS 2.0 | test | Same partition, now pinned and recorded |
| Jailbreak Classification | test | Replaces the implicit train default |
| PII Nano / Mini | validation | Replaces train |
| Detect Jailbreak | none: source exposes only train | Explicit reference-only opt-in required |
| JailbreakHub | none: source exposes only train | Explicit reference-only opt-in required |

Availability is inspected at the selected revision rather than inferred forever
from this table. If a source gains an official held-out partition, it can be
selected normally. No arbitrary local holdout is silently manufactured, and
"held-out" here describes the upstream partition, not proof that a pretrained
model has never encountered its data.

## Prepare held-out benchmark data

From the repository root:

```bash
python -m scripts.prepare_datasets \
  --datasets toxigen,hatecheck,aegis2,jailbreak-classification,pii_masking_nano_1k,pii_masking_mini_10k \
  --output-dir data/evaluation
```

The default source SHAs are committed in `data/source-revisions.json` and shared
by both preparation commands. Upstream changes to `main` do not change default
benchmark membership. `--revision SHA` explicitly overrides that pin; such a run
is a different population and its manifest records the resolved revision. `--split validation` is an explicit override for selected
datasets that actually expose that partition. Missing requested partitions fail;
the loader does not fall back to train. Preparing all datasets without an opt-in
will fail on train-only sources. All selected frames must prepare successfully
before any existing output CSV is replaced.

Run providers against the new population in a separate output directory:

```bash
python -m scripts.run_model_suite --providers fastino.ai,typesafe.ai \
  --datasets toxigen,hatecheck,aegis2,jailbreak-classification \
  --data-dir data/evaluation --output-dir outputs/held_out_evaluation
```

Both directory options preserve their historical defaults when omitted. Use fresh
directories when migrating so an active historical-parity run remains unchanged.

For a diagnostic run on a train-only source, opt in explicitly and keep it separate:

```bash
python -m scripts.prepare_datasets --datasets detect-jailbreak,jailbreakhub \
  --allow-train-eval --output-dir data/training_reference
```

These rows are labelled `partition_role=training_reference`, not held-out test
data. An explicit `--split train` also requires `--allow-train-eval`, even when a
test split is available. Do not use training-reference scores for model selection
or present them as held-out benchmark scores.

## Provenance and comparisons

Prepared rows contain `source_repository`, `source_config`, `source_revision`,
`source_split`, `partition_role`, and `source_row`. A `.manifest.json` beside each
CSV records available splits, upstream and prepared row counts, and the CSV
SHA-256. Source row identity remains available when filtering changes the suite's
sequential sample IDs. Provider output rows retain this provenance on success and
failure.

Generate comparisons using an explicit dataset selection so train-only sources
need not exist in the held-out data directory:

```bash
python -m scripts.compare_models --datasets toxigen,hatecheck,aegis2,jailbreak-classification \
  --data-dir data/evaluation --input-dir outputs/held_out_evaluation --output-dir outputs/held_out_comparison
```

The comparison reader rejects explicitly marked training-reference data. Dataset
selection does not reuse historical aggregate reports, because their evaluation
population may differ. Existing legacy files without provenance remain readable
for historical reproduction; that compatibility is not certification of their
split or held-out status. A single comparison cannot mix newly partitioned files
with unversioned legacy files; select a consistent dataset population explicitly.

Regenerate predictions for **every provider** after changing partitions. Historical
predictions must not be paired with the new data: row membership, IDs, counts,
and hashes can differ. Existing text/label/ID checks reject incompatible pairs.
This PR does not replace checked-in datasets, regenerate results, or change the
input rows of an already-running legacy-parity rerun.

## Labels and PII

This change isolates partition selection from label semantics. ToxiGen still uses
the existing generated `prompt_label` lookup; it does not switch silently to a
threshold on `toxicity_human`. The generated train-config data is loaded only as
a label lookup at the same revision and is identified as `label_lookup_only` in
the manifest. Only rows from the selected annotated split become evaluation
examples; unmatched/conflicting lookup rows remain excluded as before. The
manifest distinguishes upstream rows from the final filtered population.

PII classification labels retain their existing detection behavior. Use
`python -m scripts.pii_extraction prepare` for the dedicated span/entity workflow,
which already defaults to validation and pins its source revision. Neither path
should be compared to historical train-partition scores or to a separately sampled
1,000-row evaluation set without matching the exact rows and metric definition.

Offline regression coverage: `python -m pytest tests/test_dataset_partitions.py -q`.

## Review scope and remaining work (ENG-10199)

This PR is the evaluation-partition/provenance prerequisite, not completion of
ENG-10199. It targets `dev_autotest`, the integration branch already used for
PRs #17 and #18; merging this work into another branch is a separate repo-owner
decision. Historical comparisons and prepared files are not rewritten.

AEGIS removes `REDACTED` prompts and merges normalized duplicate prompts only
when their labels agree; all rows in a conflicting-label group are quarantined,
never resolved by choosing a label. The manifest records the conflicting normalized
text hashes as well as counts. The manifest
records exclusion counts and retained rows preserve original source-row IDs.

Follow-up PRs own the frozen joint jailbreak 80/20 membership and conflict
ledger, disjoint training exports and exact/3-gram Jaccard leakage checks,
clean Jailbreak Classification reporting, and a Fastino fine-tune provider.
HateCheck is final-report-only: never use its rows, templates or close
paraphrases for training, threshold tuning, selection or stopping. A future
template-based selection partition requires explicit agreement and separate
frozen membership.

ToxiGen remains blocked pending an agreed label rule; the command refuses
the empty legacy generated-label mapping. The proposed published
`toxicity_ai + toxicity_human > 5.5` rule is not silently enabled. Henry's
joint jailbreak split is a reuse candidate, not an agreed official HSBC split.
