# HHM Reproducible Replay (R1)

## Files
- `HHM_Pipeline_Template.yaml` — pipeline config
- `HHM_Env_Lock_Template.json` — frozen env
- `dataset.manifest.json` — dataset & checksums
- `HHM_Thresholds.v2.json` — thresholds (optional, external)
- `HHM_Nulls.v2.json` — null/boot config (optional)
- `hhm.result_card.v2.json` — schema for `result_card.json`
- `hhm.ci.schema.v2.json` — schema for `ci.json`
- `ops_metrics.schema.json` — metrics table schema
- `audit_report.template.html` — audit HTML skeleton

## Minimal replay
```bash
hhm run --pipeline HHM_Pipeline_Template.yaml \
        --env-lock HHM_Env_Lock_Template.json \
        --dataset-manifest dataset.manifest.json \
        --thresholds HHM_Thresholds.v2.json \
        --nulls HHM_Nulls.v2.json