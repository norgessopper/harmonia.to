# HHM Dataset Release — README

**Dataset ID:** `hhm-<domain>-<name>-<version>`  
**Title:** <Human-readable dataset title>  
**Version:** <vX.Y.Z or YYYY-MM-DD>  
**Release Date:** <ISO 8601 date>  
**Maintainer:** <Name / Organization / Contact Email>

---

## 📦 Contents
This release contains:

- `dataset_sheet.json` — Metadata, collection details, and processing summary.
- `manifest.json` — File list with sizes, hashes, and content addresses.
- `provenance.json` — Full lineage graph of all transformations.
- `README.md` — This document.
- `CHECKSUMS.txt` — BLAKE3 and SHA-256 hashes for all above files.
- (Optional) `/data/` — Public subset of the dataset.

---

## 🔍 Verification
To verify file integrity:

```bash
# Using BLAKE3
b3sum -c CHECKSUMS.txt

# Using SHA-256
sha256sum -c CHECKSUMS.txt