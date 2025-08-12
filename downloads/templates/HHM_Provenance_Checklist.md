# HHM Provenance Checklist

**Purpose:**  
Ensure every dataset used in the Holographic Harmonic Model (HHM) has a complete, auditable provenance chain — from raw collection to analysis-ready form.

---

## 1. Dataset Identification
- [ ] **Dataset ID** matches `dataset_sheet.json`
- [ ] Version tag (SemVer or date)
- [ ] Domain and modality recorded

## 2. Raw Data Origin
- [ ] Original source(s) documented
- [ ] Collection date range
- [ ] Devices/instruments used
- [ ] Collection protocols described
- [ ] Consent/PII status defined

## 3. Preprocessing Steps
- [ ] Each transformation step logged
- [ ] Parameters recorded (filter ranges, thresholds, etc.)
- [ ] Random seeds noted for deterministic steps
- [ ] Tool and script versions recorded
- [ ] Any manual interventions explained

## 4. Transform Records
For each transformation:
- [ ] **Input file(s)** and hash(es)
- [ ] **Output file(s)** and hash(es)
- [ ] Tool/script name + version
- [ ] Parameters and config
- [ ] Date/time started and finished
- [ ] Container or environment digest
- [ ] Notes (optional)

## 5. Environment Documentation
- [ ] OS and version
- [ ] Language/runtime (Python, R, etc.) and versions
- [ ] Libraries/packages and versions
- [ ] Hardware details (CPU, GPU, RAM)

## 6. Provenance Graph
- [ ] Nodes for every file/state with hash IDs
- [ ] Directed edges for every transform (input→output)
- [ ] Graph saved in `provenance.json` and optional `.graphml`

## 7. Integrity & Hashes
- [ ] BLAKE3 and SHA-256 for all files
- [ ] CHECKSUMS.txt at top-level
- [ ] Hashes for `dataset_sheet.json` and `manifest.json`

## 8. Cross-Links
- [ ] `dataset_sheet.json` → `provenance_ref`
- [ ] `manifest.json` hashes match provenance nodes
- [ ] Threshold profile linked

---

**Tip:** Automate this with the AI prompt:  
> "Generate provenance.json and CHECKSUMS.txt from the provided dataset folder, including hashes, transforms, and environment details following HHM schema."