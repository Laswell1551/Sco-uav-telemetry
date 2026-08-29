# Public release checklist

## Blocking before remote publication

- [x] Prepare an identifiable local release candidate with the three paper
      authors; no remote publication is performed by this workflow.
- [x] Replace the collective author placeholder in `LICENSE`, `CITATION.cff`,
      and package metadata.
- [x] Confirm the journal's current code/preprint policy before making the
      identifiable repository public.
- [x] Add the final Git hosting URL to `CITATION.cff`.
- [ ] Add the archival DOI to `CITATION.cff` after deposit.
- [ ] Obtain explicit redistribution guidance before adding any 6GL aligned
      trace or derived dataset file.
- [x] Add a per-file attribution and CC terms map for UZH/M3ED-derived
      frozen outputs in `LICENSES/DATA-RESULTS.md`.

## Required technical gate

- [x] `python scripts/reproduce.py verify` passes for rc3 after adding the
      crossed capacity--delay experiment.
- [x] `python scripts/build_tables.py` completes for rc3.
- [x] No file exceeds 95 MiB.
- [x] No absolute personal path, token, email, or private key is present.
- [x] `results/frozen/MANIFEST.sha256` matches every rc3 frozen artifact.
- [x] Git dry-run contains no caches, raw data, logs, pickle, or model archive.
- [x] CI passes on Python 3.10 and 3.12, including figure/table rendering.

## Release mechanics

- [x] Prepare this allowlisted directory as a standalone Git-ready tree with no
      inherited history or remote configuration.
- [x] Initialize the publication repository with default branch `main`.
- [x] Create the remote repository without importing the manuscript workspace.
- [x] Add a short repository description and topic tags.
- [x] Push this allowlisted directory only.
- [x] Tag the validated release candidate as `v0.1.0-rc3`.
- [ ] After author and publication sign-off, tag `v1.0.0`.
- [ ] Archive the release in Zenodo and record the DOI.
