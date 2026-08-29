# Frozen source provenance

The four files in `frozen_source/` are byte-for-byte source snapshots from
the runs that produced the v16 matched-baseline JSON artifacts. Their hashes
are recorded inside those JSON files.

They are retained for provenance and should not be used as the public command
entry points: their original relative-path and stdout-output assumptions are
not portable after repository extraction. The corresponding files under
`experiments/` contain path/output-only integration changes and are invoked by
`scripts/reproduce.py`.

Run `python scripts/verify_provenance.py` to verify every runner and source
hash recorded by the two frozen v16 artifacts.
