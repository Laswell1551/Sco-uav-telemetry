# Contributing

Please open an issue before changing a frozen protocol, seed set, statistical
aggregation, or paper-facing number.

For code changes:

1. create a focused branch;
2. run `python scripts/reproduce.py verify`;
3. document any evidence-boundary change;
4. never commit third-party raw data, model archives, local absolute paths, or
   regenerated results without provenance and a matching manifest update.

Bug fixes that change a frozen result must retain the old artifact, explain
the cause, regenerate all affected figures/tables, and update the release
notes.
