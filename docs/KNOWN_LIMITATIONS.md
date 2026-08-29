# Known limitations

1. The original 30 controlled JSON shards were not archived. The frozen
   aggregate is available, and the public runner can regenerate the shards.
2. UZH-FPV and M3ED raw/pose data are external and license-constrained.
3. The 6GL Zenodo record had no explicit displayed license at preparation
   time; its aligned trace and all derived result files are intentionally
   excluded from the public artifact.
4. Runtime values depend on CPU, BLAS, OS scheduling, and background load.
5. Several recent baselines are matched adaptations, not official source-code
   reproductions.
6. The broad historical project suite contained superseded tests and
   real-trace preprocessing thresholds outside this curated artifact. Public
   CI runs the targeted SCO suite; external-data checks are a separate gate.
7. The manuscript and IEEE template are not part of this repository release
   candidate.
