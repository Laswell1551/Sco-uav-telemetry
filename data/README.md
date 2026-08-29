# External data

Third-party raw data are intentionally excluded from Git. Download each
dataset from its official source and place only the required pose/ground-truth
subset under this directory.

## Expected layout

~~~text
data/
  uzh_fpv_gt/
    indoor_45_12/leica.txt
    ...
    outdoor_forward_5/leica.txt
  m3ed_falcon_pose/
    falcon_forest_into_forest_1_pose_gt.h5
    ...
    falcon_outdoor_night_penno_parking_2_pose_gt.h5
  6gl_cld26_v2/
    extracted/6gl-cld26-v2-main/
~~~

'data/external-data-manifest.json' records the exact local filenames, byte
counts, and SHA-256 values used for the frozen trace results.

Regenerate it without recording absolute local paths:

~~~bash
python scripts/build_external_manifest.py \
  --uzh-root /path/to/uzh_fpv_gt \
  --m3ed-root /path/to/m3ed_falcon_pose \
  --sixgl-archive /path/to/6gl-cld26-v2.zip
~~~

## UZH-FPV

- Official page: https://fpv.ifi.uzh.ch/
- Required subset: the 16 Leica ground-truth trajectories listed in the
  manifest.
- Upstream license: CC BY-NC-SA 3.0.
- Runner:

~~~bash
python experiments/run_uzh_trace_replay.py \
  --data-root data/uzh_fpv_gt \
  --output runs/traces/uzh_trace_replay_v1.json
~~~

Do not redistribute this dataset as MIT content. Preserve attribution,
non-commercial, and share-alike conditions.

## M3ED Falcon

- Official page and downloader: https://m3ed.io/download/
- Required subset: 19 Falcon 'pose_gt.h5' files listed in the manifest.
- Upstream license: CC BY-SA 4.0.
- Runner:

~~~bash
python experiments/run_m3ed_trace_replay.py \
  --data-root data/m3ed_falcon_pose \
  --output runs/traces/m3ed_trace_replay_v1.json
~~~

Preserve attribution, modification notice, and share-alike terms for any
redistributed derivative data.

## 6GL-CLD26_v2

- DOI: https://doi.org/10.5281/zenodo.21240929
- Upstream project: https://github.com/frpaolucci/6gl-cld26-v2
- Local archive SHA-256:
  'a52c7a0fd2d90c72d97069a57e27df8de62e657135ec48262fd6d7f6558dcada'

The upstream Zenodo rights field did not display an explicit license during
artifact preparation. Therefore this repository does not redistribute the
archive, extracted files, aligned trace, or any derived 6GL result file.

After obtaining the data from the upstream source:

~~~bash
python experiments/prepare_tmc_6gl_trace.py \
  --root data/6gl_cld26_v2/extracted/6gl-cld26-v2-main \
  --csv runs/6gl/tmc_6gl_aligned_trace.csv \
  --summary runs/6gl/tmc_6gl_aligned_trace_summary.json
~~~

Run the loss replay from a directory whose 'results/' folder contains the
aligned trace. The 6GL evidence is retrospective and descriptive; the source
does not provide measured end-to-end RTT or separate forward/feedback delay.
