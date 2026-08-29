# Third-party data and software

No third-party raw dataset or external source-code repository is vendored in
this release candidate.

## UZH-FPV

- Official site: https://fpv.ifi.uzh.ch/
- Paper: J. Delmerico et al., ICRA 2019.
- Upstream license: CC BY-NC-SA 3.0.
- Local policy: download on demand; do not treat any UZH-derived artifact as
  MIT-covered data. See 'LICENSES/DATA-RESULTS.md' for the frozen-result map.

## M3ED

- Official site: https://m3ed.io/
- Paper: K. Chaney et al., CVPR Workshops 2023.
- Upstream license: CC BY-SA 4.0.
- Local policy: download on demand; preserve attribution, modification notice,
  and share-alike requirements for any redistributed derivative data. See
  'LICENSES/DATA-RESULTS.md' for the frozen-result map.

## 6GL-CLD26_v2

- Zenodo: https://doi.org/10.5281/zenodo.21240929
- Upstream project: https://github.com/frpaolucci/6gl-cld26-v2
- Version used: 2026 version 2.
- Local archive SHA-256:
  'a52c7a0fd2d90c72d97069a57e27df8de62e657135ec48262fd6d7f6558dcada'
- Upstream rights field: no explicit license was displayed at artifact
  preparation time.
- Local policy: preparation code and DOI only; no raw archive, extracted data,
  or aligned trace is redistributed without explicit permission.

## Python packages

Runtime dependencies are installed from PyPI/conda and remain under their
respective upstream licenses. Exact tested versions are in
'requirements/constraints-py312.txt'.
