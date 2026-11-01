# Model checkpoints

Symmetrix-XL keeps upstream model checkpoints separate from the source tree. A
checkpoint is the original learned model; the runtime normally consumes a
compact JSON export generated from it. Downloaded files, compact exports, JIT
artifacts, and qualification records belong in a private cache such as
`~/.cache/symmetrix/checkpoints` or `benchmarks/.artifacts/`, never in Git.

The qualification manifest is the canonical source for the standard foundation
models: [`benchmarks/foundation_model_support_manifest.json`](../../benchmarks/foundation_model_support_manifest.json).
It records each model's purpose, official URL, filename, SHA256, supported
chemistry fixtures, and contract group.

## Standard foundation models

The following entries are the models used by the foundation support matrix. Run
`sha256sum <file>` after downloading and compare it with this table (the same
values are also machine-readable in the manifest).

| ID | Family / use | Download | Filename | SHA256 |
|---|---|---|---|---|
| `mace-omat-0-small` | OMAT-0 small | [release](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-small.model) | `mace-omat-0-small.model` | `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5` |
| `mace-omat-0-medium` | OMAT-0 medium | [release](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model) | `mace-omat-0-medium.model` | `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a` |
| `mace-mpa-0-medium` | MPA-0 medium | [release](https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model) | `mace-mpa-0-medium.model` | `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638` |
| `mace-mp-0b-small` | MP-0b small | [release](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model) | `mace_agnesi_small.model` | `7e3a0abcaf41e03a80e69f778e1b11b29de1cca704783dc25917a736392f8cf0` |
| `mace-mp-0b-medium` | MP-0b medium | [release](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model) | `mace_agnesi_medium.model` | `ab8baff639a8f295f3eccad3d3ccf574efb6fb63220bd52cd88664211569e521` |
| `mace-mp-0b2-small` | MP-0b2 small | [release](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model) | `mace-small-density-agnesi-stress.model` | `d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9` |
| `mace-mp-0b2-medium` | MP-0b2 medium | [release](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model) | `mace-medium-density-agnesi-stress.model` | `a90be07c8aa6623c390fcc4653d3e319c4a356f8b4a53d323f96b2012d375caf` |
| `mace-mp-0b2-large` | MP-0b2 large | [release](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model) | `mace-large-density-agnesi-stress.model` | `348390e758e1c90011c7675e864850f8e9c5b3e7c217f79a2b4bad1baaa657ad` |
| `mace-mp-0b3-medium` | MP-0b3 medium | [release](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model) | `mace-mp-0b3-medium.model` | `2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012` |
| `mace-matpes-pbe-0` | MATPES PBE | [release](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model) | `MACE-matpes-pbe-omat-ft.model` | `e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa0` |
| `mace-matpes-r2scan-0` | MATPES r2SCAN | [release](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model) | `MACE-matpes-r2scan-omat-ft.model` | `8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9` |
| `mace-mh-0` | MACE-MH-0 backbone | [release](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model) | `mace-mh-0.model` | `d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d` |
| `mace-mh-1` | MACE-MH-1 nonlinear backbone | [release](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model) | `mace-mh-1.model` | `a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47` |
| `mace-off23-small` | MACE-OFF23 small | [raw model](https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_small.model) | `MACE-OFF23_small.model` | `165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f` |
| `mace-off23-medium` | MACE-OFF23 medium | [raw model](https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model) | `MACE-OFF23_medium.model` | `4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7` |
| `mace-off23-large` | MACE-OFF23 large | [raw model](https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_large.model) | `MACE-OFF23_large.model` | `a29e397dbf3e7a24ac50a9b0dfc919bd5a62efa346f5895a6237b0950c1d76f4` |

Example download and verification:

```bash
checkpoint_dir="$HOME/.cache/symmetrix/checkpoints"
mkdir -p "$checkpoint_dir"
curl -L --fail -o "$checkpoint_dir/mace-mh-1.model" \
  https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model
printf '%s  %s\n' \
  a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47 \
  "$checkpoint_dir/mace-mh-1.model"
sha256sum "$checkpoint_dir/mace-mh-1.model"
```

Use the full hash from the table or the manifest; the shortened value above is
only a command-shape example.

## NequIP-OAM S/M/L/XL

The OAM checkpoints used by the foundation support matrix are the NequIP
0.1 packages published by the MIR Group. The [NequIP model
catalog](https://www.nequip.net/models) provides the model metadata and model
IDs; the files are archived in the [official Zenodo record](https://zenodo.org/records/18775904).
Each table link is a direct download of the `.nequip.zip` package.

| Size | Official model page | NequIP model ID | Download | Size (bytes) | MD5 (Zenodo) |
|---|---|---|---|---:|---|
| S | [OAM-S:0.1](https://www.nequip.net/models/mir-group/NequIP-OAM-S%3A0.1) | `nequip.net:mir-group/NequIP-OAM-S:0.1` | [NequIP-OAM-S-0.1.nequip.zip](https://zenodo.org/records/18775904/files/NequIP-OAM-S-0.1.nequip.zip?download=1) | 5,638,676 | `399a98bf36fc4550bc85d48f35451c6c` |
| M | [OAM-M:0.1](https://www.nequip.net/models/mir-group/NequIP-OAM-M%3A0.1) | `nequip.net:mir-group/NequIP-OAM-M:0.1` | [NequIP-OAM-M-0.1.nequip.zip](https://zenodo.org/records/18775904/files/NequIP-OAM-M-0.1.nequip.zip?download=1) | 26,554,438 | `e5b44583f4421be702b35d76503754cd` |
| L | [OAM-L:0.1](https://www.nequip.net/models/mir-group/NequIP-OAM-L%3A0.1) | `nequip.net:mir-group/NequIP-OAM-L:0.1` | [NequIP-OAM-L-0.1.nequip.zip](https://zenodo.org/records/18775904/files/NequIP-OAM-L-0.1.nequip.zip?download=1) | 78,464,590 | `67144367c710a70a53a8e21acf331980` |
| XL | [OAM-XL:0.1](https://www.nequip.net/models/mir-group/NequIP-OAM-XL%3A0.1) | `nequip.net:mir-group/NequIP-OAM-XL:0.1` | [NequIP-OAM-XL-0.1.nequip.zip](https://zenodo.org/records/18775904/files/NequIP-OAM-XL-0.1.nequip.zip?download=1) | 259,627,903 | `3d2369c7238eb83a23141abdcb055a8f` |

Verify a downloaded package against the published checksum, then compute a
local SHA256 if the package is being copied into a long-lived qualification
cache:

```bash
cd "$checkpoint_dir"
printf '%s  %s\n' \
  399a98bf36fc4550bc85d48f35451c6c NequIP-OAM-S-0.1.nequip.zip | md5sum -c -
sha256sum NequIP-OAM-S-0.1.nequip.zip
```

Replace the filename and MD5 with the row for M, L, or XL. The Zenodo record
publishes MD5 rather than SHA256, so retain the locally computed SHA256 with
the qualification record. Export a package to the Symmetrix OAM-v2 JSON
contract with the optional NequIP stack:

```bash
uv pip install "symmetrix-xl[nequip]"
symmetrix_extract_nequip \
  --model NequIP-OAM-S-0.1.nequip.zip \
  --output oam-s-nequip.json
```

OAM currently has host and CUDA routes only; the HIP backend rejects this
family. Keep OAM downloads and exports in the separate OAM qualification cache
and do not use them as HIP evidence.

## MACE-POLAR-1

MACE-POLAR-1 is the `PolarMACE` electrostatics family. The official assets are
published in the [`mace_polar_1` release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_polar_1):

| Size | Download | Filename | SHA256 |
|---|---|---|---|
| S | [model](https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-S.model) | `MACE-POLAR-1-S.model` | `e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510` |
| M | [model](https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-M.model) | `MACE-POLAR-1-M.model` | `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a` |
| L | [model](https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-L.model) | `MACE-POLAR-1-L.model` | `9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b` |

The committed [checkpoint facts](../../symmetrix/test/data/polar/checkpoint_facts.json)
pin the release tag, asset size, hash, model shape, and oracle revisions. The
pinned oracle checkouts are MACE commit
`22f0809735bd4dd1deba80cf8e16f89913a35ff4` and graph-longrange commit
`e0eb8945c21e203656f4e0bbece5d1da488b6753`. End-to-end oracle capture requires
those checkouts plus `mace-torch`, `e3nn`, and `graph-longrange`.

## MACELES and MACEField

MACELES conversion requires the training-time `les` package. The published
MACE-OFF-LES checkpoint is available from the [`les_fit` repository](https://github.com/ChengUCB/les_fit):

- Model: [`MACELES-OFF_small_converted.model`](https://raw.githubusercontent.com/ChengUCB/les_fit/main/MACELES-OFF/MACELES-OFF_small_converted.model)
- Observed SHA256: `44c0f363dd577eafc0912341f44e658fcf5d58d02b88da8682562d654889cf80`
- Convert with `symmetrix_extract_mace` after installing `les`; the runtime consumes the resulting JSON and does not need PyTorch.

MACEField checkpoints are distributed with the MACE-Field project rather than
this repository. Record the project URL, exact checkpoint revision, filename,
and SHA256 before converting. The compact export command is:

```bash
symmetrix_extract_mace --model MACEField-MH-0-omat-dielectric.model \
  --head mp-dielectric --output MACEField-MH-0-omat-dielectric.json
```

## Convert and prepare a checkpoint

Install the optional converter stack, export a universal compact model, and
keep the original checkpoint hash alongside the JSON hash:

```bash
uv pip install "symmetrix-xl[mace]"
symmetrix_extract_mace --model "$checkpoint" --output "$checkpoint.json"
sha256sum "$checkpoint" "$checkpoint.json"
```

For HIP direct qualification, set a private cache and prepare artifacts in the
HIP process. Do not import the native HIP extension and the ROCm PyTorch oracle
in the same process; run the oracle and native workers separately and compare
JSON records.

The large qualification matrix uses the manifest directly:

```bash
python benchmarks/foundation_model_support_matrix.py run \
  --checkpoint-dir "$checkpoint_dir" \
  --compact-dir "$checkpoint_dir/compact" \
  --output-jsonl benchmarks/.artifacts/foundation-hip.jsonl \
  --jit-cache "$HOME/.cache/symmetrix/hip-qualification" \
  --converter-python .venv/bin/python \
  --hip-python .venv/bin/python --hip-selector rocm7-gfx1151 \
  --models mace-mh-0 mace-mh-1 --structures H2O C64 STO40 \
  --dtypes float32 float64 --backends hip
```
