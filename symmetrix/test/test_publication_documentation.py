"""Citation-guide coverage for implemented methods and reference software."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def test_publication_guide_covers_implemented_method_families():
    guide = (_ROOT / "docs/reference/publications.md").read_text()

    required_methods = (
        "10.52202/068431-0830",  # MACE
        "10.48550/arXiv.2510.25380",  # MACE-MH-1
        "10.1038/s41524-025-01577-7",  # LES
        "10.1021/acs.jctc.5c01400",  # MACELES integration
        "10.48550/arXiv.2602.19411",  # MACE-POLAR-1
        "10.1038/s41467-022-29939-5",  # NequIP
        "10.1002/jcc.20495",  # DFT-D2
        "10.1063/1.3382344",  # DFT-D3
        "10.1002/jcc.21759",  # DFT-D3(BJ)
        "10.1021/acs.jpclett.6b00780",  # DFT-D3M damping
        "10.48550/arXiv.2607.15103",  # FourierD3
        "10.1038/s41467-022-30687-9",  # torch-dftd software citation
    )
    for identifier in required_methods:
        assert identifier in guide


def test_publication_guide_credits_numerical_reference_software():
    guide = (_ROOT / "docs/reference/publications.md").read_text()

    required_projects = (
        "ACEsuit/mace",
        "mdi-group/mace-field",
        "ChengUCB/les",
        "mir-group/nequip",
        "e3nn/e3nn",
        "pfnet-research/torch-dftd",
        "NVIDIA/nvalchemi-toolkit-ops",
        "WillBaldwin0/graph_electrostatics",
        "lab-cosmo/sphericart",
        "libAtoms/matscipy",
    )
    for project in required_projects:
        assert project in guide

    assert "not a runtime dependency" in guide
    assert "License attribution is a separate obligation" in guide


def test_method_guides_link_to_the_central_citation_guide():
    for relative_path in (
        "docs/mace_les.md",
        "docs/user/dispersion_corrections.md",
    ):
        contents = (_ROOT / relative_path).read_text()
        assert "{doc}`/reference/publications`" in contents
