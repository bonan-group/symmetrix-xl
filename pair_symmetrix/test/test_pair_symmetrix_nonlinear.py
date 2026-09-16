import json

import pytest


try:
    from lammps import lammps
except ImportError as exc:
    pytest.skip(
        f"LAMMPS Python module is not available: {exc}", allow_module_level=True
    )


@pytest.mark.parametrize(
    "cmdargs,pair_style",
    [
        ([], "symmetrix/mace"),
        (["-k", "on", "-sf", "kk"], "symmetrix/mace/kk"),
    ],
)
def test_lammps_rejects_nonlinear_mace_before_legacy_parsing(
    tmp_path,
    cmdargs,
    pair_style,
):
    model_path = tmp_path / "nonlinear.json"
    model_path.write_text(
        json.dumps(
            {
                "symmetrix_format_version": 3,
                "model_type": "MACE_Nonlinear",
            }
        )
    )
    lmp = lammps(cmdargs=["-screen", "none", *cmdargs])
    try:
        lmp.commands_string(
            f"""
            clear
            units       metal
            atom_style  atomic
            region      box block 0 4 0 4 0 4
            create_box  1 box
            mass        1 1.0
            pair_style  {pair_style}
            """
        )
        with pytest.raises(Exception, match="MACE_Nonlinear models are not supported"):
            lmp.command(f"pair_coeff * * {model_path} H")
    finally:
        lmp.close()
