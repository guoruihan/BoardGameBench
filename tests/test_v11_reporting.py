import importlib.util
from pathlib import Path

import pytest


def test_paired_interval_formula_matches_documented_sample_standard_error():
    path=Path(__file__).resolve().parents[1]/'scripts/report_v11.py'
    spec=importlib.util.spec_from_file_location('report_v11',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result=module.interval([1.,2.,3.])
    assert result['mean_difference']==2.
    assert result['normal_95_interval']==pytest.approx([2-1.96/(3**.5),2+1.96/(3**.5)])
    assert result['n_environment_pairs']==3
    assert 'not training-seed uncertainty' in result['scope']
