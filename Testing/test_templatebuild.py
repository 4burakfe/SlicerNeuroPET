"""Tests of the iterative template helpers without Slicer: python -m pytest Testing."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NeuroPET"))
from NeuroPETLib import templatebuild as tb  # noqa: E402


def _affine(scale=1.0, shift=(0.0, 0.0, 0.0), angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4)
    m[:3, :3] = scale * np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    m[:3, 3] = shift
    return m


def test_drift_taken_out():
    initial = {f"s{i}": _affine(1.0 + 0.02 * i, (i, -i, 0.5 * i), 0.01 * i) for i in range(6)}
    drift = _affine(1.08, (3.0, 1.0, -2.0))       # the whole group grew and moved
    current = {k: drift @ m for k, m in initial.items()}
    corrected, found = tb.driftCorrected(current, initial)
    assert abs(tb.scaleOf(found) - 1.08) < 1e-9
    for k in initial:
        assert np.allclose(corrected[k], initial[k])
    # individual refinements are kept, only their mean is taken out
    current = dict(current, s0=drift @ _affine(1.0, (2.0, 0, 0)) @ initial["s0"])
    corrected, _ = tb.driftCorrected(current, initial)
    assert tb.displacementMm(corrected["s0"], corrected["s1"]) > 0


def test_displacement_and_convergence():
    assert tb.displacementMm(np.eye(4), np.eye(4)) == 0
    assert abs(tb.displacementMm(_affine(shift=(1.0, 0, 0)), np.eye(4)) - 1.0) < 1e-9
    stats = tb.passStatistics({"a": _affine(shift=(0.2, 0, 0)), "b": np.eye(4)}, {"a": np.eye(4), "b": np.eye(4)})
    assert stats["n"] == 2 and abs(stats["meanMm"] - 0.1) < 1e-9 and abs(stats["maxMm"] - 0.2) < 1e-9
    assert tb.converged(stats, 0.5) and not tb.converged(stats, 0.05)
    assert not tb.converged(tb.passStatistics({}, {}))


def test_correlation():
    a = np.arange(27, dtype=float).reshape(3, 3, 3)
    assert abs(tb.correlation(a, 2 * a + 1) - 1.0) < 1e-12
    assert np.isnan(tb.correlation(a, a[:2]))


def test_transform_files(tmp_path):
    folder = str(tmp_path)
    tb.writeTransform(folder, "sub-1", 0, _affine(1.1), "/x/sub-1_pet.nii.gz")
    tb.writeTransform(folder, "sub-1", 1, _affine(1.2))
    tb.writeTransform(folder, "sub-1", 1, _affine(1.3))      # a pass written again replaces it
    document = tb.readTransforms(folder, "sub-1")
    assert [p["pass"] for p in document["passes"]] == [0, 1] and document["source"].endswith("pet.nii.gz")
    assert np.allclose(tb.passMatrix(document, 1), _affine(1.3))
    assert tb.passMatrix(document, 5) is None and tb.readTransforms(folder, "sub-2") is None
    assert tb.templateFile(2) == "templates/pass_2.nii.gz"
    assert os.path.isfile(os.path.join(folder, "subjects", "sub-1", "transform.json"))
    log = os.path.join(folder, "qc", tb.ITERATIONS_LOG)
    tb.writeIterationsLog(log, [{"pass": 0, "subjects": 3, "extra": 1}])
    assert open(log).read().splitlines()[0].startswith("pass,target")


def test_pass_taken_back(tmp_path):
    folder = str(tmp_path)
    tb.writeTransform(folder, "sub-1", 0, _affine(1.1), "/x/sub-1_pet.nii.gz")
    tb.writeTransform(folder, "sub-1", 1, _affine(1.2))
    tb.dropPass(folder, "sub-1", 1)
    tb.writeTransform(folder, "sub-1", 0, _affine(1.1))         # as the build puts a subject back
    document = tb.readTransforms(folder, "sub-1")
    assert [p["pass"] for p in document["passes"]] == [0]
    assert tb.passMatrix(document, 1) is None
    np.testing.assert_allclose(np.asarray(document["current"]["matrix"]).reshape(4, 4), _affine(1.1))
    assert document["current"]["target"] == "MNI target" and document["source"] == "/x/sub-1_pet.nii.gz"
    tb.dropPass(folder, "nobody", 1)                             # no file: nothing


def test_scaled_mean_of_summed_images():
    mask = np.zeros((4, 4, 4), bool)
    mask[1:3, 1:3, 1:3] = True
    a = np.where(mask, 10.0, 1.0)
    b = np.where(mask, 1000.0, 100.0)      # 100 times the activity: the same shape after the scaling
    mean, count = tb.scaledMean([a, b], mask)
    assert count == 2 and mean.dtype == np.float32
    assert np.allclose(mean[mask], tb.SUMMED_SCALE) and np.allclose(mean[~mask], tb.SUMMED_SCALE / 10)
    mean, count = tb.scaledMean([a, np.zeros_like(a)])          # an empty image is left out
    assert count == 1
    try:
        tb.scaledMean([np.zeros((2, 2, 2))])
        raise AssertionError("no image: an error")
    except ValueError:
        pass
