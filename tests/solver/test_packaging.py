import importlib


def test_solver_subpackage_importable():
    assert importlib.import_module("sigml.solver") is not None
