import torch

from dsorn_v31 import DSORNetV31Sequential, run_unit_tests


def test_internal_geometry_and_backward_checks():
    results = run_unit_tests()
    assert results["conv_module_count"] == 0
    assert results["zero_motion_transport_error"] < 2e-6
    assert results["constant_motion_transport_error"] < 2e-6
    assert results["transport_gradient_norm"] > 0
    assert results["full_forward_backward_finite"] is True


def test_forward_shape_and_finiteness():
    model = DSORNetV31Sequential(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 10)
    assert torch.isfinite(y).all()
