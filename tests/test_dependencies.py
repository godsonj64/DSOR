from pathlib import Path
import re


def test_python_locks_use_identical_package_versions():
    root = Path(__file__).resolve().parents[1]
    pins = []
    for minor in ('3.11', '3.12'):
        text = (root/f'requirements-ci-{minor}-lock.txt').read_text()
        pins.append(dict(re.findall(r'^([A-Za-z0-9_-]+)==([^\s\\]+)', text, re.MULTILINE)))
        assert '--hash=sha256:' in text
    assert pins[0] == pins[1]
    assert pins[0]['torch'] == '2.5.1+cpu' and pins[0]['torchvision'] == '0.20.1+cpu'
    assert pins[0]['setuptools'] == '75.8.0'
