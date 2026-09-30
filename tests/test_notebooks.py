import ast
import json
from pathlib import Path
import re

import nbformat
import pytest


@pytest.mark.parametrize('name', ['DSORNet_v31_Colab.ipynb', 'DSORNet_v32_Colab.ipynb'])
def test_notebook_schema_ids_and_python_code(name):
    path = Path(__file__).resolve().parents[1]/'colab'/name
    raw = json.loads(path.read_text())
    ids = [cell['id'] for cell in raw['cells']]
    assert len(ids) == len(set(ids))
    nbformat.validate(raw)
    code = '\n'.join(''.join(c['source']) for c in raw['cells'] if c['cell_type'] == 'code')
    ast.parse(code)
    assert re.search(r'CODE_REVISION = "[0-9a-f]{40}"', code)
    assert 'sys.executable' in code and 'check=True' in code
    assert 'rm -rf' not in code
    assert code.count('subprocess.run(cmd, cwd=REPO_DIR, check=True)') == 1
    assert code.index('drive.mount(') < code.index('subprocess.run(cmd,')
