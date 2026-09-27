import subprocess
import sys
import tomllib
from pathlib import Path


def test_public_api_exports_unique_names():
    import logitly

    assert len(logitly.__all__) == len(set(logitly.__all__))
    for name in logitly.__all__:
        assert getattr(logitly, name) is not None


def test_branded_cli_and_metadata():
    result = subprocess.run([sys.executable, "-m", "logitly", "--help"],
                            capture_output=True, text=True, check=True)
    assert "usage: logitly" in result.stdout
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert config["project"]["name"] == "logitly"
    assert config["project"]["scripts"] == {"logitly": "logitly.cli:main"}
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/logitly"]
