import os
import signal
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_cli_cleans_worker_after_interrupt(monkeypatch):
    import logitly.cli as cli
    real_popen = subprocess.Popen
    children = []
    def start(*args, **kwargs):
        process = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        children.append(process)
        original_wait = process.wait
        first = True
        def wait(*a, **kw):
            nonlocal first
            if first:
                first = False
                raise KeyboardInterrupt
            return original_wait(*a, **kw)
        process.wait = wait
        return process
    monkeypatch.setattr(cli.subprocess, "Popen", start)
    assert cli.run_worker(cli.parser().parse_args(["validate", "unused"])) == 130
    assert children[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(children[0].pid, 0)
