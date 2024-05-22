#date: 2024-05-22T16:51:52Z
#url: https://api.github.com/gists/49f713ab34ba2590941bd3761cc0750d
#owner: https://api.github.com/users/lbianchi-lbl

import json
import io
import os

from pathlib import Path

import coverage
import pytest


def pytest_configure(config: pytest.Config):
    # ensure cov_context is set, otherwise pytest-cov won't be setting the context
    # and many things will break for us
    config.option.cov_context = "test"


def pytest_runtest_logreport(report: pytest.TestReport):
    # associate the cov context set by pytest-cov with the pytest report (which also contains the duration)
    cov_context = os.environ.get("COV_CORE_CONTEXT")
    report.cov_context = cov_context


def _get_report_for_context(cov: coverage.Coverage, context: str):
    report_ctx_before = cov.config.report_contexts
    try:
        cov.config.report_contexts = [context]
        path = Path("internal-cov-report.json")
        total = cov.json_report(ignore_errors=True, outfile=path)
        data = json.loads(path.read_text())
    finally:
        cov.config.report_contexts = report_ctx_before
        path.unlink(missing_ok=True)
    return data


def pytest_terminal_summary(terminalreporter, config: pytest.Config):
    tr = terminalreporter

    cov: coverage.Coverage = config.pluginmanager.getplugin("_cov").cov_controller.cov
    tr.write_line(str(cov))

    by_cov_context = {}
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "cov_context"):
                by_cov_context[rep.cov_context] = rep

    for ctx, rep in by_cov_context.items():
        tr.write_sep("=", ctx)
        all_data = _get_report_for_context(cov, ctx)
        data = all_data["totals"]
        data["pytest_cov_context"] = ctx
        data["pytest_duration_s"] = getattr(rep, "duration", None)
        tr.write_line(json.dumps(data, indent=2))
