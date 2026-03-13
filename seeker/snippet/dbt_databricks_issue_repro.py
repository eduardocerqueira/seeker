#date: 2026-03-13T17:17:04Z
#url: https://api.github.com/gists/4ae219ab7353da234b7f7e3fcb41802e
#owner: https://api.github.com/users/vergenzt

#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dbt-core==1.11.6",
#     "dbt-databricks==1.11.6",
# ]
# ///


import json
import shlex
from pathlib import Path
from subprocess import CalledProcessError, check_call as sh
from tempfile import TemporaryDirectory


def test() -> None:
    with TemporaryDirectory(delete=False) as tmpdir:
        print(tmpdir, end="\n\n")
        dbt = lambda *a: sh(  # noqa
            ((args := ["dbt", *a]), print("+", shlex.join(args)))[0],
            cwd=tmpdir,
        )

        tmp = Path(tmpdir)
        tmp.joinpath("dbt_project.yml").write_text(
            json.dumps(
                {
                    "name": "dbt_databricks_issue_repro",
                    "version": "0.1",
                    "profile": "dbt_databricks_test",
                    "flags": {
                        # https://github.com/dbt-labs/dbt-core/issues/12268
                        "require_all_warnings_handled_by_warn_error": True,
                        "warn_error_options": {"silence": ["BehaviorChangeEvent"]},
                    },
                }
            )
        )

        (models := tmp / "models").mkdir()

        (base := models / "base.sql").write_text(
            "select 1 as id, 'foo' as name",
        )
        (_mv_sql := models / "mv.sql").write_text(
            "select * from {{ ref('base') }} {{ config(materialized='materialized_view') }}"
        )
        (mv_yml := models / "mv.yml").write_text(
            json.dumps({"models": [{"name": "mv", "columns": [{"name": "id"}, {"name": "name"}]}]})
        )

        # initialize everything (mv gets a defined set of columns)
        print("", "### initial run:", "", sep="\n")
        dbt("run")

        # add new column to base
        base.write_text(base.read_text() + ", 42 as new_column")

        print("", "### run after adding base column:", "", sep="\n")
        try:
            dbt("run")
            assert False, "shouldn't get here"
        except CalledProcessError:
            # expecting `Table 'workspace.dbt_test.mymv' has a user-specified schema that is incompatible with the schema inferred from its query.`
            pass

        # add new column to mv_yml
        mv_yml_val = json.loads(mv_yml.read_text())
        mv_yml_val["models"][0]["columns"].append({"name": "new_column"})
        mv_yml.write_text(json.dumps(mv_yml_val))

        print("", "### after adding column to mv.yml (should succeed!):", "", sep="\n")
        dbt("run")


if __name__ == "__main__":
    test()
