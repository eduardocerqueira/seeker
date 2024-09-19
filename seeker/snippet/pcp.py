#date: 2024-09-19T16:43:52Z
#url: https://api.github.com/gists/6f4d683fcd3f8797945f5f90bf0fe5d6
#owner: https://api.github.com/users/jelly

#!/usr/bin/python

import tempfile

from cpmapi import (
    PM_ID_NULL,
    PM_INDOM_NULL,
    PM_SEM_COUNTER,
    PM_SEM_DISCRETE,
    PM_SEM_INSTANT,
    PM_TYPE_64,
    PM_TYPE_DOUBLE,
    PM_TYPE_STRING,
    PM_TYPE_U32,
    PM_TYPE_U64,
)
from pcp import pmi
import cpmapi as c_api
from pcp import pmapi


def create_archive(archive_dir):
    archive_1 = pmi.pmiLogImport(f"{archive_dir}/0")

    archive_1.pmiAddMetric("mock.value", PM_ID_NULL, PM_TYPE_64, PM_INDOM_NULL,
                           PM_SEM_INSTANT, archive_1.pmiUnits(0, 0, 0, 0, 0, 0))

    domain = 60  # Linux kernel
    pmid = archive_1.pmiID(domain, 2, 0)
    indom = archive_1.pmiInDom(domain, 2)
    units = archive_1.pmiUnits(0, 0, 0, 1, 0, 0)

    archive_1.pmiAddMetric("kernel.all.load", pmid, PM_TYPE_DOUBLE, indom,
                           PM_SEM_INSTANT, units)

    pmid = archive_1.pmiID(domain, 1, 58)
    units = archive_1.pmiUnits(1, 0, 0, 1, 0, 0)
    archive_1.pmiAddMetric("mem.util.available", pmid, PM_TYPE_U64, PM_INDOM_NULL,
                           PM_SEM_INSTANT, units)

    archive_1.pmiAddInstance(indom, "1 minute", 1)
    archive_1.pmiAddInstance(indom, "5 minute", 5)
    archive_1.pmiAddInstance(indom, "15 minute", 15)

    for i in range(1000):
        archive_1.pmiPutValue("mock.value", None, str(i))
        archive_1.pmiPutValue("mem.util.available", None, str(i * 10))

        archive_1.pmiPutValue("kernel.all.load", "1 minute", "1.0")
        archive_1.pmiPutValue("kernel.all.load", "5 minute", "5.0")
        if i < 4:
            archive_1.pmiPutValue("kernel.all.load", "15 minute", "15.0")

        archive_1.pmiWrite(i, 0)

    archive_1.pmiEnd()


def get_samples():
    pass


def get_samples_fetchgroup(archive_path: str):
    pmfg = pmapi.fetchgroup(c_api.PM_CONTEXT_ARCHIVE, archive_path)
    context = pmfg.get_context()

    kernel_all_load = pmfg.extend_indom("kernel.all.load", c_api.PM_TYPE_FLOAT)
    mock_value = pmfg.extend_item("mock.value")
    mem_util_avail = pmfg.extend_item("mem.util.available", c_api.PM_TYPE_FLOAT)
    mem_util_avail_scaled = pmfg.extend_item("mem.util.available", c_api.PM_TYPE_FLOAT, "byte")
    t = pmfg.extend_timestamp()
    for _ in range(5):
        status = pmfg.fetch()
        print(t())
        if status < 0:
            print("error during fetch")

        print("mock.value", mock_value())
        print("mem.util.available", mem_util_avail())
        print("mem.util.available_scaled", mem_util_avail_scaled())
        for icode, iname, value in kernel_all_load():
            print(f"kernel.all.load {iname}={value()}")


def main():
    with tempfile.TemporaryDirectory() as archive_dir:
        create_archive(archive_dir)
        get_samples_fetchgroup(archive_dir)


if __name__ == "__main__":
    main()