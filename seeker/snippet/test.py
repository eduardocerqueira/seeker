#date: 2022-03-14T17:01:31Z
#url: https://api.github.com/gists/72ba578a357cd4b9bf3e1d2bd0bd7a9a
#owner: https://api.github.com/users/jay-zhuang

#!/usr/bin/env python3

import json
import logging
import sys
from datetime import datetime
from rfe.scubadata.scubadata_py3 import ScubaData, Sample

logging.basicConfig(stream=sys.stdout,
                    format="%(levelname)s %(asctime)s - %(message)s",
                    level=logging.INFO)

class MicroBenchScuba:
    def __init__(self, result_filename):
        with open(result_filename) as f:
            self.result = json.load(f)

    def add_context(self):
        for sample in self.samples:
            for ctx in self.result['context'].items():
                ctx_type = type(ctx[1])
                if ctx[0] == 'date':
                    assert ctx_type is str
                    ts = datetime.fromisoformat(ctx[1])
                    sample.addTimestamp(ScubaData.TIME_COLUMN, ts)
                elif ctx_type is str:
                    sample.addNormalValue(ctx[0], ctx[1])
                elif ctx_type is int:
                    sample.addIntValue(ctx[0], ctx[1])
                elif ctx_type is float:
                    sample.addFloatValue(ctx[0], ctx[1])
                elif ctx_type is bool:
                    sample.addIntValue(ctx[0], ctx[1])
                else:
                    logging.debug("Ignore benchmark context {}".format(ctx[0]))

    def load_samples(self):
        self.samples = []
        for benchmark in self.result['benchmarks']:
            sample = Sample()
            for it in benchmark.items():
                it_type = type(it[1])
                if it_type is str:
                    sample.addNormalValue(it[0], it[1])
                elif it_type is int:
                    sample.addIntValue(it[0], it[1])
                elif it_type is float:
                    sample.addFloatValue(it[0], it[1])
                elif it_type is bool:
                    sample.addIntValue(it[0], it[1])
                else:
                    raise RuntimeError("unsupported benchmark result, key: {}, value {}".format(it[0], it[1]))
            self.samples.append(sample)
        self.add_context()

    def submit(self):
        logging.info("Loading benchmark result")
        self.load_samples()
        logging.info("Submitting benchmark result")
        with ScubaData("rocksdb_benchmark_test") as scubadata:
            scubadata.addSamples(self.samples)
        logging.info("Submitted benchmark result")

def main():
    filename = sys.argv[1]
    scuba = MicroBenchScuba(filename)
    scuba.submit()

if __name__ == '__main__':
    main()