#date: 2021-09-08T17:10:27Z
#url: https://api.github.com/gists/2d6ad2210cf9f66108ff48a9c7566ebc
#owner: https://api.github.com/users/egalpin

import argparse

from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

import apache_beam as beam
from apache_beam.transforms.combiners import ToDictCombineFn
from apache_beam.transforms.combiners import ToSetCombineFn
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.pvalue import TaggedOutput

class Foobar(beam.DoFn):
    KV = 'foo'
    KKV = 'bar'

    def process(self, element: str) -> Iterable[Union[
                    Tuple[str, str],
                    Tuple[str, Tuple[str, List[str]]]
                ]]:

        if len(element) % 2:
            # Output to main i.e. KV
            yield (element, 'foo')
        else:
            yield TaggedOutput(Foobar.KKV,
                               (element, (element + 'bar', ['baz'])))


def run(argv=None, save_main_session=True):
    parser = argparse.ArgumentParser()
    _, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions()
    pipeline_options.view_as(SetupOptions).save_main_session =\
        save_main_session

    with beam.Pipeline(options=pipeline_options) as p:
        partitioned_elements = (
            p
            | 'Create some test values' >> beam.Create([
                'test',
                'tester',
                'testing',
                'tests',
            ])
            | 'Partition the test values by length' >> beam.ParDo(Foobar())
        )

        '''
        This seems to run fine on PortableRunner + DirectRunner, but on
        Dataflow v2 runner, this fails with:

        line: "shuffle_dax_writer.cc:59"

        message: "Check failed: kv_coder : expecting a KV coder, but had
        Strings
        '''
        (
            partitioned_elements[Foobar.KV]
            | beam.CombinePerKey(ToSetCombineFn())
        )

        (
            partitioned_elements[Foobar.KKV]
            | beam.CombinePerKey(ToDictCombineFn())
        )


if __name__ == '__main__':
    run()
