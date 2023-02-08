#date: 2023-02-08T17:01:26Z
#url: https://api.github.com/gists/e39cf8bb2e92e690f3ca8d884d87320c
#owner: https://api.github.com/users/timhberry

import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions

# pub/sub subscription in the format: projects/<your-project>/subscriptions/<subscription-name>
sub = ''

# bigquery table in the format: project:dataset.table

# create table first with correct schema
output_table = 'ab-academy-demo:btlearningpath.dflowtest'
schema = 'country:STRING,country_code:STRING,region:STRING,region_code:STRING,city:STRING,date:DATE,download_kbps:FLOAT,upload_kbps:FLOAT,total_tests:INTEGER,distance_miles:FLOAT'


class CustomParsing(beam.DoFn):
    def process(self, element: bytes, timestamp=beam.DoFn.TimestampParam, window=beam.DoFn.WindowParam):
        """
        Simple processing function to parse the data
        """
        parsed = json.loads(element.decode("utf-8"))
        yield parsed


# replace <your-project-name> and <your-bucket-name> below
pipeline_options = PipelineOptions(
        runner='DataflowRunner',
        project='<your-project-name>',
        job_name='dataflow-to-bigquery',
        temp_location='gs://<your-bucket-name>/temp',
        region='europe-west2',
        streaming=True,
        save_main_session=True,
        service_account_email='<your-sa-email>'
)


with beam.Pipeline(options=pipeline_options) as pipeline:
        (
            pipeline
            | "Read from Pub/Sub" >> beam.io.gcp.pubsub.ReadFromPubSub(subscription=sub)
            | "CustomParse" >> beam.ParDo(CustomParsing())
            | 'Write to BigQuery' >> beam.io.gcp.bigquery.WriteToBigQuery(
                output_table,
                schema=schema,
            )
        )
