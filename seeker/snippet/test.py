#date: 2021-11-23T17:12:10Z
#url: https://api.github.com/gists/0dff09f1887407b586486ab721ee95fe
#owner: https://api.github.com/users/vijaykiran

from sodasql.scan.scan_builder import ScanBuilder
from sodasql.scan.failed_rows_processor import FailedRowsProcessor


class TestFailedRowProcessor(FailedRowsProcessor):
    def process(self, context: dict):
        sample_name = context.get('sample_name')
        column_name = context.get('column_name')
        sample_columns = context.get('sample_columns')
        sample_description = context.get('sample_description')
        total_row_count = context.get('total_row_count')
        sample_rows = context.get('sample_rows')

        with open(f'{sample_name}.txt', 'w') as f:
            for row in sample_rows:
                f.write(f'{row}\n')

        return {'message': 'failed rows are saved somewhere else',
                'count': 42}


scan_builder = ScanBuilder()

scan_builder.warehouse_yml_file = "warehouse.yml"
scan_builder.scan_yml_file = "table_scan.yml"
scan_builder.failed_rows_processor = TestFailedRowProcessor()
scan = scan_builder.build()
scan_result = scan.execute()