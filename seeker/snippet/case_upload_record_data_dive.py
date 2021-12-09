#date: 2021-12-09T16:53:56Z
#url: https://api.github.com/gists/9e9d961cad3065418bdd3c67a96374d7
#owner: https://api.github.com/users/orangejenny

from collections import defaultdict
from datetime import datetime, timedelta
from itertools import islice
from corehq.apps.case_importer.tracking.models import *
from corehq.apps.data_dictionary.models import *
from corehq.util.workbook_reading import open_any_workbook
from corehq.toggles import DATA_DICTIONARY


domain_count = 0
uploads_with_deprecated = 0
uploads_without_deprecated = 0
domains = get_domains()
for domain in domains:
    domain_count += 1
    deprecated_properties = get_deprecated_properties(domain)
    records = get_records(domain)
    record_count = 0
    if records.count():
        print(f"Processing {domain}, domain {domain_count} of {len(domains)}, with {len(records)} records")
    for record in records:
        record_count += 1
        if record.task_status_json.get('state', {}) != 2:
            continue
        if record_count % 5 == 0:
            print(f"Processing record {record_count} of {records.count()} in domain {domain_count} of {len(domains)}: ID {record.id}")
        header = get_header(record)
        included = header & deprecated_properties.get(record.case_type, set())
        if included:
            uploads_with_deprecated += 1
        else:
            uploads_without_deprecated += 1
print(f"Deprecated properties were found in {uploads_with_deprecated} of {uploads_with_deprecated + uploads_without_deprecated} records")


def get_domains():
    return DATA_DICTIONARY.get_enabled_domains()


def get_deprecated_properties(domain):
    properties = defaultdict(set)
    for prop in CaseProperty.objects.filter(case_type__domain=domain, deprecated=True):
        properties[prop.case_type.name].add(prop.name)
    return properties


def get_records(domain):
    start_date = datetime.utcnow().date() - timedelta(days=180)
    return CaseUploadRecord.objects.filter(domain=domain, created__gte=start_date).order_by("-created")


def get_header(record):
    ref = record.get_tempfile_ref_for_upload_ref()
    with open_any_workbook(record.get_tempfile_ref_for_upload_ref()) as w:
        worksheet = w.worksheets[0]
        header = list(islice(worksheet.iter_rows(), 0, 1))[0]
        return {cell.value for cell in header}
