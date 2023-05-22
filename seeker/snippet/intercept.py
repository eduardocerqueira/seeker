#date: 2023-05-22T17:01:16Z
#url: https://api.github.com/gists/a80a72b63fc82664075d6177adf03fdd
#owner: https://api.github.com/users/thewisenerd

# - run with `mitmdump -s intercept.py -q`
# - clear counters with
#   curl -sx localhost:8080 http://example.com/clear | jq
# - show counters with
#   curl -sx localhost:8080 http://example.com/show | jq

import json
import typing

import mitmproxy.http

pk_id_header = 'x-ms-documentdb-partitionkey'
pkr_id_header = 'x-ms-documentdb-partitionkeyrangeid'
item_count_header = 'x-ms-item-count'
ru_charge = 'x-ms-request-charge'


class RequestLogger:
    def __init__(self):
        self.pk_page_total = {}
        self.pk_docs_total = {}
        self.pk_ru_sum = {}

    def make_response_object(self) -> bytes:
        response = {}
        total_page = 0
        total_docs = 0
        total_ru = 0

        for pk_id in self.pk_page_total:
            obj = {
                'page_count': self.pk_page_total[pk_id],
                'docs_count': self.pk_docs_total[pk_id],
                'ru': self.pk_ru_sum[pk_id]
            }
            total_page += obj['page_count']
            total_docs += obj['docs_count']
            total_ru += obj['ru']
            response[pk_id] = obj

        response['total'] = {
            'page_count': total_page,
            'docs_count': total_docs,
            'ru': total_ru
        }

        return json.dumps(response).encode()

    def request(self, flow: mitmproxy.http.HTTPFlow):
        if flow.request.pretty_url == 'http://example.com/show':
            flow.response = mitmproxy.http.Response.make(200, self.make_response_object(), {
                'content-type': 'application/json'
            })
            return

        if flow.request.pretty_url == 'http://example.com/clear':
            self.pk_page_total = {}
            self.pk_docs_total = {}
            self.pk_ru_sum = {}
            flow.response = mitmproxy.http.Response.make(200, b'{"status": "ok"}', {
                'content-type': 'application/json'
            })

    def response(self, flow: mitmproxy.http.HTTPFlow):
        req = flow.request
        res = flow.response

        pkr_id: typing.Optional[str] = None
        mode: typing.Optional[int] = None
        if pkr_id_header in req.headers:
            pkr_id = req.headers[pkr_id_header]
            mode = 2
        if pk_id_header in req.headers:
            pkr_id = req.headers[pk_id_header]
            mode = 1

        if pkr_id is None:
            print(f"unhandled response {req.method} {req.path}")
            return

        res_count = res.headers[item_count_header] if item_count_header in res.headers else None
        res_ru = res.headers[ru_charge] if ru_charge in res.headers else None

        if res_count is None:
            if mode > 1:
                print("unhandled, res_count is None")
                return
            else:
                res_count = 1
        if res_ru is None:
            print("unhandled, res_ru is None")
            return

        if pkr_id in self.pk_page_total:
            self.pk_page_total[pkr_id] = self.pk_page_total[pkr_id] + 1
        else:
            self.pk_page_total[pkr_id] = 1
            self.pk_docs_total[pkr_id] = 0
            self.pk_ru_sum[pkr_id] = 0.0

        self.pk_docs_total[pkr_id] += int(res_count)
        self.pk_ru_sum[pkr_id] += float(res_ru)


addons = [
    RequestLogger()
]