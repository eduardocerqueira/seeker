#date: 2022-06-27T16:58:47Z
#url: https://api.github.com/gists/6ec6419f6229f7549f82b5b87ee2f658
#owner: https://api.github.com/users/jasonmfehr

import mitmproxy
from datetime import datetime
import math

class RequestsToCSV:
  def __init__(self):
    self.file_handle = open("/Users/jasonmfehr/tmp/requests-" + datetime.now().isoformat().split(".")[0] + ".csv", "w")
    self.file_handle.write("Request,Start Time,EndTime,Request Method,Path,Response Size (bytes),Duration (ms)\n")
    self.file_handle.flush()
    self.request_count = 0

  def __del__(self):
    self.file_handle.close()
  
  def response(self, flow: mitmproxy.http.HTTPFlow):
    start_time = datetime.fromtimestamp(flow.request.timestamp_start).isoformat()
    end_time = datetime.fromtimestamp(flow.response.timestamp_end).isoformat()
    duration = str((flow.response.timestamp_end - flow.request.timestamp_start) * 1000).split(".")[0]

    self.request_count += 1
    self.file_handle.write(str(self.request_count) + "," + start_time + "," + end_time + "," + flow.request.method + "," + flow.request.path + "," + str(len(flow.response.content)) + "," + str(duration) + "\n")
    self.file_handle.flush()



addons = [
  RequestsToCSV()
]