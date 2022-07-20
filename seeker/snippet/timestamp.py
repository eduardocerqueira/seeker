#date: 2022-07-20T17:12:18Z
#url: https://api.github.com/gists/728cddd9825ad8ce8ad048837433a741
#owner: https://api.github.com/users/tmacam

import datetime
import sublime, sublime_plugin
 
class TimestampCommand(sublime_plugin.TextCommand):
  def run(self, edit):
    timestamp = "\n[%s]\t" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    self.view.insert(edit, self.view.sel()[0].begin(), timestamp)