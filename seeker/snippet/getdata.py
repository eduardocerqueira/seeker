#date: 2022-12-29T16:26:25Z
#url: https://api.github.com/gists/5c3f8bad0e56b87f0066dc781d4d2866
#owner: https://api.github.com/users/canalquant

# region imports
from AlgorithmImports import *
# endregion

class UglyGreenDonkey(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2017, 1, 1)  # Set Start Date
        self.SetCash(100000)  # Set Strategy Cash
        self.e = self.AddForex("EURUSD",Resolution.Hour).Symbol

    def OnData(self, data: Slice):
        self.Log(f"Time: {self.Time},Open: {data[self.e].Open},High: {data[self.e].High},Low: {data[self.e].Low},Close: {data[self.e].Close}")

    