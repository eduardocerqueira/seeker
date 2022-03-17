#date: 2022-03-17T17:04:52Z
#url: https://api.github.com/gists/bc98247a8d7ada8094df286be03fb000
#owner: https://api.github.com/users/iiAnderson

from metaflow import FlowSpec, step, Parameter, conda_base
from gdelt_columns import columns
import pandas as pd

@conda_base(libraries={ "s3fs": "0.6.0", "pandas": "1.1.5", 'boto3': '1.17.11'})
class FetchGDELTData(FlowSpec):

    url = Parameter('url', help="URL for GDELT data", required=True)

    @step
    def start(self):
        print(f"Fetching data for {self.url}")

        self.next(self.fetch_data)

    @step
    def fetch_data(self):

        df = pd.read_csv(self.url, compression='zip', delimiter="\t", names=columns)

        print(df.info(memory_usage='deep'))

        if df is None:
            raise Exception(f"Response from GDELT did not return any data")

        self.data = df
        print(df)
        print(df.columns)

        self.next(self.filter_data)

    @step
    def filter_data(self):

        self.columns_to_keep = [
            "avgtone",
            "goldsteinscale",
            "sourceurl",
            "actor1name",
            "actor2name",
            "numarticles",
            "nummentions",
            "numsources",
            "dateadded"
        ]
        self.results = []

        self.data = self.data[self.columns_to_keep]
        
        # Drop columns with no date
        self.data = self.data.dropna(subset=['dateadded'])

        # Dates are in format 20210429081500, we just want the year/month/day
        self.data['dateadded'] = self.data['dateadded'].astype(str).str[0:8]
        self.data['dateadded'] = pd.to_datetime(self.data['dateadded'], errors='coerce', format='%Y%m%d')

        # Gives: {day: pd.DataFrame}
        self.data = self.data.groupby(['dateadded'])

        for day, df in self.data:
            print(f"Processing {day}")

            pattern = "coronavirus|covid-19|covid19|covid"
            
            import re
            # Find terms relating to Covid in the sourceurl
            df = df[
                df['sourceurl'].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
            ]
            # Convert our metrics to numeric values
            df['goldsteinscale'] = df['goldsteinscale'].apply(pd.to_numeric, errors='coerce')
            df['avgtone'] = df['avgtone'].apply(pd.to_numeric, errors='coerce')

            # Save our results to self.results!
            self.results.append({
                "date": day,
                "count": df.shape[0],
                "goldsteinscale": df['goldsteinscale'].mean(),
                "avgtone": df['avgtone'].mean()
            })

        self.next(self.end)

    @step
    def end(self):
        print(f"The Flow has finished")

if __name__ == "__main__":
    FetchGDELTData()