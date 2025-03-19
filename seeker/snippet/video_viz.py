#date: 2025-03-19T16:42:00Z
#url: https://api.github.com/gists/116c0ef654a44400d70ef73bef8428cf
#owner: https://api.github.com/users/bigsnarfdude

from flask import Flask, render_template
import altair as alt
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chart')
def chart_json():
    data = {
        'Year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'Video Count': [960, 1009, 1471, 1473, 1420, 1567, 1177, 407, 1003, 1083, 1062, 1377]
    }
    df = pd.DataFrame(data)

    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Year:Q', title='Year'),
        y=alt.Y('Video Count:Q', title='Video Count'),
        tooltip=['Year', 'Video Count']
    ).properties(
        title='Video Count Over Time (Excluding 2025)'
    ).interactive()

    return chart.to_json()

if __name__ == '__main__':
    app.run(debug=True)