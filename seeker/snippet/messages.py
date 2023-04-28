#date: 2023-04-28T16:56:07Z
#url: https://api.github.com/gists/e1611db5175862b94bb2a2ccd77518bf
#owner: https://api.github.com/users/PubliusAu

response_format = """
Topic: <theme in one or two words>
Sentiment: <sentiment of all texts>
Summary: <sentence>
"""
cluster_summarizer_context = [
    {
        "role": "system",
        "content": "**********": <DIVIDER>, so separate the input response by this separator token to analyze what each sentence is talking about and then generate a single sentence that represents the general theme of the sentences. Be a bit general but not too general. Make sure each summary is unique to the cluster. The format of this response will be the following: {response_format}",
    },
]ot too general. Make sure each summary is unique to the cluster. The format of this response will be the following: {response_format}",
    },
]