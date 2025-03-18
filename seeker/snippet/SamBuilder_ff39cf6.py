#date: 2025-03-18T16:53:59Z
#url: https://api.github.com/gists/ff39cf6bc421e32659790d30954f87bb
#owner: https://api.github.com/users/fulcrum-blog

with FastxFile(fastq_file) as fh:
    query_to_fastq: dict[str, FastqProxy] = {
        entry.name: entry for entry in fh
    }

    assert len(query_to_fastq.keys()) == 2
    assert "query1" in query_to_fastq
    assert "query2" in query_to_fastq
    assert "query3" not in query_to_fastq
