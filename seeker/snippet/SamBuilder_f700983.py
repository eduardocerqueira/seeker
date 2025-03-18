#date: 2025-03-18T16:52:43Z
#url: https://api.github.com/gists/f7009839214f973f4642a564c5853455
#owner: https://api.github.com/users/fulcrum-blog

def test_bam2fastq(tmp_path: Path) -> None:
    """
    Tests converting a BAM file to FASTQ format, excluding supplementary
    reads.
    """
    builder = SamBuilder(r1_len=10, base_quality=30)
    builder.add_single(name="query1", bases="A" * 10)
    builder.add_single(name="query2", bases="C" * 10)
    builder.add_single(
        name="query3", bases="G" * 10, supplementary=True
    )
    bam_file = tmp_path / "test.bam"
    builder.to_path(bam_file)
