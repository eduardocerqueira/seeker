#date: 2025-03-18T16:53:18Z
#url: https://api.github.com/gists/cc755cf12d0e91621785e7eded1f215a
#owner: https://api.github.com/users/fulcrum-blog

fastq_file = tmp_path / “test.fastq” 

bam2fastq(
  bam_file=bam_file,
  fastq_file=fastq_file,
  include_supplementary=False
)
