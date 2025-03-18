#date: 2025-03-18T16:51:41Z
#url: https://api.github.com/gists/b43c0968342cb175f797daed237595b9
#owner: https://api.github.com/users/fulcrum-blog

def bam2fastq(
    bam_file: Path,
    fastq_file: Path,
    include_supplementary: bool = False,
) -> None:
    """
    Converts reads in a BAM file to FASTQ format.
    Excludes supplementary reads by default.
    Conversion to ASCII encoded quality scores adds 33 to each integer
    score and converts it to an ASCII character, then joins the 
    converted quality scores together into a string.

    Args:
        bam_file: Path to the input BAM file.
        fastq_file: Path to the output FASTQ file.
        include_supplementary: Whether to include supplementary reads
          in the output (default: False).
    """
    with reader(bam_file) as bam, open(fastq_file, "w") as fastq:
        for alignment in bam:
            if include_alignment(
                alignment=alignment,
                include_supplementary=include_supplementary,
            ):
                fastq.write(f"{alignment_to_fastq(alignment)}")
