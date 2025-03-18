#date: 2025-03-18T16:55:54Z
#url: https://api.github.com/gists/93eee9ed898f12441c246c6e79a3ce84
#owner: https://api.github.com/users/fulcrum-blog

builder = SamBuilder(r1_len=10, r2_len=10)

for offset in range(num_reads): 
    builder.add_pair(
        chrom1=”chr1″,
        start1=100 + offset,
        chrom2=”chr2″,
        start2=200 – offset
    )

for offset in range(num_unmapped): 
    builder.add_pair()

builder.to_path(test_bam_file)
