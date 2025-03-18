#date: 2025-03-18T16:59:28Z
#url: https://api.github.com/gists/2f215d2e2030135c8dec89b04ff22bb5
#owner: https://api.github.com/users/fulcrum-blog

builder = SamBuilder(r1_len=20, r2_len=20)
(read1, read2) = builder.add_pair(
    name="query",
    chrom1="chr1",
    start1=100,
    bases1="A" * 10,
    chrom2="chr1",
    start2=200,
    bases2="C" * 20,
)
read1_supp = builder.add_single(
    name="query",
    read_num=1,
    chrom="chr2",
    start=190,
    supplementary=True
)
template = Template.build([read1, read2, read1_supp])
