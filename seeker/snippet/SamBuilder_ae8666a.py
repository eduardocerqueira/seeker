#date: 2025-03-18T16:59:01Z
#url: https://api.github.com/gists/ae8666aa26de6a71e6999ceb50ad927b
#owner: https://api.github.com/users/fulcrum-blog

builder = SamBuilder()
(read1, read2) = builder.add_pair()
template = Template.build([read1, read2])
