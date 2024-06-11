#date: 2024-06-11T16:58:59Z
#url: https://api.github.com/gists/caec9c5d3fea4e7622e0ccecf34d0af9
#owner: https://api.github.com/users/mojtaba42

def prepare_cases(cve_id, cve_description):
    return [
        f"This is a CVE {cve_id} detail: {cve_description}. Reply by only returning yes or no, would this affect me if I have a debian based container in aws ecs with a python web application that does not terminate ssl has this vulnerability.",
        f"This is a CVE {cve_id} detail: {cve_description}. Reply by only returning yes or no, would this affect me if I have an ubuntu based container running a nodejs script on events from a queue inside aws fargate have this vulnerability.",
        f"This is a CVE {cve_id} detail: {cve_description}. Reply by only returning yes or no, would this affect me if I have a python aws lambda behind an api gateway application that have this vulnerability",
        f"This is a CVE {cve_id} detail: {cve_description}. Reply by only returning yes or no, would this affect me if I have an ubuntu based ec2 machine hosting a nodejs web app have this vulnerability",
        f"This is a CVE {cve_id} detail: {cve_description}. Reply by only returning yes or no, would this affect me if I have an ubuntu based container running CI scripts that provisions AWS resources have this vulnerability",
    ]