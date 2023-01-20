#date: 2023-01-20T17:08:49Z
#url: https://api.github.com/gists/3e2e9d37f191d5eca65a3477de496ec0
#owner: https://api.github.com/users/theresa-hg

from cmatch.services.redshift_service import RedshiftService

redshift_service = RedshiftService()
df = redshift_service.read_from_redshift("select count(*) from ods.company_matching_owler_organizations_pii;")
print(df.head())