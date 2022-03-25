#date: 2022-03-25T16:56:49Z
#url: https://api.github.com/gists/ff5d1cb94efc0156807f013d7c569057
#owner: https://api.github.com/users/harpreetsahota204

cc_df.loc[(cc_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = cc_df['MINIMUM_PAYMENTS'].median()
cc_df = cc_df[cc_df['CREDIT_LIMIT'].isnull() == False]