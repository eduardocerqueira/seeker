#date: 2022-11-04T16:55:15Z
#url: https://api.github.com/gists/d95f9594668b9ae4791cedd4716bd42f
#owner: https://api.github.com/users/Polaris000

    conf_df = pd.DataFrame()
    conf_df["conf"] = y_conf
    conf_df["expected"] = y_true
    conf_df.columns = ["conf", "expected"]
    conf_df = conf_df.sort_values("conf", ascending=False)