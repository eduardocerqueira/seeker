#date: 2023-03-03T17:08:55Z
#url: https://api.github.com/gists/c407d3f918d01606d228f1d0c3524bd0
#owner: https://api.github.com/users/maxwellbade

def fraud_rate(df, agg, cols=None, threshold=None, limit=None, days=None, minrate=None, maxrate=None):
    if isinstance(cols, str):
        groupcols = [cols]
    elif cols is None:
        groupcols = [] 
    else:
        try:
            groupcols = list(cols)
        except:
            raise TypeError('Unable to convert cols to a list.')
    fraudcols = groupcols + ['is_risky']
    cnts = df.groupby(fraudcols)[agg].nunique()
    if groupcols:
        wide = cnts.unstack('is_risky').fillna(0)
    else:
        wide = cnts.to_frame().stack(level=0).unstack('is_risky').fillna(0)
    wide = wide.assign(risky_rate = wide.risky / (wide.risky + wide.not_risky))
    wide = wide.assign(total_users = wide.risky + wide.not_risky)
    wide = wide.assign(fpr=wide.not_risky / wide.risky)
    if threshold:
        wide = wide.loc[wide['not_risky'] >= (threshold)]
    if limit:
        wide = wide.head(n=limit)
    if days:
        wide = wide.assign(daily_impact = (wide.risky + wide.not_risky) / days)
    if minrate:
        wide = wide[wide['risky_rate'] >= minrate]
    if maxrate:
        wide = wide[wide['risky_rate'] <= maxrate]
    wide = wide.sort_values(['risky_rate','risky'], ascending=False)
    return wide

def multiple_defect(df, agg, cols, n, count_risky=0, minrate=0):
    combos = list(itertools.combinations(cols, n))
    num_cols = ['col' + str(i) for i in range(1, n + 1)]
    res = []
    for combo in combos:
        fr = fraud_rate(df, agg, combo)
        fr = fr.reset_index()
        fr = fr.assign(groupcols = ', '.join(combo)) #groupcols
        fr = fr[fr['risky'] >= count_risky]
        # fr = fr[fr['risky'] >= count_non_risky]
        fr = fr.loc[fr['risky_rate'] >= (minrate)] #filter
        # fr = fr.loc[fr['risky_rate'] <= (maxrate)] #filter
        res.append(fr)
    return pd.concat(res).sort_values(by=['risky'], ascending=False)
def unique_cols(which_df,exclude_cols,pct_lt_cols,some_id):
    cols = []
    exclude = exclude_cols
    nunique = []
    pct_less = df[some_id].nunique() - (df[some_id].nunique() * pct_lt_cols)

    for c in which_df.columns:
        cols.append(c)
        nunique.append(which_df[c].nunique())
    df_cols = pd.DataFrame(nunique)
    df_cols['cols'] = cols
    df_cols.columns = ['nunique','column']
    df_cols = df_cols[['column','nunique']]
    df_cols_non_unique = df_cols[
        (df_cols['nunique'] <= pct_less)
        & (df_cols['nunique'] > 1)
        & (~df_cols['column'].isin(exclude))
    ].sort_values(by='nunique',ascending=True)
    print(df_cols_non_unique.shape)
    return df_cols_non_unique

interesting_cols = unique_cols(df
                   ,exclude_cols = ['is_risky'
                                    ,'listing_state'
                                    ,'is_softblocked'
                                    ,'fraud_listing_id'
                                    ,'item_create_date','row_number','target'
                                    ,'ua_string','signup_date','user_id']
                   ,pct_lt_cols=.5
                   ,some_id='listing_id'
                  )
interesting_cols_list = list(interesting_cols['column'])
interesting_cols_list = [ x for x in interesting_cols_list if 'date' not in x ]
print(interesting_cols_list,'\n')