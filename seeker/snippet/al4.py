#date: 2022-06-22T17:13:16Z
#url: https://api.github.com/gists/fd27b8538bfdf6fded133d1958109d18
#owner: https://api.github.com/users/peytonrunyan

[ins] In [37]: print(Base.metadata.tables)
Out[37]: FacadeDict(
  {
    'dogs': Table(
      'dogs', 
      MetaData(), 
      Column(
        'id', Integer(), table=<dogs>, primary_key=True, nullable=False
      ), 
      Column(
        'name', Text(), table=<dogs>, nullable=False, default=ColumnDefault('Sparky')
      ), 
      schema=None
    )
  }
)