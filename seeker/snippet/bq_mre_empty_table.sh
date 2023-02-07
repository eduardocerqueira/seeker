#date: 2023-02-07T16:43:52Z
#url: https://api.github.com/gists/5736fa6232c4926bcc727a37c2d435fa
#owner: https://api.github.com/users/mmngreco

bq query --format=sparse --nouse_legacy_sql '
    WITH temporal AS (
      --   | x      | y            | z        |
      SELECT 1 AS x , "foo"   AS y , true  AS z UNION ALL
      SELECT 2 AS x , "apple" AS y , false AS z UNION ALL
      SELECT 3 AS x , ""      AS y , true  AS z
    ),
    staging as (
      SELECT 1       AS x    , "boo"   AS y   , true  AS z UNION ALL
      SELECT 2       AS x    , "apple" AS y   , false AS z UNION ALL
      SELECT 3       AS x    , ""      AS y   , true  AS z UNION ALL
      SELECT 4       AS x    , "yay"   AS y   , NULL  AS z
      -- SELECT NULL AS x    , NULL    AS y   , NULL  AS z
    )
    SELECT
    count(*)
    FROM temporal
    WHERE (x,y,z) not in (select (x,y,z) from staging)
'




bq query --format=sparse --nouse_legacy_sql '
    WITH temporal AS (
      SELECT NULL AS x , "foo"   AS y , true  AS z UNION ALL
      SELECT 1 AS x , "foo"   AS y , true  AS z UNION ALL
      SELECT 2 AS x , "apple" AS y , false AS z UNION ALL
      SELECT 3 AS x , ""      AS y , true  AS z
    ),
    staging as (
      SELECT 1       AS x    , "boo"   AS y   , true  AS z UNION ALL
      SELECT 2       AS x    , "apple" AS y   , false AS z UNION ALL
      SELECT 3       AS x    , ""      AS y   , true  AS z UNION ALL
      SELECT 4       AS x    , "yay"   AS y   , NULL  AS z
      -- SELECT NULL AS x    , NULL    AS y   , NULL  AS z
    )
    SELECT
        *
    FROM temporal
    WHERE (
        IFNULL(CAST(x as STRING), ""), 
        IFNULL(CAST(y as STRING), ""),
        IFNULL(CAST(z as string), "")
        ) not in (select (
        IFNULL(CAST(x as STRING), ""), 
        IFNULL(CAST(y as STRING), ""),
        IFNULL(CAST(z as string), "")
    ) from staging)
'


bq query --format=sparse --nouse_legacy_sql '
    WITH temporal AS (
      SELECT NULL AS x , NULL   AS y , NULL AS z UNION ALL
      SELECT 1 AS x , "foo"   AS y , true  AS z UNION ALL
      SELECT 2 AS x , "apple" AS y , false AS z UNION ALL
      SELECT 3 AS x , ""      AS y , true  AS z
    ),
    staging as (
      SELECT 1       AS x    , "boo"   AS y   , true  AS z UNION ALL
      SELECT 2       AS x    , "apple" AS y   , false AS z UNION ALL
      SELECT 3       AS x    , ""      AS y   , true  AS z UNION ALL
      SELECT 4       AS x    , "yay"   AS y   , NULL  AS z
      -- SELECT NULL AS x    , NULL    AS y   , NULL  AS z
    )
    SELECT
        temporal.*
    FROM temporal
    LEFT JOIN staging
    using (x,y,z)
    where staging.x is null
    and staging.y is null
    and staging.z is null
'

