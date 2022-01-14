#date: 2022-01-14T16:59:54Z
#url: https://api.github.com/gists/ccf98141448a81afb3b09e729b6afcd7
#owner: https://api.github.com/users/danko1521

import sqlalchemy

URI = 'postgresql+psycopg2://danko1521:795364@localhost/musical_db'
engine = sqlalchemy.create_engine(URI)
connection = engine.connect()
print(connection)

sel = connection.execute("""SELECT g.genre_name, count(a.performer) 
FROM genre as g
left join artist_genre as ag on g.id = ag.genre_id
left join artist as a on ag.artist_id = a.id
group by g.genre_name
order by count(a.id) DESC;
""").fetchall()
print(sel)

sel2 = connection.execute("""SELECT s.name, a.date_released  
FROM album as a
left join songs as s on s.album_id = a.id
where (a.date_released >= 2019) and (a.date_released <= 2020)
""").fetchall()
print(sel2)

sel3 = connection.execute("""SELECT a.name, AVG(s.duration)
FROM album as a
left join songs as s on s.album_id = a.id
group by a.name
order by AVG(s.duration)
""").fetchall()
print(sel3)

sel4 = connection.execute("""SELECT distinct a.performer
FROM artist as a 
where a.performer not in (
    select distinct a.performer
    from artist as a
    left join album_artist as aa on a.id = aa.performer_id
    left join album as ab on ab.id = aa.album_id
    where ab.date_released = 2020
)
order by a.performer
""").fetchall()
print(sel4)

sel5 = connection.execute("""SELECT distinct c.name
FROM collection as c 
left join collection_track as ct on c.id = ct.collections_id
left join songs as s on s.id = ct.songs_id
left join album as ab on ab.id = s.album_id 
left join album_artist as aa on aa.album_id = ab.id
left join artist as art on art.id = aa.performer_id
where art.performer like '%%ooes%%'
order by c.name
""").fetchall()
print(sel5)

sel6 = connection.execute("""SELECT ab.name 
FROM album as ab
left join album_artist as aa on ab.id = aa.album_id
left join artist as a on a.id = aa.performer_id
left join artist_genre as ag on a.id = ag.artist_id
left join genre as g on g.id = ag.genre_id
group by ab.name
having count(distinct g.genre_name) > 1
order by ab.name
""").fetchall()
print(sel6)

sel7 = connection.execute("""SELECT s.name
FROM songs as s
left join collection_track as ct on s.id = ct.songs_id
where ct.songs_id is null
""").fetchall()
print(sel7)

sel8 = connection.execute("""SELECT a.performer, s.duration 
FROM songs as s 
left join album as ab on ab.id = s.album_id
left join album_artist as aa on aa.album_id = ab.id
left join artist as a on a.id = aa.performer_id
group by a.performer, s.duration
having s.duration = (select min(duration) from songs)
order by a.performer
""").fetchall()
print(sel8)

sel9 = connection.execute("""SELECT distinct ab.name 
FROM album as ab 
left join songs as s on s.album_id = ab.id
where s.album_id in (
    select album_id
    from songs
    group by album_id 
    having count(id) = (
        select count(id)
        from songs 
        group by album_id
        order by count 
        limit 1
    )
)
order by ab.name
""").fetchall()
print(sel9)