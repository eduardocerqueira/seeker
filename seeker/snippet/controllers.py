#date: 2023-02-27T16:46:35Z
#url: https://api.github.com/gists/504143c23a3d11b8c4ec2a62f440c78d
#owner: https://api.github.com/users/carlosm27

def update_book(title:str, author, pages_num, review, id:int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('UPDATE books SET title = %s, author=%s WHERE id = %s RETURNING *;', (title, author, id))
    book = cur.fetchone()[:]
    book_dict = to_dict(book)
    
    conn.commit()
    cur.close()
    conn.close()
  
    return  json.dumps(book_dict)