#date: 2023-04-25T16:58:40Z
#url: https://api.github.com/gists/f129ec1b8bb4de69e51afb8b50ee9471
#owner: https://api.github.com/users/tensaye-o

def generate_sql_file(filename, sql_generators):
    sql_statements = []
    delete_inserted = set()

    for sql_gen_func, params in sql_generators:
        sql_statement = sql_gen_func(*params)
        sql_statements.append(sql_statement)

        if sql_gen_func in func_to_delete and sql_gen_func not in delete_inserted:
            delete_func = func_to_delete[sql_gen_func]
            param_index = param_index_for_delete[sql_gen_func]
            delete_statement = delete_func(params[param_index])
            sql_statements.insert(0, delete_statement)
            delete_inserted.add(sql_gen_func)

    sql_script = "\n\n".join(sql_statements)

    with open(filename, "w") as f:
        f.write(sql_script)


def insert_to_users(name, age, id):
    return f"""--users
INSERT INTO users (id, name, age) 
VALUES ({id}, '{name}', {age});"""


def delete_users(id):
    return f"DELETE FROM users WHERE id = {id};"


def insert_to_todos(user_id):
    return f"""--todos
INSERT INTO todos (id, user_id, title, is_completed) 
VALUES (1, {user_id}, 'Buy Coke', FALSE);"""


OUTPUT_FILENAME = "output.sql"
GLOBAL_ID = 1

func_to_delete = {
    insert_to_users: delete_users
}

param_index_for_delete = {
    insert_to_users: 2,
}

sql_generators = [
    (insert_to_users, ('Tensaye Yuan', 27, 1)),
    (insert_to_todos, (1,))
]

generate_sql_file(OUTPUT_FILENAME, sql_generators)
