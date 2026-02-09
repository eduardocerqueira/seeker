#date: 2026-02-09T17:43:16Z
#url: https://api.github.com/gists/d8065dff8b45fef33f8c19d4ce9dda18
#owner: https://api.github.com/users/fivetran-chinmayichandrasekar

import psycopg2

def probe_postgres(host: "**********": int, dbname: str, user: str, password: str, sslmode: str = "require") -> None:
    """
    Custom helper: opens a short-lived DB connection and runs SELECT 1.
    Fails fast on auth/connection problems.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password= "**********"
            sslmode=sslmode,
            connect_timeout=10,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
    except psycopg2.OperationalError as exc:
        # OperationalError covers many auth + connectivity failures.
        # Keep the message concise; don’t echo secrets.
        msg = str(exc).splitlines()[3] if str(exc) else "OperationalError"
        raise RuntimeError(f"DB_CONNECTION_OR_AUTH_FAILED: {msg}") from exc
    finally:
        if conn:
            conn.close()
