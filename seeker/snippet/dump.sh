#date: 2024-06-26T16:32:56Z
#url: https://api.github.com/gists/7f0130f395259ce0f47fdeff83123afb
#owner: https://api.github.com/users/vasilalexander

scp dump_26_06.custom root@65.21.191.235:~


sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo tee /usr/share/keyrings/postgresql-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/postgresql-archive-keyring.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
sudo apt update
sudo apt install postgresql-16
sudo /usr/pgsql-16/bin/postgresql-16-setup initdb
sudo systemctl enable postgresql-16
sudo systemctl start postgresql-16
sudo cp /root/dump_26_06.custom /tmp/
sudo chown postgres:postgres /tmp/dump_26_06.custom
sudo chmod 644 /tmp/dump_26_06.custom
sudo -i -u postgres
createdb demix3
pg_restore -d demix3 /root/dump_26_06.custom



pg_restore: error: could not execute query: ERROR:  extension "mysql_fdw" is not available
DETAIL:  Could not open extension control file "/usr/share/postgresql/16/extension/mysql_fdw.control": No such file or directory.
HINT:  The extension must first be installed on the system where PostgreSQL is running.
Command was: CREATE EXTENSION IF NOT EXISTS mysql_fdw WITH SCHEMA public;


pg_restore: error: could not execute query: ERROR:  extension "mysql_fdw" does not exist
Command was: COMMENT ON EXTENSION mysql_fdw IS 'Foreign data wrapper for querying a MySQL server';


pg_restore: error: could not execute query: ERROR:  extension "pldbgapi" is not available
DETAIL:  Could not open extension control file "/usr/share/postgresql/16/extension/pldbgapi.control": No such file or directory.
HINT:  The extension must first be installed on the system where PostgreSQL is running.
Command was: CREATE EXTENSION IF NOT EXISTS pldbgapi WITH SCHEMA public;


pg_restore: error: could not execute query: ERROR:  extension "pldbgapi" does not exist
Command was: COMMENT ON EXTENSION pldbgapi IS 'server-side support for debugging PL/pgSQL functions';


pg_restore: error: could not execute query: ERROR:  extension "postgis" is not available
DETAIL:  Could not open extension control file "/usr/share/postgresql/16/extension/postgis.control": No such file or directory.
HINT:  The extension must first be installed on the system where PostgreSQL is running.
Command was: CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


pg_restore: error: could not execute query: ERROR:  extension "postgis" does not exist
Command was: COMMENT ON EXTENSION postgis IS 'PostGIS geometry and geography spatial types and functions';


pg_restore: error: could not execute query: ERROR:  extension "postgis_topology" is not available
DETAIL:  Could not open extension control file "/usr/share/postgresql/16/extension/postgis_topology.control": No such file or directory.
HINT:  The extension must first be installed on the system where PostgreSQL is running.
Command was: CREATE EXTENSION IF NOT EXISTS postgis_topology WITH SCHEMA topology;


pg_restore: error: could not execute query: ERROR:  extension "postgis_topology" does not exist
Command was: COMMENT ON EXTENSION postgis_topology IS 'PostGIS topology spatial types and functions';







pg_restore: error: could not execute query: ERROR:  type "public.geometry" does not exist
LINE 12:     geom public.geometry(Point,4326),
                  ^
Command was: CREATE TABLE public.load_zip_codes (
    id integer NOT NULL,
    countrycode text,
    zip text,
    city text,
    state_id text,
    state_name text,
    created_at timestamp without time zone,
    updated_at timestamp without time zone,
    valid integer,
    zip_city text,
    geom public.geometry(Point,4326),
    name text NOT NULL,
    lat double precision,
    lng double precision
);







pg_restore: error: could not execute query: ERROR:  relation "public.load_zip_codes" does not exist
Command was: ALTER TABLE public.load_zip_codes OWNER TO postgres;

pg_restore: error: could not execute query: ERROR:  relation "public.load_zip_codes" does not exist
Command was: ALTER SEQUENCE public.load_zip_codes_id_seq OWNED BY public.load_zip_codes.id;


pg_restore: error: could not execute query: ERROR:  type "public.geometry" does not exist
LINE 14:     geom public.geometry(Point,4326),
                  ^
Command was: CREATE TABLE public.zip_codes (
    id integer NOT NULL,
    countrycode text,
    zip text,
    lat double precision NOT NULL,
    lng double precision NOT NULL,
    city text,
    state_id text,
    state_name text,
    created_at timestamp without time zone,
    updated_at timestamp without time zone,
    valid integer DEFAULT 0,
    zip_city text,
    geom public.geometry(Point,4326),
    name text
);


pg_restore: error: could not execute query: ERROR:  relation "public.zip_codes" does not exist
Command was: ALTER TABLE public.zip_codes OWNER TO postgres;

pg_restore: error: could not execute query: ERROR:  relation "public.zip_codes" does not exist
Command was: ALTER SEQUENCE public.zip_codes_id_seq OWNED BY public.zip_codes.id;


pg_restore: error: could not execute query: ERROR:  relation "public.load_zip_codes" does not exist
Command was: ALTER TABLE ONLY public.load_zip_codes ALTER COLUMN id SET DEFAULT nextval('public.load_zip_codes_id_seq'::regclass);


pg_restore: error: could not execute query: ERROR:  relation "public.zip_codes" does not exist
Command was: ALTER TABLE ONLY public.zip_codes ALTER COLUMN id SET DEFAULT nextval('public.zip_codes_id_seq'::regclass);


pg_restore: error: could not execute query: ERROR:  relation "public.load_zip_codes" does not exist
Command was: COPY public.load_zip_codes (id, countrycode, zip, city, state_id, state_name, created_at, updated_at, valid, zip_city, geom, name, lat, lng) FROM stdin;
pg_restore: error: could not execute query: ERROR:  relation "public.spatial_ref_sys" does not exist
Command was: COPY public.spatial_ref_sys (srid, auth_name, auth_srid, srtext, proj4text) FROM stdin;
pg_restore: error: could not execute query: ERROR:  relation "public.zip_codes" does not exist
Command was: COPY public.zip_codes (id, countrycode, zip, lat, lng, city, state_id, state_name, created_at, updated_at, valid, zip_city, geom, name) FROM stdin;
pg_restore: error: could not execute query: ERROR:  relation "topology.topology" does not exist
Command was: COPY topology.topology (id, name, srid, "precision", hasz) FROM stdin;
pg_restore: error: could not execute query: ERROR:  relation "topology.layer" does not exist
Command was: COPY topology.layer (topology_id, layer_id, schema_name, table_name, feature_column, feature_type, level, child_id) FROM stdin;
pg_restore: error: could not execute query: ERROR:  relation "topology.topology_id_seq" does not exist
LINE 1: SELECT pg_catalog.setval('topology.topology_id_seq', 1, fals...
                                 ^
Command was: SELECT pg_catalog.setval('topology.topology_id_seq', 1, false);


pg_restore: error: could not execute query: ERROR:  relation "public.load_zip_codes" does not exist
Command was: ALTER TABLE ONLY public.load_zip_codes
    ADD CONSTRAINT load_zip_codes_pkey PRIMARY KEY (id);


pg_restore: error: could not execute query: ERROR:  relation "public.load_zip_codes" does not exist
Command was: ALTER TABLE ONLY public.load_zip_codes
    ADD CONSTRAINT load_zip_codes_zip_key UNIQUE (zip);


pg_restore: error: could not execute query: ERROR:  relation "public.zip_codes" does not exist
Command was: ALTER TABLE ONLY public.zip_codes
    ADD CONSTRAINT zip_codes_pkey PRIMARY KEY (id);


pg_restore: error: could not execute query: ERROR:  relation "public.load_zip_codes" does not exist
Command was: CREATE INDEX idx_zip ON public.load_zip_codes USING btree (zip);


pg_restore: error: could not execute query: ERROR:  relation "public.zip_codes" does not exist
Command was: CREATE INDEX idx_zip_city ON public.zip_codes USING btree (zip_city);


pg_restore: warning: errors ignored on restore: 27
