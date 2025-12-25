#date: 2025-12-25T17:11:27Z
#url: https://api.github.com/gists/ee30f1d7e4e245f7f964e460775d1045
#owner: https://api.github.com/users/9Mad-Max5

# =======================
# Plex play-queue schema reset (Unraid/Docker)
# container: plexmediaserver
# appdata:   /mnt/user/appdata/plexmediaserver
# originally created by: Splicing1524
# can be found here https://pastebin.com/8hxVkjpi
# slight updates by: 9Mad-Max5
# =======================
 
set -euo pipefail

# Here you need to make your adjustments
# It needs to be run in the host of your docker host
CN="plex" # The name of your plex container
CONFIG="/container/Media-Managment/Plex/config" # Path to the files of your docker container
DB_HOST="$CONFIG/Library/Application Support/Plex Media Server/Plug-in Support/Databases/com.plexapp.plugins.library.db"
# If a different user is used like in the linuxserver env you need to set it here
PUID=911
PGID=911


echo "==> Stopping container: $CN"
docker stop "$CN"
 
echo "==> Backing up DB on host"
ts="$(date +%F_%H%M)"
cp -v "$DB_HOST" "${DB_HOST}.bak.${ts}"
 
echo "==> Finding image used by $CN"
IMAGE="$(docker inspect -f '{{.Config.Image}}' "$CN")"
echo "    -> $IMAGE"
 
echo "==> Running SQL using Plex's bundled SQLite in a helper container (Plex remains stopped)"
docker run --rm \
  -e PUID=$PUID \
  -e PGID=$PGID \
  -v "$CONFIG":/config \
  "$IMAGE" \
  bash -lc '
set -euo pipefail
PXS="/usr/lib/plexmediaserver/Plex SQLite"
[ -x "$PXS" ] || PXS="/usr/lib/plexmediaserver/Resources/Plex SQLite"
if [ ! -x "$PXS" ]; then
  echo "ERROR: Could not find Plex SQLite inside image." >&2
  ls -l /usr/lib/plexmediaserver || true
  exit 1
fi
 
DB="/config/Library/Application Support/Plex Media Server/Plug-in Support/Databases/com.plexapp.plugins.library.db"
echo "Plex SQLite at: $PXS"
echo "Target DB: $DB"
ls -l "$DB"
 
"$PXS" "$DB" <<SQL
DROP TABLE IF EXISTS play_queue_items;
DROP TABLE IF EXISTS play_queue_generators;
DROP TABLE IF EXISTS play_queues;
 
CREATE TABLE play_queue_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  play_queue_id integer,
  metadata_item_id integer,
  "order" float,
  up_next boolean,
  play_queue_generator_id integer
);
 
CREATE TABLE play_queues (
  id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  client_identifier varchar(255),
  account_id integer,
  playlist_id integer,
  sync_item_id integer,
  play_queue_generator_id integer,
  generator_start_index integer,
  generator_end_index integer,
  generator_items_count integer,
  generator_ids blob,
  seed integer,
  current_play_queue_item_id integer,
  last_added_play_queue_item_id integer,
  version integer,
  created_at integer,
  updated_at integer,
  metadata_type integer,
  total_items_count integer,
  generator_generator_ids blob,
  extra_data varchar(255)
);
 
CREATE TABLE play_queue_generators (
  id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  playlist_id integer,
  metadata_item_id integer,
  uri varchar(255),
  "limit" integer,
  continuous boolean,
  "order" float,
  created_at integer NOT NULL,
  updated_at integer NOT NULL,
  changed_at integer DEFAULT 0,
  recursive boolean,
  type integer,
  extra_data varchar(255)
);
 
CREATE INDEX index_play_queue_items_on_play_queue_id ON play_queue_items (play_queue_id);
CREATE INDEX index_play_queue_items_on_metadata_item_id ON play_queue_items (metadata_item_id);
CREATE INDEX index_play_queue_items_on_order ON play_queue_items ("order");
CREATE INDEX index_play_queues_on_account_id ON play_queues (account_id);
CREATE INDEX index_play_queues_on_playlist_id ON play_queues (playlist_id);
CREATE INDEX index_play_queues_on_sync_item_id ON play_queues (sync_item_id);
CREATE INDEX index_play_queue_generators_on_playlist_id ON play_queue_generators (playlist_id);
CREATE INDEX index_play_queue_generators_on_metadata_item_id ON play_queue_generators (metadata_item_id);
CREATE INDEX index_play_queue_generators_on_order ON play_queue_generators ("order");
CREATE UNIQUE INDEX index_play_queues_on_client_identifier_and_account_id_and_metadata_type
  ON play_queues (client_identifier, account_id, metadata_type);
CREATE INDEX index_play_queue_generators_on_changed_at ON play_queue_generators (changed_at);
SQL
'
 
# Optional: normalize DB file mode/owner on the host
# If you use linuxserver/plex (PUID=99/PGID=100), these are typically right already.
# Uncomment if you need to force them:
# chown nobody:users "$DB_HOST"
chmod 664 "$DB_HOST"
 
echo "==> Starting container: $CN"
docker start "$CN"
 
echo "==> Done. If Plex Web was open, refresh it after ~30–60s."