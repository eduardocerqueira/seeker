#date: 2024-02-05T16:55:54Z
#url: https://api.github.com/gists/ab66768562de1393d3bc6f6a4157c4f4
#owner: https://api.github.com/users/amotl

"""
Attempt to trip a low-memory condition in CrateDB, resulting in a weird error message
like `SQLParseException[ShardCollectContext for 0 already added]`.

This program tries to emulate the MLflow test case `test_search_runs_returns_expected_results_with_large_experiment`,
succeeded by a `DELETE FROM` table truncation operation.

Remark: It did not work out well. This program trips `OutOfMemoryError[Java heap space]`
        right away. Please use the MLflow test case reproducer demonstrated at:
        https://github.com/crate-workbench/mlflow-cratedb/issues/53#issuecomment-1927234463

Synopsis::

    docker run --rm -it --name=cratedb \
      --publish=4200:4200 --publish=5432:5432 \
      --env=CRATE_HEAP_SIZE=128m \
      crate/crate:nightly -Cdiscovery.type=single-node

    pip install --upgrade crate

    python heap_exchaust_wierd_error.py

"""
from crate import client
import uuid


def generate_large_data(nb_runs=1000):
    metrics_list = []
    for _ in range(nb_runs):
        run_id = str(uuid.uuid4())
        for i in range(100):
            metric = {
                "key": f"mkey_{i}",
                "value": i,
                "timestamp": i * 2,
                "step": i * 3,
                "is_nan": False,
                "run_uuid": run_id,
            }
            metric_record = list(metric.values())
            metrics_list.append(metric_record)
    return metrics_list


def load_into_database(dburi, data):
    #print("data:", data)
    with client.connect(dburi) as connection:
        cursor = connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS metrics;")

        cursor.execute("""
        CREATE TABLE metrics (
            "key" TEXT NOT NULL,
            "value" DOUBLE PRECISION NOT NULL,
            "timestamp" BIGINT NOT NULL,
            "step" BIGINT NOT NULL,
            "is_nan" BOOLEAN NOT NULL,
            "run_uuid" TEXT NOT NULL,
            PRIMARY KEY ("key", "value", "timestamp", "step", "is_nan", "run_uuid")
        );
        """)

        cursor.executemany("""
        INSERT INTO metrics
            (key, value, timestamp, step, is_nan, run_uuid)
        VALUES
            (?, ?, ?, ?, ?, ?);
        """, data)
        #cursor.execute("REFRESH TABLE metrics;")
        cursor.execute("DELETE FROM metrics;")

        #result = cursor.fetchall()
        #print("result:", result)
        cursor.close()


def trip_error(dburi):
    with client.connect(dburi) as connection:
        cursor = connection.cursor()
        cursor.execute("REFRESH TABLE metrics;")
        cursor.execute("DELETE FROM metrics;")
        result = cursor.fetchall()
        print("result:", result)
        cursor.close()


def main():
    dburi = "http://crate@localhost:4200"
    data = generate_large_data()
    load_into_database(dburi, data)
    #trip_error(dburi)



if __name__ == "__main__":
    main()
