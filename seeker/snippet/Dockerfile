#date: 2022-09-08T17:03:44Z
#url: https://api.github.com/gists/3d782500ee6c658d19ba03678072902a
#owner: https://api.github.com/users/igouss

# To build locally:
# docker build -t xm-database-postgress .
FROM eclipse-temurin:11 as JDK
# use the same version of flyway in build.gradle
FROM flyway/flyway:6.2.4 as flyway

# Replace with your base PG image here:
FROM XXXXXourBasePostgresImageXXXXXXXX:14.3 as database

ENV JAVA_HOME=/opt/jdk
ENV FLYWAY_HOME=/opt/flyway

ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PATH="${FLYWAY_HOME}:${PATH}"

USER postgres

COPY --from=JDK --chown=postgres:postgres /opt/java/openjdk $JAVA_HOME
COPY --from=flyway --chown=postgres:postgres /flyway $FLYWAY_HOME
COPY --chown=postgres:postgres flywayConf/* $FLYWAY_HOME/conf
COPY --chown=postgres:postgres init.sql /
COPY --chown=postgres:postgres src/main/resources/db/migration $FLYWAY_HOME/sql


# Delete data because `pg_ctl initdb` fails to start otherwise
RUN rm -rf /var/lib/postgresql/data
RUN mkdir /var/lib/postgresql/data && chown postgres:postgres /var/lib/postgresql/data

# Initalize `your_db_name` db with flyway
RUN pg_ctl initdb \
    && pg_ctl start  \
    && psql -U postgres -h localhost < /init.sql \
    && echo "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"" | psql -U your_db_name -w your_db_name -h localhost \
    && echo "CREATE EXTENSION IF NOT EXISTS hstore" | psql -U your_db_name -w your_db_name -h localhost \
    && echo "CREATE EXTENSION IF NOT EXISTS pg_trgm" | psql -U your_db_name -w your_db_name -h localhost \
    && echo "SELECT * FROM pg_extension" | psql -U your_db_name -w your_db_name -h localhost \
    && $FLYWAY_HOME/flyway -user=your_db_name -password=your_db_name -url=jdbc: "**********"://localhost:5432/your_db_name -placeholderPrefix="⁅⁅" -placeholderSuffix="⁆⁆"  -placeholders.dbName=your_db_name -placeholders.dbname=your_db_name migrate \
    && pg_ctl stop

#COPY --chown=postgres:postgres postgresql.conf /var/lib/postgresql/data/postgresql.conf
COPY --chown=postgres:postgres pg_hba.conf /var/lib/postgresql/data/pg_hba.conf

USER root
RUN  rm -rf $JAVA_HOME \
    && rm -rf $FLYWAY_HOME \
    && rm -rf /init.sql

