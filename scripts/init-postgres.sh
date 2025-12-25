#!/bin/bash
set -e

# Create additional databases if POSTGRES_MULTIPLE_DATABASES is set
if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
  echo "Creating multiple databases: $POSTGRES_MULTIPLE_DATABASES"
  
  for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
    if [ "$db" != "$POSTGRES_DB" ]; then
      echo "Creating database: $db"
      psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE $db;
        GRANT ALL PRIVILEGES ON DATABASE $db TO $POSTGRES_USER;
EOSQL
    fi
  done
fi
