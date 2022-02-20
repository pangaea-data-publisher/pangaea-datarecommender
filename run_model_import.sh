#!/bin/bash

cd `dirname $0`/datarec_model
rm -f results/usage.json

set -e
echo "Running analysis..."
python3 main.py -c config/usage_prod.ini
if [ -f results/usage.json ]; then
	echo "Importing results into Elasticsearch..."
	../elasticsearch/import.sh results/usage.json
else
	echo "ERROR: No results file found, cancelled import to Elasticsearch!"
fi
