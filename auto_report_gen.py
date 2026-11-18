# auto_report_gen.py
#
# See https://duckdblab.org/en/post/duckdb-connect-uri-ephemeral-attach/
# Scenario 3: Automated Report Generation
#
import duckdb
import pandas
conn = duckdb.connect(':memory:')
conn.execute('load quack')

# One-time connection to remote database for data retrieval
result = conn.execute("""
 CONNECT 'quack:localhost' (TOKEN 'example-token');
 SELECT * FROM photos WHERE file_mtime < CURRENT_DATE - INTERVAL '31' DAY;
 DISCONNECT;
""").fetchdf()

# Local processing
print(result.describe())
