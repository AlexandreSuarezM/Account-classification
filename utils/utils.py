from google.cloud import bigquery
from google.oauth2 import service_account
import os 
from dotenv import load_dotenv 

load_dotenv ()

PROJECT_ID = os.getenv("PROJECT_ID")

def bigquery(query):
    credentials = service_account.Credentials.from_service_account_file(filename='credentials/insights-credentials.json')
    client = bigquery.Client(
        credentials = credentials, 
        project = PROJECT_ID)

    query_job = client.query(query)
    rows = query_job.result() 
    results = [dict(row) for row in rows]
    return results