import os
import google.auth
from google.auth.transport.requests import Request
from google.auth import default
from google.auth import exceptions
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession
from google.auth import impersonated_credentials
from google.auth.transport import requests

from google.cloud import aiplatform

# Function to authenticate and get the credentials
def authenticate_with_google_cloud():
    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            # If credentials are not valid and cannot be refreshed, open the browser for authentication
            flow = google.auth.oauth2client.OAuth2WebServerFlow(
                client_id=os.getenv("GOOGLE_CLIENT_ID"),
                client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
                scope="https://www.googleapis.com/auth/cloud-platform",
                redirect_uri="urn:ietf:wg:oauth:2.0:oob"
            )
            auth_uri = flow.step1_get_authorize_url()
            print("Please go to this URL: {}".format(auth_uri))
            auth_code = input("Enter the authorization code: ")
            credentials = flow.step2_exchange(auth_code)
    
    return credentials, project

# Function to export Vertex AI data to a .txt file
def export_vertex_ai_data():
    # Authenticate and get the credentials
    credentials, project_id = authenticate_with_google_cloud()

    # Initialize the AI Platform client
    aiplatform.init(project=project_id, credentials=credentials)

    # Define the file to save the data
    export_file = "vertex_ai_data.txt"

    with open(export_file, "w") as file:
        # Example: List all datasets in the project
        datasets = aiplatform.Dataset.list()
        file.write("Datasets:\n")
        for dataset in datasets:
            file.write(f"Name: {dataset.name}, Display Name: {dataset.display_name}\n")

        # Example: List all models in the project
        models = aiplatform.Model.list()
        file.write("\nModels:\n")
        for model in models:
            file.write(f"Name: {model.name}, Display Name: {model.display_name}\n")

        # Example: List all endpoints in the project
        endpoints = aiplatform.Endpoint.list()
        file.write("\nEndpoints:\n")
        for endpoint in endpoints:
            file.write(f"Name: {endpoint.name}, Display Name: {endpoint.display_name}\n")

    print(f"Data exported to {export_file}")

if __name__ == "__main__":
    export_vertex_ai_data()
