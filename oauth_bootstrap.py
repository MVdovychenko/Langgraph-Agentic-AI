# oauth_bootstrap.py  (run on the HOST, not in Docker)
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
flow = InstalledAppFlow.from_client_secrets_file("secrets/client_secret.json", SCOPES)
creds = flow.run_local_server(port=0)  # opens browser
with open("secrets/token.json", "w") as f:
    f.write(creds.to_json())
print("✅ token.json written to secrets/")