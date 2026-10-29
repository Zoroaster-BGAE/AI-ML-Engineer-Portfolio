"""Seed the local Cognito emulator (moto) with a pool, an app client, the three groups and one user per role.

    python -m app.cli.seed_local_cognito        (run by docker-compose's `cognito-seed` service)

LOCAL DEVELOPMENT ONLY: it refuses to run unless COGNITO_ENDPOINT is set and the database is local. moto keeps
state in memory, so this runs again whenever the emulator restarts; it is idempotent.

moto is started with the HASH id strategies, so the pool and client ids depend only on their names and are the
same on every start. docker-compose hard-codes them; if a moto upgrade ever changes them, this fails loudly and
prints the new ids to paste into docker-compose.yml.

The passwords below are test values for the emulator, which exists only on a developer machine.
"""
from __future__ import annotations

import sys

import boto3

from app.config import settings

POOL_NAME = "local-pool"
CLIENT_NAME = "local-frontend"
PASSWORD = "Local-dev-1!"
USERS = [  # (email, group, reviewer flag is set in the app, not here)
    ("admin@local.test", "admin"),
    ("tech@local.test", "technician"),
    ("lead@local.test", "technician"),   # flag as reviewer in Admin > Users to test the lead-technician flow
    ("customer@local.test", "customer"),
]


def main() -> None:
    if not settings.cognito_endpoint:
        sys.exit("seed_local_cognito: COGNITO_ENDPOINT is not set; this only seeds a local emulator")
    settings.assert_emulators_are_local()
    c = boto3.client("cognito-idp", region_name=settings.cognito_region, endpoint_url=settings.cognito_endpoint)

    pool = next((p for p in c.list_user_pools(MaxResults=60)["UserPools"] if p["Name"] == POOL_NAME), None)
    pool_id = pool["Id"] if pool else c.create_user_pool(
        PoolName=POOL_NAME, UsernameAttributes=["email"], AutoVerifiedAttributes=["email"])["UserPool"]["Id"]

    clients = c.list_user_pool_clients(UserPoolId=pool_id, MaxResults=60)["UserPoolClients"]
    client = next((x for x in clients if x["ClientName"] == CLIENT_NAME), None)
    client_id = client["ClientId"] if client else c.create_user_pool_client(
        UserPoolId=pool_id, ClientName=CLIENT_NAME, GenerateSecret=False,
        ExplicitAuthFlows=["ALLOW_USER_PASSWORD_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"])["UserPoolClient"]["ClientId"]

    groups = {g["GroupName"] for g in c.list_groups(UserPoolId=pool_id)["Groups"]}
    for g in ("admin", "technician", "customer"):
        if g not in groups:
            c.create_group(UserPoolId=pool_id, GroupName=g)

    existing = {a["Value"] for u in c.list_users(UserPoolId=pool_id)["Users"]
                for a in u.get("Attributes", []) if a["Name"] == "email"}
    for email, group in USERS:
        if email not in existing:
            c.admin_create_user(UserPoolId=pool_id, Username=email, MessageAction="SUPPRESS",
                                UserAttributes=[{"Name": "email", "Value": email},
                                                {"Name": "email_verified", "Value": "true"}])
            c.admin_set_user_password(UserPoolId=pool_id, Username=email, Password=PASSWORD, Permanent=True)
            c.admin_add_user_to_group(UserPoolId=pool_id, Username=email, GroupName=group)

    print(f"seeded {POOL_NAME}: pool={pool_id} client={client_id} users={[u for u, _ in USERS]}")
    if (settings.cognito_user_pool_id, settings.cognito_client_id) != (pool_id, client_id):
        sys.exit("seed_local_cognito: the emulator's ids differ from COGNITO_USER_POOL_ID / COGNITO_CLIENT_ID. "
                 f"Set them to pool={pool_id} client={client_id} in docker-compose.yml.")


if __name__ == "__main__":
    main()
