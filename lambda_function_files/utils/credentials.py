import boto3
import json
from botocore.exceptions import ClientError


def get_secret():
    secret_name = "france_travail"
    region_name = "eu-west-3"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # Raise the exception for any error that occurs
        raise e

    # Extract the secret string
    secret_string = get_secret_value_response.get('SecretString')
    
    if secret_string:
        # Parse the secret string as JSON
        secret_dict = json.loads(secret_string)

        # Extract client_id and client_secret
        client_id = secret_dict.get('client_id')
        client_secret = secret_dict.get('client_secret')

        return client_id, client_secret
    else:
        raise ValueError("Secret is not in the expected format.")


