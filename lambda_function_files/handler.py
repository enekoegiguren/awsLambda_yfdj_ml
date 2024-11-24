from datetime import datetime, date
import os
import re
import pickle

import pandas as pd

from utils.credentials import *
from utils.train_model import *



import logging


# Set up logging
logging.basicConfig(level=logging.INFO)

def handler(event, context):
    try:
        client_id, client_secret = get_secret()
        
        train_and_load_model()
        return {
            'statusCode': 200,
            'body': json.dumps('Function executed successfully!')
        }

    except Exception as e:

        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }