#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978

    #APP_TYPE = os.environ.get("MicrosoftAppType", "MultiTenant")
    
    # Enter details below
    #APP_ID = os.environ.get("MicrosoftAppId", "276b5d9f-de43-4124-bfe9-9d91ce16a7dd")
    #APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "sig788-task7-identity-password")

    APP_TYPE = os.environ.get("MicrosoftAppType", "web")
    
    # Enter details below
    APP_ID = os.environ.get("MicrosoftAppId", "")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "")
