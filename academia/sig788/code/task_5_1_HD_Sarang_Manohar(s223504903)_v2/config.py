#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978

    APP_TYPE = os.environ.get("MicrosoftAppType", "")
    
    # Enter details below
    #APP_ID = os.environ.get("MicrosoftAppId", "7e549871-41a3-4163-af83-5546a9679c54")
    APP_ID = os.environ.get("MicrosoftAppId", "")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "")
    
