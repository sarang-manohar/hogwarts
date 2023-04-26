import os

class DefaultCreds:
    """ Bot Configuration """

    APP_GS_API_KEY = os.environ.get("API_KEY", "AIzaSyAAzu8U_tNbySGRkZy2n0VXnub21hn2gzA")
    APP_CLU_ENDPOINT = os.environ.get("CLU_ENDPOINT","https://sig788-task7-ls-006.cognitiveservices.azure.com")
    APP_CLU_KEY = os.environ.get("CLU_KEY","03dd4da6b78347dca783607c64bdf39e")
    APP_PROJECT_NAME = os.environ.get("PROJECT_NAME","sig788-task7-CLU-prj")
    APP_DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME","sig788-task7-dpl-001")