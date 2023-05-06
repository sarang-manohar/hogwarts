import os

class DefaultCreds:
    """ Bot Configuration """

    APP_GS_API_KEY = os.environ.get("API_KEY", "AIzaSyAAzu8U_tNbySGRkZy2n0VXnub21hn2gzA")
    APP_CLU_ENDPOINT = os.environ.get("CLU_ENDPOINT","https://sig788-task7-ls-006.cognitiveservices.azure.com")
    APP_CLU_KEY = os.environ.get("CLU_KEY","03dd4da6b78347dca783607c64bdf39e")
    APP_PROJECT_NAME = os.environ.get("PROJECT_NAME","sig788-task7-CLU-prj")
    APP_DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME","sig788-task7-dpl-001")

    # Cognitive Services Endpoint
    CV_PRED_URL = os.environ.get("CV_PRED_URL","https://sig788task7cv001-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/311c11df-ab0d-4721-b3b8-3d7a8e4fa57a/detect/iterations/Iteration1/image")
    CV_PRED_KEY = os.environ.get("CV_PRED_KEY","3d8002fdfe7c4b689b485f4b977a65a2")