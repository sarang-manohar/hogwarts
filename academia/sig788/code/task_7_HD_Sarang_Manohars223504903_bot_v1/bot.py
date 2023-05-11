from botbuilder.core import ActivityHandler, TurnContext, MessageFactory
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes, CardAction, ActionTypes, SuggestedActions

import requests
import cv2
import numpy as np


from creds import DefaultCreds
CREDS = DefaultCreds()


class MyBot(ActivityHandler):
    def __init__(self):
        self.API_KEY = CREDS.APP_GS_API_KEY
        
                
    async def on_message_activity(self, turn_context: TurnContext):
    # Check if the user sent an image attachment
        if turn_context.activity.attachments:
            for attachment in turn_context.activity.attachments:
                # Check if the attachment is an image
                if attachment.content_type.startswith('image/'):

                    image_data = requests.get(attachment.content_url).content
                                       
                    # Define a ConversationRequest object to send the image to Google Search API
                    dtct_obj = await self.detect_objects(image_data)

                    result = await self.recommend_products("buy {}".format(dtct_obj))

                    # Send result back to user
                    reply_activity = Activity(
                        type=ActivityTypes.message,
                        text=result
                    )
                    print("Formatted query: ", result)
                    await turn_context.send_activity(reply_activity)
                    await turn_context.send_activity("Would you like to provide additional details about your product, like brand, price range, intended use or purpose?")
                    
        # If the user did not send an image, process the message as text
        else:
            # Get the user's message from the turn context
            user_input = turn_context.activity.text

            # Format query
            formatted_query = await self.format_query(user_input)   

            # Call recommend_products function to get recommended products
            result = await self.recommend_products(formatted_query)

            # Send result back to user
            reply_activity = Activity(
                type=ActivityTypes.message,
                text=result
            )
            print("Formatted query: ", formatted_query)
            await turn_context.send_activity(reply_activity)
            await turn_context.send_activity("If you would like me to help you with recommendation, please share specific details or ask me about another product.")
    # Define function to recommend products
    async def recommend_products(self, formatted_query):
        # Set API endpoint and parameters
        text_search_url = f'https://www.googleapis.com/customsearch/v1?key={self.API_KEY}&cx=b731f8a663d17422e&cr=countryIN&q={formatted_query}&num=5&safe=active'

        # Send GET request to API endpoint
        response = requests.get(text_search_url)

        # Check if request was successful
        if response.status_code == 200:
            # Parse JSON response and extract product information
            data = response.json()
            products = data.get("items", [])
            if not products:
                return "Sorry, I couldn't find any products that match your query."

            # Debugging code: print out JSON response
            print(data)

            product_names = []
            product_URL = []
            for product in products:
                if "title" in product:
                    product_names.append(product["title"])
                else:
                    product_names.append("Unknown product")
                
                if "formattedUrl" in product and product["formattedUrl"]:
                    product_URL.append(product["formattedUrl"])
                else:
                    product_URL.append("Unknown URL")

            # Format product information as a string and return
            result = "Here are some products I recommend:\n"
            for i in range(len(products)):
                result += f"{i+1}. {product_names[i]} - {product_URL[i]}\n"
            return result
        else:
            return "Sorry, I couldn't find any resources that match your query."
    

# Send query to Azure Conversation Language Understanding API
    async def format_query(self, query):
    # import libraries
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.language.conversations.aio import ConversationAnalysisClient

        # create client
        async with ConversationAnalysisClient(CREDS.APP_CLU_ENDPOINT, AzureKeyCredential(CREDS.APP_CLU_KEY)) as client:
            # send query
            result = await client.analyze_conversation(
                task={
                    "kind": "Conversation",
                    "analysisInput": {
                        "conversationItem": {
                            "participantId": "1",
                            "id": "1",
                            "modality": "text",
                            "language": "en",
                            "text": query
                        },
                        "isLoggingEnabled": False
                    },
                    "parameters": {
                        "projectName": CREDS.APP_PROJECT_NAME,
                        "deploymentName": CREDS.APP_DEPLOYMENT_NAME,
                        "verbose": True
                    }
                }
            )

        keyword_list = []
        for entity in result['result']['prediction']['entities']:
            keyword_list.append(entity['text'])
            keyword_list.append(" ")

        formatted_query = ''.join(keyword_list)

        return formatted_query

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome! What would you like to do today?")


    # Set up the Custom Vision API call
    async def detect_objects(self, image_data):
        image = np.asarray(bytearray(image_data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Perform object detection using OpenCV
        # Replace this with your own object detection code
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

        # Compress the image as a JPEG image
        _, encoded_frame = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        headers = {
            "Prediction-Key": CREDS.CV_PRED_KEY,
            "Content-Type": 'application/json',
        }

        body = {
        "Data": img.tobytes(),
        }
        
        response = requests.post(CREDS.CV_PRED_URL, headers=headers, json=body, data=encoded_frame.tobytes())
        response.raise_for_status()
        predictions = response.json()["predictions"]

        if response.status_code == requests.codes.ok:
            query = self.detected_objects_2_query(predictions)
            return query
        else:
            return None

    def detected_objects_2_query(self, predictions):
        max_prediction = max(predictions, key=lambda x: x["probability"]) # get highest probability
        dtct_obj = max_prediction["tagName"] # get tag
        return dtct_obj

    

          