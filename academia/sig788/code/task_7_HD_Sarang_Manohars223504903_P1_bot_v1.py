from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes
import requests
from datetime import datetime
import pytz

class MyBot(ActivityHandler):
    def __init__(self):
        self.API_KEY = "AIzaSyAAzu8U_tNbySGRkZy2n0VXnub21hn2gzA"  # Replace with your actual API key

    async def on_message_activity(self, turn_context: TurnContext):
        # Read user's input from activity.text
        user_input = turn_context.activity.text
            
        print("user_input: ", user_input)

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


    # Define function to recommend products
    async def recommend_products(self, formatted_query):
        # Set API endpoint and parameters
        url = f'https://www.googleapis.com/customsearch/v1?key={self.API_KEY}&cx=b731f8a663d17422e&cr=countryIN&q={formatted_query}&num=5&safe=active'

        # Send GET request to API endpoint
        response = requests.get(url)

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

        # get secrets
        clu_endpoint = "https://sig788-task7-ls-006.cognitiveservices.azure.com"
        clu_key = "03dd4da6b78347dca783607c64bdf39e"
        project_name = "sig788-task7-CLU-prj"
        deployment_name = "sig788-task7-dpl-001"

        # create client
        async with ConversationAnalysisClient(clu_endpoint, AzureKeyCredential(clu_key)) as client:
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
                        "projectName": project_name,
                        "deploymentName": deployment_name,
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
          