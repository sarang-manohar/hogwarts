from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes
import requests

from creds import DefaultCreds

CREDS = DefaultCreds()

class MyBot(ActivityHandler):
    def __init__(self):
        self.API_KEY = CREDS.APP_GS_API_KEY  # Replace with your actual API key

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
          