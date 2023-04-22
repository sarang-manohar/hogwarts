# Before integration with Azure Bot Service

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes
import requests

class MyBot(ActivityHandler):
    def __init__(self):
        self.API_KEY = "AIzaSyAAzu8U_tNbySGRkZy2n0VXnub21hn2gzA"  # Replace with your actual API key

    async def on_message_activity(self, turn_context: TurnContext):
        # Read user's input from activity.text
        query = turn_context.activity.text
        print(f"Received message: {query}")
        num_products = 5

        # Call recommend_products function to get recommended products
        result = self.recommend_products(query, num_products)

        # Send result back to user
        reply_activity = Activity(
            type=ActivityTypes.message,
            text=result
        )
        await turn_context.send_activity(reply_activity)

    # Define function to recommend products
    def recommend_products(self, query, num_products):
        # Set API endpoint and parameters
        url = f'https://www.googleapis.com/customsearch/v1?key={self.API_KEY}&cx=b731f8a663d17422e&cr=countryIN&q={query}&maxResults={num_products}'

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
            return "Sorry, I couldn't find any products that match your query."
