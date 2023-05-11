# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azure.core.credentials import AzureKeyCredential
from botbuilder.core import ActivityHandler, TurnContext
from azure.ai.language.questionanswering import models as qna
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes
from azure.ai.language.questionanswering.aio import QuestionAnsweringClient


class MyBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        if turn_context.activity.attachments:
            reply = "I can answer only when the question is in text."
        else:
            reply = await self.qna(turn_context.activity.text)
        
        reply_activity = Activity(
                        type=ActivityTypes.message,
                        text=reply)
        await turn_context.send_activity(reply_activity)

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome! How may I help you today?")

    async def qna(self, question):
        endpoint = "https://sig788-task5-ls-001.cognitiveservices.azure.com/"
        key = "14211f81e1ae48488eb06c46d6ae447f"
        knowledge_base_project = "sig788-task5-1-prj"

        client = QuestionAnsweringClient(endpoint, AzureKeyCredential(key))

        async with client:
            
            output = await client.get_answers(
                question=question,
                top=3,
                confidence_threshold=0.2,
                include_unstructured_sources=True,
                short_answer_options=qna.ShortAnswerOptions(
                    confidence_threshold=0.2,
                    top=1
                ),
                project_name=knowledge_base_project,
                deployment_name='production'
            )
            if output.answers:
                best_candidate = [a for a in output.answers if a.confidence > 0.7]
                if best_candidate:
                    response = best_candidate[0].answer
                    return response
                else:
                    best_candidate = [a for a in output.answers if 0.5 <= a.confidence <= 0.7]
                    if best_candidate:
                        response = "I can answer this question with a confidence of {:.1%}:\n\n{} \n\nPlease be more specific with your query and I would try to find the answer for you.".format(best_candidate[0].confidence, best_candidate[0].answer)

                        return response
                    else:
                        response = "Sorry! I do not know the answer to that question. Please refine your query further."
                        return response
            else:
                response = "Sorry! I do not know the answer to that question. Please refine your query further."
                return response
