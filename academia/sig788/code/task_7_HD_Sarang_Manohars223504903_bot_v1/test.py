async def on_message_activity(turn_context):
    # Check if the user sent an image attachment
    if 'attachments' in turn_context.activity:
        for attachment in turn_context.activity.attachments:
            # Check if the attachment is an image
            if attachment.content_type.startswith('image/'):
                # Get the binary data of the image
                image_data = await turn_context.adapter.get_attachment(attachment.id)

                # Define a ConversationRequest object to send the image to CLU
                conversation_request = ConversationRequest(input_image=image_data)

                # Use the LanguageConversationClient object to start a new conversation
                async with conversation_client.begin_conversation(conversation_request) as conversation:
                    # Get the response from CLU
                    response = await conversation.send_request(conversation_request)

                # Send the bot's response back to the user
                await turn_context.send_activity(response.get_first_generated_message().message)
                return
    # If the user did not send an image, process the message as text
    else:
        # Get the user's message from the turn context
        user_message = turn_context.activity.text

        # Define a ConversationRequest object to send the user's message to CLU
        conversation_request = ConversationRequest(input_text=user_message)

        # Use the LanguageConversationClient object to start a new conversation
        async with conversation_client.begin_conversation(conversation_request) as conversation:
            # Get the response from CLU
            response = await conversation.send_request(conversation_request)

        # Send the bot's response back to the user
        await turn_context.send_activity(response.get_first_generated_message().message)
