from botbuilder.core import ActivityHandler, TurnContext, MessageFactory
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes # Added Activity, ActivityTypes
from .rag_pipeline import run_rag_pipeline
import html # To escape user messages if they are reflected in logs or certain types of responses

class AzureRAGBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        """
        Handles incoming message activities.
        It runs the RAG pipeline with the user's query and sends back the response.
        """
        user_query = turn_context.activity.text
        print(f"Received user query: {user_query}") # Log received query

        # It's good practice to show a "typing" indicator if the response might take time.
        typing_activity = Activity(type=ActivityTypes.typing)
        await turn_context.send_activity(typing_activity)

        try:
            # Get the response from the RAG pipeline
            response_text = run_rag_pipeline(user_query)

            # Sanitize or format the response if needed.
            # For example, if responses might contain markdown and the channel supports it.
            # Or if you need to ensure plain text.
            # For now, we send it as is.

            print(f"Sending response: {response_text[:200]}...") # Log a snippet of the response
            await turn_context.send_activity(MessageFactory.text(response_text))

        except Exception as e:
            error_message = f"Sorry, an error occurred while processing your request: {str(e)}"
            print(f"Error in RAG pipeline or sending message: {e}")
            # Sanitize error message before sending to user, to avoid leaking sensitive details
            user_friendly_error = "Sorry, I encountered an issue. Please try again later."
            await turn_context.send_activity(MessageFactory.text(user_friendly_error))


    async def on_members_added_activity(
        self,
        members_added: list[ChannelAccount],
        turn_context: TurnContext
    ):
        """
        Greets new members when they are added to the conversation.
        """
        for member in members_added:
            # Greet anyone that was not the target (recipient) of this message.
            # The bot is the recipient of all events from the channel, including members being added.
            # This check prevents the bot from greeting itself.
            if member.id != turn_context.activity.recipient.id:
                welcome_message = (
                    "Hello and welcome! I'm an Azure RAG Bot. "
                    "Ask me questions, and I'll try to answer them using my knowledge base."
                )
                await turn_context.send_activity(MessageFactory.text(welcome_message))
                print("Sent welcome message to new member.")

if __name__ == "__main__":
    # This section is primarily for illustration or local testing of the class structure,
    # as the bot is typically run via an adapter like in run_bot.py.

    class MockTurnContext:
        def __init__(self, text: str):
            self.activity = type('Activity', (), {'text': text, 'recipient': type('Recipient', (), {'id': 'bot_id'})})()
            self.sent_activities = []

        async def send_activity(self, activity_or_text):
            if isinstance(activity_or_text, str):
                self.sent_activities.append(MessageFactory.text(activity_or_text))
            else: # It's an Activity object
                 # If it's a typing indicator, just note it
                if activity_or_text.type == ActivityTypes.typing:
                    print("[MockTurnContext] Typing indicator sent.")
                    return
                self.sent_activities.append(activity_or_text)
            print(f"[MockTurnContext] Activity sent: {activity_or_text.text if hasattr(activity_or_text, 'text') else activity_or_text.type}")

    import asyncio

    async def run_bot_test():
        bot = AzureRAGBot()

        # Test welcome message (simulating member added)
        print("\n--- Testing Welcome Message ---")
        member_added_context = MockTurnContext("")
        # Simulate a new member being added (not the bot itself)
        new_member = ChannelAccount(id="user1", name="Test User")
        await bot.on_members_added_activity([new_member], member_added_context)
        assert any("Hello and welcome!" in str(act.text) for act in member_added_context.sent_activities if hasattr(act, 'text'))
        print("Welcome message test complete.")

        # Test message activity (simulating user query)
        # This requires the RAG pipeline to be functional, including .env setup
        print("\n--- Testing Message Activity (requires .env setup) ---")
        from .openai_client import OPENAI_API_KEY # Check if RAG can run
        if not OPENAI_API_KEY:
            print("Skipping message activity test as OpenAI key is not set.")
            return

        test_query = "What is the purpose of this bot?"
        message_context = MockTurnContext(test_query)
        await bot.on_message_activity(message_context)

        print("\nSent activities from mock context:")
        for activity in message_context.sent_activities:
            print(f"- Type: {activity.type}, Text: {getattr(activity, 'text', 'N/A')}")

        assert len(message_context.sent_activities) > 0, "Bot did not send a response."
        final_response = message_context.sent_activities[-1] # Last activity should be the answer
        assert hasattr(final_response, 'text') and test_query in final_response.text or "[RAG - generated]" in final_response.text or "[CAG - cached]" in final_response.text or "error" in final_response.text.lower()
        print("Message activity test complete (check output for RAG/CAG response).")

    # Python 3.7+
    # asyncio.run(run_bot_test()) # Commented out as it requires full .env setup to pass fully

    # Fallback for older versions or to avoid auto-run if not desired
    if __name__ == "__main__":
        print("To test AzureRAGBot functionality, ensure .env is configured and uncomment asyncio.run(run_bot_test())")
        print("Or run the full bot server using `python run_bot.py`.")
