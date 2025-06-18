import os
import asyncio
from aiohttp import web
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity, ActivityTypes # Required for process_activity
from dotenv import load_dotenv

from app.bot import AzureRAGBot # Import the bot logic

# Load environment variables from .env file at the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Bot Framework Adapter settings
# These environment variables are typically provided by Azure Bot Service registration
APP_ID = os.getenv("MICROSOFT_APP_ID", "")
APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "")

if not APP_ID or not APP_PASSWORD:
    print("Warning: MICROSOFT_APP_ID or MICROSOFT_APP_PASSWORD not found in .env file.")
    print("The bot may not be able to authenticate with Azure Bot Service.")
    # Depending on the channel, the bot might still work locally with Bot Framework Emulator
    # if APP_ID and APP_PASSWORD are not strictly required by the emulator.

SETTINGS = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
ADAPTER = BotFrameworkAdapter(SETTINGS)

# Create an instance of the bot
BOT = AzureRAGBot()

# --- Define Bot Status/Health Check Endpoint ---
async def health_check(request: web.Request) -> web.Response:
    # Basic health check, can be expanded (e.g., check Azure service connectivity)
    return web.json_response({"status": "healthy", "message": "Bot is running."})

# --- Define Bot Messaging Endpoint ---
async def messages(request: web.Request) -> web.Response:
    """
    Main bot endpoint to handle incoming messages.
    """
    if "application/json" not in request.headers.get("Content-Type", ""):
        return web.Response(status=415, text="Unsupported Media Type: Expected application/json")

    body = await request.json()
    activity = Activity().deserialize(body) # Deserialize JSON to Activity object
    auth_header = request.headers.get("Authorization", "")

    try:
        # Process the activity using the adapter
        # The adapter's process_activity method will call the bot's on_turn method
        response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)

        if response: # If the adapter handled the request and generated a response (e.g., for invoke activities)
            return web.json_response(response.body, status=response.status)
        return web.Response(status=200) # Default status for successful processing if no specific response body

    except Exception as e:
        print(f"Error processing activity: {e}")
        # Consider logging the full exception trace for debugging
        # Return a generic error to the client
        return web.Response(status=500, text="Internal Server Error")

# --- Setup Web Application ---
APP = web.Application()
APP.router.add_post("/api/messages", messages) # Standard endpoint for Bot Framework
APP.router.add_get("/health", health_check)   # Health check endpoint

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 3978)) # Default port for Bot Framework bots
    print(f"Bot server starting on http://localhost:{PORT}")
    print(f"Messaging endpoint available at http://localhost:{PORT}/api/messages")
    print(f"Health check available at http://localhost:{PORT}/health")
    try:
        web.run_app(APP, port=PORT)
    except Exception as e:
        print(f"Error running web app: {e}")
