import logging
from command_processor import CommandProcessor
from services.message_service import MessageService

class MessageHandler:
    def __init__(self, bot):
        self.bot = bot
        self.command_processor = CommandProcessor(bot)
        self.message_service = MessageService(bot)  # Initialize the message service

    async def process_message(self, message):
        if message.echo:
            return

        logging.info(f"Received message from {message.author.name}: {message.content}")

        if message.content.startswith(self.bot.prefix):
            logging.info("Message is a command; processing")
            await self.command_processor.process_command(message)
        else:
            logging.info("Message is regular chat; handling appropriately")
            await self.message_service.handle_regular_message(message)  # Delegate to message service
