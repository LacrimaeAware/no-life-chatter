import logging
import time

import config
from command_processor import CommandProcessor
from services.message_service import MessageService
from utils.chat_archive import record_author_id, record_live
from utils import reaction_tracker

class MessageHandler:
    def __init__(self, bot):
        self.bot = bot
        self.command_processor = CommandProcessor(bot)
        self.message_service = MessageService(bot)  # Initialize the message service

    async def process_message(self, message):
        if message.echo:
            return

        logging.info(f"Received message from {message.author.name}: {message.content}")

        # Implicit funniness labels: every message may be a reaction to a line
        # the bot just posted (utils/reaction_tracker.py). Never raises.
        if message.channel and message.author:
            reaction_tracker.observe(message.channel.name, message.author.name,
                                     message.content, self.bot.nick or "")

        # Append to the searchable chat archive (docs/CHAT_ARCHIVE.md).
        # record_live swallows its own errors — it can never break chat handling.
        # message.channel is None for whispers (twitchio routes them here too).
        if message.author:
            record_author_id(message.author.name, getattr(message.author, "id", None))
        if config.ARCHIVE_LIVE and message.channel and message.author and message.content:
            record_live(
                message.channel.name,
                message.author.name,
                message.content,
                time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        if message.content.startswith(self.bot.prefix):
            logging.info("Message is a command; processing")
            await self.command_processor.process_command(message)
        else:
            logging.info("Message is regular chat; handling appropriately")
            await self.message_service.handle_regular_message(message)  # Delegate to message service
