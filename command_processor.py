import logging
from command_registry import command_handlers, load_command_handlers
from utils.command_bans import is_banned

class CommandProcessor:
    def __init__(self, bot):
        self.bot = bot

    async def process_command(self, message):
        # command-banned users are silently ignored (~banuser/~unbanuser)
        if message.author and is_banned(message.author.name):
            logging.info(f"Ignored command from banned user {message.author.name}")
            return
        parts = message.content.lstrip(self.bot.prefix).split()
        if not parts:  # bare '~' / '~~' — not a command, don't crash
            return
        command = parts[0].lower()
        params = parts[1:]

        # Optional: Reload command handlers if needed (useful in development)
        # load_command_handlers()

        handler = command_handlers.get(command)
        if handler:
            try:
                await handler(self.bot, message, params)
            except Exception as e:
                logging.error(f"Error handling command {command}: {e}")
                await message.channel.send("An error occurred while processing your command.")
        else:
            logging.warning(f"Command not recognized: {command}")
            await message.channel.send(f"Command not recognized. Try {self.bot.prefix}help for command list.")
