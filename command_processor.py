import logging
from command_registry import command_handlers, load_command_handlers
from utils.command_bans import is_banned
from utils import cooldowns

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

        handler = command_handlers.get(command)
        if not handler:
            logging.warning(f"Command not recognized: {command}")
            await message.channel.send(f"Command not recognized. Try {self.bot.prefix}help for command list.")
            return

        # Anti-spam: stacking commands while your previous one is still
        # processing is the troll pattern; heavy-but-patient use never
        # triggers (utils/cooldowns.py — escalating, per-user, reviewable
        # via ~warnings).
        user = message.author.name if message.author else ""
        verdict, secs = cooldowns.before(user)
        if verdict == "drop":
            return
        if verdict == "cooldown":
            await message.channel.send(
                f"@{user} slow down — command cooldown {secs // 60}m {secs % 60}s "
                "(stacking commands before the bot answers).")
            return
        try:
            await handler(self.bot, message, params)
        except Exception as e:
            logging.error(f"Error handling command {command}: {e}")
            await message.channel.send("An error occurred while processing your command.")
        finally:
            cooldowns.after(user)
