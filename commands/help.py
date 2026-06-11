# commands/help.py
import logging
from importlib import import_module

description = (
    'List commands, or show details for one.\n'
    '  ~help [command]'
)


async def handle_help(bot, message, params):
    from command_registry import command_handlers

    logging.info(f"Available commands: {', '.join(command_handlers.keys())}")  # Debugging

    if not params:
        available_commands = ', '.join(command_handlers.keys())
        help_text = f"Available commands: {available_commands}. Use {bot.prefix}help <command> for more details."
        await message.channel.send(help_text)
    else:
        command = params[0].lower()
        if command in command_handlers:
            try:
                module = import_module(f"commands.{command}")
                description = getattr(module, 'description', "No description available for this command.")
            except ImportError as e:
                description = f"Failed to load command module: {str(e)}"
            await message.channel.send(f"{command}: {description}")
        else:
            await message.channel.send(f"Command not found. Use {bot.prefix}help to list all commands.")
