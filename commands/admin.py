description = (
    "List the admin / super-admin commands (kept out of the public ~help). The "
    "commands themselves still enforce their own permissions.\n"
    "  ~admin"
)


async def handle_admin(bot, message, params):
    from command_registry import command_handlers
    from commands.help import format_admin_list
    await message.channel.send(format_admin_list(bot.prefix, command_handlers.keys()))
