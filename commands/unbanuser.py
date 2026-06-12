from auth import is_super_admin
from utils import command_bans

description = (
    "Lift a command ban (super admins only).\n"
    "  ~unbanuser <name>"
)


async def handle_unbanuser(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    if not params:
        await message.channel.send("Usage: ~unbanuser <name>")
        return
    target = params[0].lower().lstrip("@")
    if command_bans.unban(target):
        await message.channel.send(f"🔊 {target} can use commands again.")
    else:
        await message.channel.send(f"{target} wasn't banned.")
