from auth import is_super_admin
from utils import command_bans

description = (
    "Ban a chatter from using ANY bot command (super admins only). Their "
    "commands are silently ignored; translation/archiving unaffected.\n"
    "  ~banuser <name> · ~banuser list"
)


async def handle_banuser(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    if not params:
        await message.channel.send("Usage: ~banuser <name> | ~banuser list")
        return
    if params[0].lower() == "list":
        banned = command_bans.banned_list()
        await message.channel.send(
            ("command-banned: " + ", ".join(banned)) if banned else "nobody is command-banned.")
        return
    target = params[0].lower().lstrip("@")
    if is_super_admin(target):
        await message.channel.send("not banning a super admin.")
        return
    command_bans.ban(target)
    await message.channel.send(f"🔇 {target} is banned from all commands.")
