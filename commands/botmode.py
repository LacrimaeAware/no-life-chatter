from auth import is_super_admin
from utils import resident_persona

description = (
    "Set resident persona response mode (super admins).\n"
    "  ~botmode regular|response|random|silent [minutes] [chat=<channel>]"
)


async def handle_botmode(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    if not params or params[0].lower() not in resident_persona.MODES:
        await message.channel.send("Usage: ~botmode regular|response|random|silent [minutes] [chat=<channel>]")
        return
    mode = params[0].lower()
    channel = message.channel.name
    minutes = None
    for token in params[1:]:
        low = token.lower()
        if low.startswith("chat="):
            channel = token.split("=", 1)[1].strip().lstrip("#") or channel
        else:
            try:
                minutes = float(token)
            except ValueError:
                pass
    state = resident_persona.get(channel)
    if not state:
        await message.channel.send(f"No resident persona set in #{channel}. Use ~botpersona first.")
        return
    until = resident_persona.now() + minutes * 60 if minutes else None
    state = resident_persona.set_state(channel, mode=mode, until=until)
    await message.channel.send("resident mode set: " + resident_persona.format_status(state))
