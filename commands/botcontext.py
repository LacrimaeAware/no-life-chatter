from auth import is_super_admin
from utils import resident_persona

description = (
    "Set extra standing instruction for the resident persona (super admins).\n"
    "  ~botcontext [chat=<channel>] <text|clear>"
)


async def handle_botcontext(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    channel = message.channel.name
    rest = []
    for token in params or []:
        if token.lower().startswith("chat="):
            channel = token.split("=", 1)[1].strip().lstrip("#") or channel
        else:
            rest.append(token)
    state = resident_persona.get(channel)
    if not state:
        await message.channel.send(f"No resident persona set in #{channel}.")
        return
    text = " ".join(rest).strip()
    if not text:
        await message.channel.send(f"resident context in #{channel}: {state.get('context') or '(empty)'}")
        return
    state = resident_persona.set_state(channel, context="" if text.lower() == "clear" else text)
    await message.channel.send("resident context updated: " + resident_persona.format_status(state))
