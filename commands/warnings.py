from auth import is_super_admin
from utils.cooldowns import warnings_list

description = (
    "Review recent anti-spam cooldown offenses (super admins only).\n"
    "  ~warnings [n]"
)


async def handle_warnings(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    n = 5
    if params and params[0].isdigit():
        n = max(1, min(int(params[0]), 15))
    rows = warnings_list(n)
    if not rows:
        await message.channel.send("no cooldown offenses on record.")
        return
    parts = [f"{u} ({at[:16]}, {m:g}m)" for u, at, m in rows]
    await message.channel.send("⚠️ recent offenses: " + " · ".join(parts))
