from auth import is_super_admin
from utils import resident_persona

description = (
    "Set resident persona reply chances (super admins).\n"
    "  ~botchance <base> [directed] [chat=<channel>] [topic=] [idle=] [greeting=] [cooldown=]"
)


async def handle_botchance(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    if not params:
        await message.channel.send("Usage: ~botchance <base> [directed] [chat=<channel>]")
        return
    channel = message.channel.name
    nums = []
    named = {}
    for token in params:
        if token.lower().startswith("chat="):
            channel = token.split("=", 1)[1].strip().lstrip("#") or channel
        elif "=" in token:
            key, value = token.split("=", 1)
            key = key.strip().lower()
            try:
                val = float(value)
            except ValueError:
                await message.channel.send(
                    "Usage: ~botchance <base> [directed] [chat=<channel>] [topic=] [idle=]")
                return
            if key == "topic":
                named["topic_chance"] = val
            elif key == "idle":
                named["idle_chance"] = val
            elif key == "greeting":
                named["greeting_chance"] = val
            elif key == "cooldown":
                named["cooldown"] = val
        else:
            try:
                nums.append(float(token))
            except ValueError:
                await message.channel.send("Usage: ~botchance <base> [directed] [chat=<channel>]")
                return
    if not nums and not named:
        await message.channel.send("Usage: ~botchance <base> [directed] [chat=<channel>]")
        return
    state = resident_persona.get(channel)
    if not state:
        await message.channel.send(f"No resident persona set in #{channel}.")
        return
    updates = dict(named)
    if nums:
        updates["chance"] = nums[0]
    if len(nums) > 1:
        updates["directed_chance"] = nums[1]
    state = resident_persona.set_state(channel, **updates)
    await message.channel.send("resident chance set: " + resident_persona.format_status(state))
