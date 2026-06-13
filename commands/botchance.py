from auth import is_super_admin
from utils import resident_persona

description = (
    "Set resident persona base/directed reply chances (super admins).\n"
    "  ~botchance <base> [directed] [chat=<channel>]"
)


async def handle_botchance(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    if not params:
        await message.channel.send("Usage: ~botchance <base> [directed] [chat=<channel>]")
        return
    channel = message.channel.name
    nums = []
    for token in params:
        if token.lower().startswith("chat="):
            channel = token.split("=", 1)[1].strip().lstrip("#") or channel
        else:
            try:
                nums.append(float(token))
            except ValueError:
                await message.channel.send("Usage: ~botchance <base> [directed] [chat=<channel>]")
                return
    if not nums:
        await message.channel.send("Usage: ~botchance <base> [directed] [chat=<channel>]")
        return
    state = resident_persona.get(channel)
    if not state:
        await message.channel.send(f"No resident persona set in #{channel}.")
        return
    updates = {"chance": nums[0]}
    if len(nums) > 1:
        updates["directed_chance"] = nums[1]
    state = resident_persona.set_state(channel, **updates)
    await message.channel.send("resident chance set: " + resident_persona.format_status(state))
