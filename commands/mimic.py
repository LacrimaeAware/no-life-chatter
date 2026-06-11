import config
from utils import persona_markov
from utils.output_filter import is_clean

description = (
    "Generate a fake message in a chatter's style, from their chat history.\n"
    "  ~mimic <user>   posts a bot-made line that sounds like them"
)


async def handle_mimic(bot, message, params):
    if not getattr(config, "MIMIC_ENABLED", True):
        return
    if not params:
        await message.channel.send("Usage: ~mimic <user>")
        return

    user = params[0].lstrip("@")
    model = persona_markov.get_model(user)
    if not model:
        await message.channel.send(f"Not enough archived messages to mimic {user}.")
        return

    # Generate until we get a clean, non-trivial line (or give up). The output
    # filter is what keeps the bot from posting a bannable line to Twitch.
    line = None
    for _ in range(15):
        cand = persona_markov.generate(model)
        if cand and len(cand.split()) >= 2 and is_clean(cand):
            line = cand
            break
    if not line:
        await message.channel.send(f"Couldn't make a clean {user}-style line — try again.")
        return

    if len(line) > 280:
        line = line[:279] + "…"
    await message.channel.send(f"🎭 {user}-bot: {line}")
