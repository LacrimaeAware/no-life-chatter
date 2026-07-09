description = (
    "List experimental / evidence-prototype commands.\n"
    "  ~experimental"
)

EXPERIMENTAL_COMMANDS = [
    ("askchat", "archive/lore QA; use raw/noai for receipts-only"),
    ("emote", "learned emote-meaning guess; use raw for vector/archive signals"),
    ("explain emote", "same emote explainer through the general explain route"),
    ("why emote", "emote evidence through the why command"),
    ("irony", "rough irony/sincerity read"),
    ("iq", "roster-relative text-cognition toy score"),
]


def format_experimental_list(prefix: str = "~") -> str:
    bits = [f"{prefix}{name} ({note})" for name, note in EXPERIMENTAL_COMMANDS]
    return "Experimental: " + " | ".join(bits)


async def handle_experimental(bot, message, params):
    await message.channel.send(format_experimental_list(bot.prefix))
