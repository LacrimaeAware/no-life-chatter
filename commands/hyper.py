from commands.persona import _run

description = (
    "Talk to an AI persona of a chatter in HYPERBOLE mode — their traits "
    "exaggerated for comedy.\n"
    "  ~hyper <user> [message]"
)


async def handle_hyper(bot, message, params):
    await _run(bot, message, params, mode="hyper")
