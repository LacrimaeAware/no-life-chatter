from commands.persona import _run

description = (
    "Talk to an AI persona of a chatter in HYPERBOLE mode — their traits "
    "exaggerated for comedy. model= forces a configured model.\n"
    "  ~hyper <user> [message] [model=llama|lora]"
)


async def handle_hyper(bot, message, params):
    await _run(bot, message, params, mode="hyper")
