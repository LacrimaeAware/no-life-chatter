from commands.mimic import run_markov

description = (
    "Generate a Markov-chain line in a chatter's style (built from their messages in THIS channel when they have enough).\n"
    "  ~markov <user>"
)


async def handle_markov(bot, message, params):
    await run_markov(message, params, command_name="markov")
