from commands.mimic import run_markov

description = (
    "Generate a Markov-chain line in a chatter's style.\n"
    "  ~markov <user>"
)


async def handle_markov(bot, message, params):
    await run_markov(message, params, command_name="markov")
