from utils import persona_llm
from utils.output_filter import is_clean

description = (
    "Talk to an AI persona of a chatter (natural mode), built from their chat "
    "history and the live conversation.\n"
    "  ~persona <user> [message]"
)


async def _run(bot, message, params, mode):
    if not params:
        await message.channel.send(f"Usage: ~{'hyper' if mode=='hyper' else 'persona'} <user> [message]")
        return
    user = params[0].lstrip("@")
    said = " ".join(params[1:]) or None
    out = await persona_llm.generate(user, message.channel.name, said, mode=mode)
    if not out:
        await message.channel.send(
            f"Couldn't generate {user} — is the model server (LM Studio) running, "
            f"and is there enough chat history for them?"
        )
        return
    if not is_clean(out):
        out = await persona_llm.generate(user, message.channel.name, said, mode=mode)
        if not out or not is_clean(out):
            await message.channel.send(f"({user}-bot said something I won't repeat — try again.)")
            return
    if len(out) > 480:
        out = out[:479] + "…"
    await message.channel.send(f"🎭 {user}: {out}")


async def handle_persona(bot, message, params):
    await _run(bot, message, params, mode="normal")
