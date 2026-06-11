import asyncio
import random

import config
from services import llm
from utils import chat_archive, persona_llm
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
    out = await persona_llm.generate_with_retry(user, message.channel.name, said, mode=mode)
    if not out:
        history_count = len(chat_archive.messages_for(user))
        reason = llm.last_error() or "model returned an empty response"
        if history_count:
            await message.channel.send(
                f"Couldn't generate {user} - LM Studio failed ({reason}). "
                f"{user} has {history_count:,} archived messages, so history is not the issue."
            )
        else:
            await message.channel.send(f"Couldn't generate {user} - no archived messages found for them.")
        return

    if not is_clean(out):
        out = await persona_llm.generate_with_retry(user, message.channel.name, said, mode=mode)
        if not out or not is_clean(out):
            await message.channel.send(f"({user}-bot said something I won't repeat - try again.)")
            return

    if len(out) > 480:
        out = out[:479] + "..."
    await message.channel.send(f"🎭 {user}: {out}")
    await _maybe_continue(message, user, said, out, mode)


async def _maybe_continue(message, user, original_prompt, previous_line, mode):
    chance = getattr(config, "PERSONA_COMMAND_CONTINUE_CHANCE", 0.0)
    max_lines = max(0, getattr(config, "PERSONA_COMMAND_MAX_CONTINUATIONS", 0))
    if chance <= 0 or max_lines <= 0:
        return

    last_line = previous_line
    for _ in range(max_lines):
        if random.random() >= chance:
            break
        await asyncio.sleep(getattr(config, "PERSONA_COMMAND_CONTINUE_DELAY", 1.5))
        follow_prompt = (
            f'{original_prompt or ""}\n'
            f'You just said: "{last_line}". Send one natural short follow-up '
            f"chat message as {user}, like a real chatter double-texting."
        )
        follow = await persona_llm.generate(
            user,
            message.channel.name,
            follow_prompt,
            mode=mode,
            exemplar_count=getattr(config, "LLM_RETRY_EXEMPLARS", 60),
            context_count=getattr(config, "LLM_RETRY_CONTEXT", 12),
        )
        if not follow or not is_clean(follow):
            break
        if len(follow) > 480:
            follow = follow[:479] + "..."
        await message.channel.send(f"🎭 {user} (cont.): {follow}")
        last_line = follow


async def handle_persona(bot, message, params):
    await _run(bot, message, params, mode="normal")
