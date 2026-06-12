import asyncio
import random

import config
from services import llm
from utils import chat_archive, persona_llm
from utils.output_filter import is_clean

description = (
    "Talk to an AI persona of a chatter (natural mode), built from their chat "
    "history (this channel's when they have enough) and the live conversation. "
    "model= forces a configured model.\n"
    "  ~persona <user> [message] [model=llama|lora]"
)


def _stop_followup(text):
    return (text or "").strip().upper() in {"STOP", "NO", "NONE"}


async def _run(bot, message, params, mode):
    if not params:
        await message.channel.send(f"Usage: ~{'hyper' if mode=='hyper' else 'persona'} <user> [message]")
        return

    # model=lora / model=llama (or a full LM Studio id) forces the model
    override = None
    kept = []
    for p in params:
        if p.lower().startswith("model="):
            override = persona_llm.resolve_model(p.split("=", 1)[1])
        else:
            kept.append(p)
    params = kept
    if not params:
        await message.channel.send(f"Usage: ~{'hyper' if mode=='hyper' else 'persona'} <user> [message] [model=lora]")
        return
    user = params[0].lstrip("@")
    said = " ".join(params[1:]) or None
    invoker = message.author.name if message.author else None
    out = await persona_llm.generate_with_retry(
        user, message.channel.name, said, mode=mode, invoked_by=invoker,
        model_override=override)
    if not out:
        history_count = len(chat_archive.messages_for(user))
        rejection = persona_llm.last_rejection()
        reason = rejection or llm.last_error() or "model returned an empty response"
        if history_count:
            label = "generation blocked" if rejection else "LM Studio failed"
            await message.channel.send(
                f"Couldn't generate {user} - {label} ({reason}). "
                f"{user} has {history_count:,} archived messages, so history is not the issue."
            )
        else:
            await message.channel.send(f"Couldn't generate {user} - no archived messages found for them.")
        return

    if not is_clean(out):
        persona_llm.log_event({
            "type": "output_filtered", "author": user, "mode": mode,
            "channel": message.channel.name, "invoked_by": invoker,
            "user_message": said, "text": out,
        })
        out = await persona_llm.generate_with_retry(
            user, message.channel.name, said, mode=mode, invoked_by=invoker,
            model_override=override)
        if not out or not is_clean(out):
            if out:
                persona_llm.log_event({
                    "type": "output_filtered", "author": user, "mode": mode,
                    "channel": message.channel.name, "invoked_by": invoker,
                    "user_message": said, "text": out,
                })
            await message.channel.send(f"({user}-bot said something I won't repeat - try again.)")
            return

    if len(out) > 480:
        out = out[:479] + "..."
    tag = persona_llm.last_model_tag()
    prefix = f"#{tag} " if tag else ""
    await message.channel.send(f"🎭 {prefix}{user}: {out}")
    from utils import reaction_tracker
    reaction_tracker.watch(message.channel.name, out,
                           {"kind": "persona", "persona": user, "mode": mode,
                            "model_tag": tag})
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
            f'Original message to you: "{original_prompt or ""}"\n'
            f'You just said: "{last_line}". Decide if {user} would naturally '
            f"send one immediate second chat message that continues the SAME "
            f"thought. If yes, output only that short follow-up. If no coherent "
            f"follow-up is natural, output exactly STOP. Do not change topic."
        )
        follow = await persona_llm.generate(
            user,
            message.channel.name,
            follow_prompt,
            mode=mode,
            exemplar_count=getattr(config, "LLM_RETRY_EXEMPLARS", 60),
            context_count=getattr(config, "LLM_RETRY_CONTEXT", 12),
        )
        if not follow or _stop_followup(follow) or not is_clean(follow):
            break
        if len(follow) > 480:
            follow = follow[:479] + "..."
        tag = persona_llm.last_model_tag()
        prefix = f"#{tag} " if tag else ""
        await message.channel.send(f"↳ 🎭 {prefix}{user}: {follow}")
        last_line = follow


async def handle_persona(bot, message, params):
    await _run(bot, message, params, mode="normal")
