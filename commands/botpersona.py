from auth import is_super_admin
from utils import resident_persona

description = (
    "Set, inspect, or clear a channel resident persona (super admins).\n"
    "  ~botpersona status [chat=<channel>]\n"
    "  ~botpersona off [chat=<channel>]\n"
    "  ~botpersona <user> [chat=<channel>] [minutes=360] [mode=regular|response|random|silent] "
    "[chance=] [topic=] [curve=] [directed=] [greeting=] [cooldown=] [idle=]"
)


def _as_float(value, field):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number")


def _as_bool(value):
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _parse(params, current_channel):
    channel = current_channel
    flags = {}
    rest = []
    for token in params:
        low = token.lower()
        if low.startswith("chat="):
            channel = token.split("=", 1)[1].strip().lstrip("#") or channel
        elif low.startswith(("minutes=", "mins=")):
            flags["minutes"] = _as_float(token.split("=", 1)[1], "minutes")
        elif low.startswith("mode="):
            flags["mode"] = token.split("=", 1)[1].strip().lower()
        elif low.startswith("chance="):
            flags["chance"] = _as_float(token.split("=", 1)[1], "chance")
        elif low.startswith("topic="):
            flags["topic_chance"] = _as_float(token.split("=", 1)[1], "topic")
        elif low.startswith("curve="):
            flags["topic_curve"] = _as_float(token.split("=", 1)[1], "curve")
        elif low.startswith("directed="):
            flags["directed_chance"] = _as_float(token.split("=", 1)[1], "directed")
        elif low.startswith("greeting="):
            flags["greeting_chance"] = _as_float(token.split("=", 1)[1], "greeting")
        elif low.startswith("cooldown="):
            flags["cooldown"] = _as_float(token.split("=", 1)[1], "cooldown")
        elif low.startswith("idle="):
            flags["idle_chance"] = _as_float(token.split("=", 1)[1], "idle")
        elif low.startswith("idle_after="):
            flags["idle_after"] = _as_float(token.split("=", 1)[1], "idle_after")
        elif low.startswith("idle_interval="):
            flags["idle_interval"] = _as_float(token.split("=", 1)[1], "idle_interval")
        elif low.startswith("idle_cooldown="):
            flags["idle_cooldown"] = _as_float(token.split("=", 1)[1], "idle_cooldown")
        elif low.startswith(("max_streak=", "streak=")):
            flags["max_bot_streak"] = int(_as_float(token.split("=", 1)[1], "max_streak"))
        elif low.startswith("reply="):
            flags["reply_to_trigger"] = _as_bool(token.split("=", 1)[1])
        elif low.startswith("prefix="):
            flags["prefix"] = token.split("=", 1)[1].replace("\\s", " ")
        else:
            rest.append(token)
    return channel, flags, rest


async def handle_botpersona(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    try:
        channel, flags, rest = _parse(params or [], message.channel.name)
    except ValueError as exc:
        await message.channel.send(f"Usage error: {exc}")
        return
    if not rest or rest[0].lower() == "status":
        state = resident_persona.get(channel)
        await message.channel.send(resident_persona.format_status(state))
        return
    if rest[0].lower() in {"off", "clear", "stop"}:
        resident_persona.clear(channel)
        await message.channel.send(f"resident persona off in #{channel}")
        return
    persona = rest[0].lstrip("@")
    mode = flags.get("mode")
    if mode and mode not in resident_persona.MODES:
        await message.channel.send("Usage: mode must be regular, response, random, or silent.")
        return
    minutes = flags.pop("minutes", None)
    until = resident_persona.now() + minutes * 60 if minutes else None
    state = resident_persona.set_state(channel, persona=persona, until=until, **flags)
    await message.channel.send("resident persona set: " + resident_persona.format_status(state))
