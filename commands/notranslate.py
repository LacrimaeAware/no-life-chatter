from auth import is_super_admin
from utils import translate_optout
from utils.chat_archive import normalize_author

description = (
    "~notranslate <user> — stop auto-translating a user's messages (super admins). "
    "~notranslate <user> undo to re-enable; ~notranslate alone lists who's opted out."
)


async def handle_notranslate(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    if not params:
        opted = translate_optout.list_opted_out()
        await message.channel.send(
            "Auto-translate opted-out: " + (", ".join(opted) if opted else "(nobody)")
        )
        return
    user = params[0].lstrip("@").lower()
    off = not (len(params) > 1 and params[1].lower() in
               ("undo", "remove", "clear", "on", "enable", "reset"))
    translate_optout.set_opt_out(user, off)
    state = "OFF (won't auto-translate)" if off else "ON (auto-translate again)"
    await message.channel.send(f"{normalize_author(user)}: auto-translate {state}.")
