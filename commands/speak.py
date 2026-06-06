from utils.speaker_profile import flag_speaker, get_profile

description = (
    "Tell the bot which language(s) you speak so it translates you reliably.\n"
    "  ~speak <lang>       flag yourself as a speaker (e.g. ~speak es)\n"
    "  ~speak <lang> off   remove that flag\n"
    "  ~speak show         show your language profile"
)


async def handle_speak(bot, message, params):
    user_id = message.author.id

    if not params or params[0].lower() == "show":
        profile = get_profile(user_id)
        if not profile:
            await message.channel.send("No language profile yet — just keep chatting.")
            return
        parts = [
            f"{lang}({'flagged' if v['flagged'] else v['count']})"
            for lang, v in sorted(profile.items())
        ]
        await message.channel.send("Your languages: " + ", ".join(parts))
        return

    lang = params[0].upper()
    off = len(params) > 1 and params[1].lower() == "off"
    flag_speaker(user_id, lang, on=not off)
    if off:
        await message.channel.send(f"Removed your {lang} speaker flag.")
    else:
        await message.channel.send(f"Got it — flagged you as a {lang} speaker.")
