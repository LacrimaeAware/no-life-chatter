from utils.user_settings import update_user_setting, ensure_user_settings, get_user_settings

description = (
    "Practice mode:\n"
    "  ~practice on <learn_langs_csv> [native]\n"
    "  ~practice off\n"
    "  ~practice show\n"
    "Examples: ~practice on es        | ~practice on es,th\n"
    "          ~practice on ja,en en  | ~practice off | ~practice show"
)

async def handle_practice(bot, message, params):
    # Ensure the user has a settings row
    ensure_user_settings(message.author.id)

    if not params:
        await message.channel.send(description)
        return

    cmd = params[0].lower()

    # ----- SHOW -----
    if cmd == "show":
        s = get_user_settings(message.author.id) or {}
        learn = s.get("learn_lang", "ES")
        native = s.get("native_lang", "EN")
        pmode = bool(s.get("practice_mode", 0))
        await message.channel.send(
            f"practice_mode={pmode}  learn_lang={learn}  native_lang={native}"
        )
        return

    # ----- OFF -----
    if cmd == "off":
        update_user_setting(message.author.id, "practice_mode", 0)
        await message.channel.send("Practice mode OFF.")
        return

    # ----- ON -----
    if cmd == "on":
        if len(params) < 2:
            await message.channel.send("Usage: ~practice on <learn_langs_csv> [native]")
            return

        # accept one or multiple learn languages, comma-separated
        learn_csv_raw = params[1].upper()
        learn_list = [p.strip() for p in learn_csv_raw.split(",") if p.strip()]
        learn_csv = ",".join(learn_list) if learn_list else "ES"

        native = params[2].upper() if len(params) > 2 else "EN"

        # store settings
        update_user_setting(message.author.id, "practice_mode", 1)
        update_user_setting(message.author.id, "learn_lang", learn_csv)   # reuse existing column
        update_user_setting(message.author.id, "native_lang", native)

        await message.channel.send(f"Practice mode ON. learn_lang={learn_csv} native_lang={native}")
        return

    # ----- FALLBACK -----
    await message.channel.send(description)
