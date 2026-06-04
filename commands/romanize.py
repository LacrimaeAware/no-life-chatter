from utils.user_settings import ensure_user_settings, update_user_setting, get_user_settings

description = "Toggle romanization of practice outputs. Usage: ~romanize on | ~romanize off | ~romanize show"

async def handle_romanize(bot, message, params):
    ensure_user_settings(message.author.id)

    if not params:
        await message.channel.send(description)
        return

    cmd = params[0].lower()

    if cmd == "on":
        update_user_setting(message.author.id, "romanize_enabled", 1)
        await message.channel.send("Romanization: ON")
        return

    if cmd == "off":
        update_user_setting(message.author.id, "romanize_enabled", 0)
        await message.channel.send("Romanization: OFF")
        return

    if cmd == "show":
        s = get_user_settings(message.author.id) or {}
        await message.channel.send(f"romanize_enabled={bool(s.get('romanize_enabled', 0))}")
        return

    await message.channel.send(description)
