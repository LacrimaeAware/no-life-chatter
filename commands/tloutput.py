from utils.user_settings import update_user_setting, ensure_user_settings
from auth import is_authorized

description = (
    'Where your translations are sent (admins).\n'
    '  ~tloutput local|whisper|channel <name>'
)


async def handle_tloutput(bot, message, params):
    if not is_authorized(message.author.name):
        await message.channel.send("You do not have permission to change the output mode.")
        return

    # must include a valid mode
    if not params or params[0] not in ['local', 'whisper', 'channel']:
        await message.channel.send(
            "Invalid mode. Use `~tloutput local`, `~tloutput whisper`, or `~tloutput channel <channel_name>`."
        )
        return

    ensure_user_settings(message.author.id)

    mode = params[0]

    # --- Validation for channel mode ---
    if mode == 'channel':
        if len(params) < 2:
            await message.channel.send(
                "Please specify a channel name. Example: `~tloutput channel xqc`."
            )
            return

        channel_name = params[1].lower().lstrip('#').lstrip('~')
        update_user_setting(message.author.id, 'output_channel', channel_name)
        update_user_setting(message.author.id, 'output_mode', mode)
        await message.channel.send(f"Output mode set to **channel** in **{channel_name}**.")
        return

    # --- Local and whisper modes ---
    update_user_setting(message.author.id, 'output_mode', mode)
    update_user_setting(message.author.id, 'output_channel', None)
    await message.channel.send(f"Output mode set to **{mode}** (will send in the same channel).")
