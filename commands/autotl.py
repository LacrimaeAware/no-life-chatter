from utils.user_settings import get_user_settings, ensure_user_settings, update_user_setting
from auth import is_authorized  # Ensure this is the correct path to the authorization function

async def handle_autotl(bot, message, params):
    """
    Toggles the translation setting for authorized users.

    Parameters:
        bot (Bot): The instance of the bot.
        message (Message): The message object from Twitch chat.
        params (list): List of parameters passed with the command.
    """
    if is_authorized(message.author.name):  # Check if the user is authorized
        ensure_user_settings(message.author.id)  # Ensure settings are initialized
        current_settings = get_user_settings(message.author.id)
        new_status = not current_settings.get('translation_enabled', False)
        update_user_setting(message.author.id, 'translation_enabled', int(new_status))
        status_msg = "enabled" if new_status else "disabled"
        await message.channel.send(f"Automatic translation has been {status_msg}.")
    else:
        await message.channel.send("You do not have permission to toggle translation.")
