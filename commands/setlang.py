from utils.user_settings import update_user_setting, ensure_user_settings
from auth import is_authorized  # Ensure this is the correct path to the authorization function

description = (
    'Set your translation target language (admins).\n'
    '  ~setlang <LANG>   e.g. ~setlang EN'
)


async def handle_setlang(bot, message, params):
    """
    Sets the translation language for authorized users.

    Parameters:
        bot (Bot): The instance of the bot.
        message (Message): The message object from Twitch chat.
        params (list): List of parameters passed with the command.
    """
    if is_authorized(message.author.name):  # Check authorization
        ensure_user_settings(message.author.id)  # Ensure settings are initialized
        if params:
            update_user_setting(message.author.id, 'translation_language', params[0].upper())
            await message.channel.send(f"Translation language set to {params[0].upper()}.")
        else:
            await message.channel.send("Please specify a language.")
    else:
        await message.channel.send("You do not have permission to set the language.")
