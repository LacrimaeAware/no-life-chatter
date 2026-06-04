# auth.py
"""Authorization helpers.

The admin / super-admin lists live in config.toml ([auth] section) so no Twitch
usernames are baked into the source.
"""

import config


def is_authorized(user_name: str) -> bool:
    """Return True if the user may change their own bot settings."""
    return user_name.lower() in config.ADMINS or user_name.lower() in config.SUPER_ADMINS


def is_super_admin(user_name: str) -> bool:
    """Return True if the user may change global / per-channel settings."""
    return user_name.lower() in config.SUPER_ADMINS
