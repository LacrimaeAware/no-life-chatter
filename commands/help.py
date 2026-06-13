from importlib import import_module

description = (
    'List commands, or show details for one.\n'
    '  ~help [page|command]'
)

CATEGORIES = [
    ("translation", [
        "practice", "romanize", "speak", "autotl", "setlang",
        "tloutput", "chan_autotl", "global_autotl",
    ]),
    ("archive", [
        "said", "saidnext", "regex", "userregex", "quote", "firstseen",
        "chatstats", "regulars",
    ]),
    ("personas", [
        "markov", "mimic", "persona", "hyper", "generate",
    ]),
    ("analysis", [
        "whosaid", "markers", "like", "vibes", "twin", "traits", "top",
        "distinct", "why", "emote", "iq", "irony",
    ]),
    ("moderation", [
        "banuser", "unbanuser", "warnings",
    ]),
    ("utility", [
        "help", "ping", "artifacts",
    ]),
]

GROUPS_PER_PAGE = 2


def _known_command_pages(command_names):
    grouped = []
    seen = set()
    command_set = set(command_names)
    for title, names in CATEGORIES:
        present = [name for name in names if name in command_set]
        if present:
            grouped.append((title, present))
            seen.update(present)
    leftovers = sorted(command_set - seen)
    if leftovers:
        grouped.append(("other", leftovers))
    return [
        grouped[index:index + GROUPS_PER_PAGE]
        for index in range(0, len(grouped), GROUPS_PER_PAGE)
    ]


def _format_command_list(prefix: str, page: int, pages) -> str:
    total = max(1, len(pages))
    page = max(1, min(page, total))
    chunks = []
    for title, names in pages[page - 1]:
        rendered = " ".join(f"{prefix}{name}" for name in names)
        chunks.append(f"{title}: {rendered}")
    return (
        f"Commands {page}/{total}: " + " | ".join(chunks) +
        f" | {prefix}help <command> for details"
    )


async def handle_help(bot, message, params):
    from command_registry import command_handlers

    if not params:
        pages = _known_command_pages(command_handlers.keys())
        await message.channel.send(_format_command_list(bot.prefix, 1, pages))
    else:
        command = params[0].lower()
        if command.isdigit():
            pages = _known_command_pages(command_handlers.keys())
            await message.channel.send(_format_command_list(bot.prefix, int(command), pages))
            return
        if command in command_handlers:
            try:
                module = import_module(f"commands.{command}")
                description = getattr(module, 'description', "No description available for this command.")
            except ImportError as e:
                description = f"Failed to load command module: {str(e)}"
            await message.channel.send(f"{command}: {description}")
        else:
            await message.channel.send(f"Command not found. Use {bot.prefix}help to list all commands.")
