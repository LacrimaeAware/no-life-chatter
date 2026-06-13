import asyncio

from utils import archive_qa

description = (
    "Evidence-backed archive/lore search over claims, chat lines, and emotes.\n"
    "  ~askchat [user=<name>|<name>] [chat=<channel>|chat=here] <question>"
)


async def handle_askchat(bot, message, params):
    parsed = archive_qa.parse_params(params or [], getattr(message.channel, "name", None))
    if not parsed["query"]:
        await message.channel.send(
            "Usage: ~askchat [user=<name>|<name>] [chat=<channel>|chat=here] <question>"
        )
        return
    report = await asyncio.to_thread(
        archive_qa.build_report,
        parsed["query"],
        author=parsed["author"],
        channel=parsed["channel"],
    )
    await message.channel.send(archive_qa.format_chat(report))
