import asyncio

from utils import artifact_status

description = (
    "Show whether generated persona artifacts are current or stale.\n"
    "  ~artifacts"
)


async def handle_artifacts(bot, message, params):
    summary = await asyncio.to_thread(artifact_status.status_summary)
    await message.channel.send(summary)
