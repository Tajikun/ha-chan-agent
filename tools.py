import discord


async def collect_collab_metadata(
    guild: discord.Guild,
    *,
    collab_category_id: int | None = None,
) -> str:
    """Return a string summary of collab channels for the LLM."""
    if guild is None:
        return "No guild context available."

    channels = []
    if collab_category_id:
        category = guild.get_channel(collab_category_id)
        if isinstance(category, discord.CategoryChannel):
            channels = category.text_channels
    else:
        channels = guild.text_channels

    summaries = []
    for channel in channels:
        created = channel.created_at.isoformat()
        topic = channel.topic or "No topic set"
        summaries.append(
            f"{channel.name} (created {created}) topic: {topic}"
        )

    return "\n".join(summaries) or "No collab channels found."