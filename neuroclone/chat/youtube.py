"""YouTube Live chat via the Data API v3 (needs an API key and the live video's id)."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional, Union

import aiohttp

from ..config import YouTubeConfig
from ..events import ChatMessage, StreamEvent

log = logging.getLogger(__name__)

API = "https://www.googleapis.com/youtube/v3"
Sink = Callable[[Union[ChatMessage, StreamEvent]], Union[None, Awaitable[None]]]


def item_to_events(item: dict) -> list[Union[ChatMessage, StreamEvent]]:
    snippet = item.get("snippet", {})
    author = item.get("authorDetails", {})
    user = author.get("displayName", "someone")
    kind = snippet.get("type", "")
    badges = set()
    if author.get("isChatOwner"):
        badges.add("broadcaster")
    if author.get("isChatModerator"):
        badges.add("moderator")
    if author.get("isChatSponsor"):
        badges.add("member")
    if kind == "textMessageEvent":
        text = snippet.get("textMessageDetails", {}).get("messageText") or snippet.get("displayMessage", "")
        return [ChatMessage(user=user, text=text, platform="youtube", user_id=author.get("channelId", ""),
                            badges=badges)]
    if kind in ("superChatEvent", "superStickerEvent"):
        details = snippet.get("superChatDetails") or snippet.get("superStickerDetails") or {}
        amount = float(details.get("amountMicros", 0) or 0) / 1_000_000
        comment = details.get("userComment", "")
        events: list = [StreamEvent("superchat", user, amount=amount, message=comment, platform="youtube",
                                    display_amount=details.get("amountDisplayString", ""))]
        if comment:
            events.append(ChatMessage(user=user, text=comment, platform="youtube", badges=badges, bits=int(amount * 100)))
        return events
    if kind == "newSponsorEvent":
        return [StreamEvent("member", user, amount=1, platform="youtube")]
    if kind == "memberMilestoneChatEvent":
        details = snippet.get("memberMilestoneChatDetails", {})
        return [StreamEvent("member", user, amount=float(details.get("memberMonth", 1) or 1),
                            message=details.get("userComment", ""), platform="youtube")]
    if kind == "membershipGiftingEvent":
        count = snippet.get("membershipGiftingDetails", {}).get("giftMembershipsCount", 1)
        return [StreamEvent("gift", user, amount=float(count), platform="youtube")]
    return []


class YouTubeChat:
    def __init__(self, cfg: YouTubeConfig, sink: Sink, ignore_users: Optional[list] = None) -> None:
        self.cfg = cfg
        self.sink = sink
        self.ignore = {u.lower() for u in (ignore_users or [])}
        self._closing = False
        self.connected = False

    async def _get(self, session: aiohttp.ClientSession, path: str, params: dict) -> dict:
        params = {**params, "key": self.cfg.api_key}
        async with session.get(f"{API}/{path}", params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                raise RuntimeError(f"YouTube API {resp.status}: {str(data)[:200]}")
            return data

    async def run(self) -> None:
        if not (self.cfg.video_id and self.cfg.api_key):
            return
        async with aiohttp.ClientSession(trust_env=True) as session:
            chat_id = None
            while not self._closing and chat_id is None:
                try:
                    data = await self._get(session, "videos", {"part": "liveStreamingDetails", "id": self.cfg.video_id})
                    items = data.get("items", [])
                    chat_id = items[0].get("liveStreamingDetails", {}).get("activeLiveChatId") if items else None
                    if not chat_id:
                        log.warning("YouTube video %s has no active live chat yet; retrying", self.cfg.video_id)
                        await asyncio.sleep(30)
                except Exception as exc:  # noqa: BLE001
                    log.warning("YouTube chat lookup failed: %s", exc)
                    await asyncio.sleep(30)
            page_token, first = None, True
            self.connected = True
            while not self._closing:
                try:
                    params = {"liveChatId": chat_id, "part": "snippet,authorDetails", "maxResults": 200}
                    if page_token:
                        params["pageToken"] = page_token
                    data = await self._get(session, "liveChat/messages", params)
                    page_token = data.get("nextPageToken", page_token)
                    if not first:  # skip the backlog that existed before we connected
                        for item in data.get("items", []):
                            for event in item_to_events(item):
                                if getattr(event, "user", "").lower() in self.ignore:
                                    continue
                                result = self.sink(event)
                                if asyncio.iscoroutine(result):
                                    await result
                    first = False
                    wait = max(self.cfg.poll_interval_s, data.get("pollingIntervalMillis", 0) / 1000)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.warning("YouTube chat poll failed: %s", exc)
                    wait = 15.0
                await asyncio.sleep(wait)
        self.connected = False

    async def aclose(self) -> None:
        self._closing = True
