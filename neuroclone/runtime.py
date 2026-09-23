"""Assembles every component from a Config and runs the stream."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import random
import time
from typing import Optional

from .avatar.vtube_studio import VTubeStudio
from .chat.console import ConsoleInput
from .chat.selector import ChatSelector
from .chat.twitch import TwitchChat
from .chat.youtube import YouTubeChat
from .conductor import Character, Conductor
from .config import Config
from .emotion import EmotionEngine
from .events import EventBus
from .games.agent import GameAgent
from .games.neuro_api import NeuroApiServer
from .games.voice_chat import VoiceChatHub
from .llm import create_llm
from .memory import MemoryManager, MemoryStore, create_embedder
from .overlay.server import OverlayServer
from .persona import BitTracker, Persona, load_persona
from .prompt_builder import PromptBuilder
from .repetition import RepetitionGuard
from .safety.filter import InputFilter, LLMModerator, OutputFilter, build_blocklist
from .speaker import Speaker, SpeakerHooks
from .speech.audio import create_player
from .speech.stt import MicrophoneListener, create_transcriber
from .speech.tts import SilentTTS, TTSError, create_tts
from .transcript import TranscriptLogger
from .vision import VisionModule

log = logging.getLogger("neuroclone")


class Runtime:
    def __init__(self, cfg: Config, *, console: bool = False, print_captions: bool = False,
                 chat_sources: bool = True, servers: bool = True) -> None:
        self.cfg = cfg
        self.console = console
        self.print_captions = print_captions
        self.chat_sources = chat_sources
        self.servers = servers
        self.bus = EventBus()
        self.conductor: Optional[Conductor] = None
        self.memory: Optional[MemoryManager] = None
        self.games: Optional[NeuroApiServer] = None
        self.overlay: Optional[OverlayServer] = None
        self.avatars: list[VTubeStudio] = []
        self.sources: list = []
        self.closables: list = []
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    def submit(self, event: object) -> None:
        if self.conductor is not None:
            self.conductor.submit(event)

    # ------------------------------------------------------------------ assembly
    async def setup(self) -> Conductor:
        cfg = self.cfg
        persona = load_persona(cfg.persona, cfg.personas_dir, cfg.creator)
        twin: Optional[Persona] = load_persona(cfg.twin, cfg.personas_dir, cfg.creator) if cfg.twin else None
        self.llm = create_llm(cfg.llm)
        self.utility_llm = create_llm(cfg.utility_llm) if cfg.utility_llm else self.llm
        self.closables += [self.llm] + ([self.utility_llm] if self.utility_llm is not self.llm else [])

        blocklist = build_blocklist(cfg.safety)
        input_filter = InputFilter(cfg.safety, blocklist)
        moderator = (LLMModerator(self.utility_llm, cfg.safety.moderation_timeout_s, cfg.safety.moderation_fail_closed)
                     if cfg.safety.llm_moderation else None)

        if cfg.memory.enabled:
            store = MemoryStore(cfg.memory.path)
            embedder = create_embedder(cfg.memory.embedder, cfg.memory.embed_base_url or cfg.llm.base_url,
                                       cfg.memory.embed_model, cfg.memory.embed_api_key or cfg.llm.api_key)
            self.memory = MemoryManager(cfg.memory, store, embedder, self.utility_llm, character=persona.name)
            try:
                await self.memory.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("memory re-embedding skipped: %s", exc)

        names = [persona.name] + ([twin.name] if twin else [])
        selector = ChatSelector(cfg.selector, names)
        transcriber = create_transcriber(cfg.stt)

        if cfg.games.enabled and self.servers:
            self.games = NeuroApiServer(cfg.games, persona.id, persona.name, self.submit, self.bus)
            if cfg.games.voice_chat:
                self.games.voice = VoiceChatHub(transcriber, self.submit, self.bus)
        voice_hub = self.games.voice if self.games is not None else None

        characters = []
        for idx, p in enumerate([persona] + ([twin] if twin else [])):
            characters.append(self._character(p, twin if idx == 0 else persona, idx == 0, blocklist, moderator, voice_hub))

        game_agent = (GameAgent(self.llm, self.games, persona.system_prompt(twin, compact=cfg.prompt_style == "compact"),
                                cfg.games) if self.games else None)
        session_id = self.memory.session_id if self.memory else time.strftime("%Y%m%d-%H%M%S")
        transcripts = TranscriptLogger(cfg.logging.transcripts_dir, session_id)
        vision = None
        if cfg.vision.enabled:
            vision_llm = create_llm(cfg.vision.llm) if cfg.vision.llm else self.llm
            if vision_llm is not self.llm:
                self.closables.append(vision_llm)
            vision = VisionModule(cfg.vision, vision_llm, self.submit)
            self.sources.append(vision)

        self.conductor = Conductor(
            cfg, llm=self.llm, characters=characters, selector=selector, input_filter=input_filter, bus=self.bus,
            memory=self.memory, games=self.games, game_agent=game_agent, transcripts=transcripts, vision=vision,
            creator=persona.creator, rng=random.Random(cfg.llm.seed), on_quit=self.request_stop,
        )

        if self.chat_sources:
            if cfg.chat.twitch.channel:
                self.sources.append(TwitchChat(cfg.chat.twitch, self.submit, cfg.chat.ignore_users))
            if cfg.chat.youtube.video_id:
                self.sources.append(YouTubeChat(cfg.chat.youtube, self.submit, cfg.chat.ignore_users))
        if self.console:
            self.sources.append(ConsoleInput(self.submit, creator=persona.creator))
        if transcriber is not None:
            main = characters[0]
            self.sources.append(MicrophoneListener(
                cfg.stt, transcriber, self.submit, cfg.stt.speaker or persona.creator,
                ai_speaking=lambda: self.conductor.speaking is not None,
                recent_ai_text=lambda: " ".join(list(main.repetition.recent)[-3:]),
            ))
        if cfg.overlay.enabled and self.servers:
            self.overlay = OverlayServer(cfg.overlay, self.bus, self.conductor, self.memory)
        if self.print_captions:
            self.bus.subscribe(self._print_caption)
        return self.conductor

    def _character(self, p: Persona, twin: Optional[Persona], is_main: bool, blocklist, moderator,
                   voice_hub: Optional[VoiceChatHub]) -> Character:
        cfg = self.cfg
        emotion = EmotionEngine()
        repetition = RepetitionGuard()
        stage_twin = twin if cfg.twin else None
        builder = PromptBuilder(p, stage_twin, compact=cfg.prompt_style == "compact")
        try:
            tts = create_tts(cfg.tts, p)
        except TTSError as exc:
            log.error("%s: %s; falling back to silent TTS", p.name, exc)
            tts = SilentTTS(cfg.tts.chars_per_second)
        self.closables.append(tts)
        player = create_player(cfg.audio)
        self.closables.append(player)
        avatar = None
        url = cfg.avatar.url if is_main else cfg.avatar.twin_url
        if cfg.avatar.enabled and url:
            avatar = VTubeStudio(dataclasses.replace(cfg.avatar, url=url, token_path=cfg.avatar.token_path
                                                     if is_main else cfg.avatar.token_path + ".twin"),
                                 hotkeys=p.avatar_hotkeys)
            self.avatars.append(avatar)

        games = self.games

        async def speech_finished(_speaker: str, is_final: bool, cancelled: bool, reason) -> None:
            if is_main and games is not None:
                await games.speech_finished(is_final, cancelled, reason)
            if cancelled and voice_hub is not None:
                await voice_hub.cancel()

        async def speaking(_speaker: str, is_speaking: bool) -> None:
            if voice_hub is not None and is_main:
                await voice_hub.speaking(is_speaking)

        def sentence(_speaker: str, _text: str, _tags, clip) -> None:
            if voice_hub is not None and is_main:
                voice_hub.send_clip(clip)

        hooks = SpeakerHooks(
            on_sentence=sentence,
            on_speaking=speaking,
            on_speech_finished=speech_finished,
            on_level=avatar.set_mouth if avatar else None,
            on_expression=avatar.express if avatar else None,
        )
        output_filter = OutputFilter(cfg.safety, blocklist, secrets=[p.system_prompt(stage_twin)])
        speaker = Speaker(p, tts, player, output_filter, emotion=emotion, repetition=repetition, bus=self.bus,
                          moderator=moderator, hooks=hooks, max_sentences=cfg.conductor.max_sentences,
                          first_clause_chars=cfg.conductor.first_clause_chars)
        return Character(p, speaker, builder, emotion, repetition, BitTracker(p), avatar)

    def _print_caption(self, topic: str, data: dict) -> None:
        if topic == "caption":
            print(f"  {data['speaker']} [{data.get('emotion') or 'neutral'}]: {data['text']}", flush=True)
        elif topic == "chat" and not data.get("allowed", True):
            print(f"  (chat from {data['user']} ignored: {data['reason']})", flush=True)

    # ------------------------------------------------------------------ lifecycle
    def request_stop(self) -> None:
        if self.conductor is not None:
            self.conductor.stop()
        self._stopped.set()

    async def run(self) -> None:
        conductor = await self.setup()
        try:
            if self.games is not None:
                await self.games.start()
            if self.overlay is not None:
                await self.overlay.start()
            for avatar in self.avatars:
                await avatar.start()
            for source in self.sources:
                self._tasks.append(asyncio.ensure_future(self._run_source(source)))
            await conductor.run()
        finally:
            await self.aclose()

    async def _run_source(self, source) -> None:
        try:
            await source.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one broken source must not stop the stream
            log.error("%s stopped: %s", type(source).__name__, exc)

    async def aclose(self) -> None:
        for source in self.sources:
            try:
                await source.aclose()
            except Exception:  # noqa: BLE001
                pass
        for task in self._tasks:
            task.cancel()
        if self.memory is not None:
            try:
                summary = await asyncio.wait_for(self.memory.end_session(), timeout=60)
                if summary.get("summary"):
                    log.info("stream summary saved to memory")
            except Exception as exc:  # noqa: BLE001
                log.warning("could not write the stream summary: %s", exc)
            await self.memory.aclose()
        for avatar in self.avatars:
            await avatar.aclose()
        if self.overlay is not None:
            await self.overlay.stop()
        if self.games is not None:
            await self.games.stop()
        for item in self.closables:
            try:
                await item.aclose()
            except Exception:  # noqa: BLE001
                pass
