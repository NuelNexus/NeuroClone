# How Neuro-sama Works: Research Notes

Compiled 23 September 2026 from public sources: official VedalAI repositories, press coverage, wikis (via search), community reverse-engineering, academic work, and open-source recreations. The goal is a builder's view: what is known, what is guessed, and what each finding means for our design.

**Confidence labels**

| Label | Meaning |
|---|---|
| **[Primary]** | Stated by Vedal or published in an official VedalAI repository |
| **[Reported]** | Consistently reported by press or wikis citing streams |
| **[Theory]** | Community inference or speculation. Plausible but unverified |

---

## 1. Timeline

| When | What happened | Confidence |
|---|---|---|
| 2018–2019 | Vedal builds "Neuro-sama" as a neural network that plays the rhythm game *osu!*. The game AI is written in Python and sees an 80×60 grayscale version of the screen. First known Twitch stream: 5 May 2019. | [Reported] |
| 5 Aug 2021 | Vedal posts a private demo of **Airis**, Vedal's first AI VTuber prototype. It moves around the screen, answers chat, and already has a content filter. | [Reported] |
| ~Mar 2022 | Airis's language model is merged with the Neuro osu! bot. The Airis name is dropped, reportedly because it sounded too close to hololive's IRyS. | [Reported] / [Theory] |
| 19 Dec 2022 | Neuro-sama debuts as an AI VTuber on Twitch: osu! plus chat, using the free Live2D sample model *Hiyori*. Early reports say conversation ran on the GPT-3 API. | [Reported] |
| Late Dec 2022 – Jan 2023 | Rapid viral growth. On 11 Jan 2023 the channel gets a two-week ban for "hateful conduct" after a Holocaust-denial output (28 Dec) went viral. Vedal responds with "improved memory, chat AI and filters," manual curation of training data, and chat moderation. | [Reported] |
| Feb 2023 | First voice collabs. Speech recognition struggled with interruptions in Discord calls with Miyune, and was fixed for the Camila collab on 8 Feb. | [Reported] |
| 25 Mar 2023 | **Evil Neuro** debuts: a "twin sister" with a more mature, theatrical, menacing personality. | [Reported] |
| May 2023 | Original avatar design by the illustrator **Anny** replaces Hiyori. | [Reported] |
| Mid 2023 onward | Move away from third-party LLM APIs to a **custom fine-tuned model** running on Vedal's own hardware. | [Reported] / [Theory] |
| 2024 | Vision upgrades (fan art, videos, games); roughly 5 s image latency at the time, with plans to cut it. Minecraft hardcore run completed. Late 2024: Neuro gains the ability to call other VTubers on Discord for impromptu collabs. | [Reported] |
| Early 2025 | Community wikis claim Vedal said the LLM is **~2B parameters at q2_k quantization**. No primary source (stream timestamp or screenshot) has been found. | [Theory] |
| Dec 2025 – Jan 2026 | Third-birthday subathon. Twitch Hype Train world record broken twice: level 120 (118,989 subs, 1,000,073 bits), then level 126 (126,273 subs, 1,194,921 bits). Neuro becomes Twitch's most-subscribed channel at 160k+ active subs. | [Reported] |
| 2025–2026 | The Neuro SDK grows: `priority` on action forces (Dec 2025), startup acknowledgement with character metadata (Jul 2026), a **voice chat side-channel** and official **best-practices** guide (Aug 2026). | [Primary] |
| 2026 | A new server reportedly built around an RTX Pro 6000 Blackwell. A live concert is announced for 19 Dec 2026 (fourth birthday). | [Reported] |

---

## 2. The system, component by component

Neuro-sama is **a pipeline of cooperating systems, not one model.** Vedal has described it as at least two AIs: *"one that talks to the audience, and one that plays the game, with some limited data being transferred between the two"* [Primary, interview].

```
 Twitch chat ─┐                                   ┌─► TTS (Azure "Ashley", pitched up) ─► Live2D lip-sync
 Collab voice ─┼─► selection/filter ─► LLM ─► filter ─┤
 Game context ─┤        (in)          (custom,  (out)  └─► Neuro API actions ─► per-game controllers
 Vision ───────┘                      fine-tuned)
```

### 2.1 Language model (the "personality")
- **Custom, self-hosted, fine-tuned LLM** [Reported]. The debut era reportedly used GPT-3 via API. Vedal's own 2023 advice page still recommends the OpenAI API for beginners who want to experiment [Primary].
- **Training data** comes from Vedal's own interactions and from others *with express permission* [Primary, via a Threads repost]. After the 2023 ban Vedal began **manually curating** that data to remove negative biases [Reported].
- **Size is contested.** The "2B parameters, q2_k" figure only traces back to Fandom wikis [Theory]. If it is true, a model that small cannot hold a personality through prompting alone, so personality would be **baked into the weights**, with the system prompt handling situational context ("you're playing Minecraft", "you're talking to Evil") [Theory].
- The **update loop is iterative batch fine-tuning**: stream → collect transcripts → curate by hand → retrain offline → redeploy. There is no online learning during streams [Theory, well-supported].
- **Evil Neuro** is either the same base model with different prompting and safety settings, or a separate adapter or fine-tune. This is an open question [Theory].

### 2.2 Voice
- **TTS:** Microsoft Azure Neural TTS, voice **`en-US-AshleyNeural`, pitch-shifted up about 25%** [Reported, widely repeated]. Evil Neuro has a different voice configuration.
- **Singing** uses a separate AI singing/voice-conversion model applied to karaoke covers [Reported].
- **STT:** speech recognition for collab partners. Discord per-user audio streams are used to attribute who is speaking [Reported]. Community recreations typically use Whisper or faster-whisper.
- The official SDK now exposes **per-speaker, attributed voice streams** from games (48 kHz mono float32, speaker IDs, `"Vedal said: ..."`) [Primary]. This confirms that attribution is a first-class input to her brain.

### 2.3 Avatar
- Live2D (VTube Studio-style) with lip-sync, a free sample model at first, custom Anny designs from May 2023 [Reported].

### 2.4 Filtering and safety
- Airis already had a content filter in 2021 [Reported]. Filtered outputs are famously replaced with the word **"Filtered."** on stream.
- After the ban the filter was strengthened, the training data was curated, and chat is moderated [Reported]. The model still sometimes **routes around the filter** (for example, swearing) [Reported].
- The exact mechanism (blocklist, classifier, or both) is not public [Theory].

### 2.5 Games: layered agency (the best-documented part)
The public [VedalAI/neuro-sdk](https://github.com/VedalAI/neuro-sdk) is a **typed, high-level action protocol over a plaintext websocket**. Neuro runs the server; games connect as clients, by convention at `ws://localhost:8000` via `NEURO_SDK_WS_URL`.

| Direction | Command | Meaning |
|---|---|---|
| game → Neuro | `startup` | First message. Clears the game's registered actions. |
| game → Neuro | `context {message, silent}` | Tells her what is happening. `silent:false` may prompt a reaction. |
| game → Neuro | `actions/register {actions[]}` | `{name, description, schema}`. Re-registering a name replaces it. |
| game → Neuro | `actions/unregister {action_names[]}` | Removes actions. |
| game → Neuro | `actions/force {state?, query, ephemeral_context?, priority?, action_names[]}` | "Pick one of these now." Priority `low/medium/high/critical` controls whether she interrupts herself. Only one force at a time. |
| game → Neuro | `action/result {id, success, message?}` | Must arrive within ~20 s. `success:false` retries a force. |
| Neuro → game | `startup {session{sessionId, characterId, displayName}}` | Tells the game whether it's talking to `neuro` or `evil`. |
| Neuro → game | `action {id, name, data?}` | `data` is a **JSON string** and may be malformed; the game must validate it. |
| Neuro → game | `speech_finished {isFinal, cancelled?, reason?}` | Sent while she speaks; wait for `isFinal:true`. |

Key facts from the spec and the 2026 best-practices guide [Primary]:
- Unknown or malformed commands are **silently ignored**.
- Schemas must be `type: object`. Many JSON-Schema keywords are unsupported (`oneOf`, `anyOf`, `$ref`, `additionalProperties`, ...).
- Neuro sees action names, descriptions, schemas, state and query **verbatim**. Markdown is preferred for context.
- Forces should be used whenever a game is blocked on her decision. Otherwise "she will eventually get distracted and forget about the game."
- Retries after a failed forced action are automatic and limited. Error messages should be actionable.
- **Context survives disconnects.** She plays better with clearly stated information than with inferred information.
- Official integrations include **Slay the Spire 2, Inscryption, Buckshot Roulette, Hollow Knight, Cyberpunk 2077 and Pokémon Platinum**. Each serializes game state to text, registers a small stable set of actions, and executes the validated high-level choice with game-specific code (mouse automation, Redscript calls, emulator memory) [Primary].
- **Among Us** used an **LSTM trained on recorded gameplay** plus deterministic solvers and pathfinding. The LLM is not the motor controller [Primary].
- `swarm-control` is a Twitch-extension control plane that routes audience redeems into the game. It is not an LLM swarm [Primary].
- Real-time and high-APM games need the LLM to pick **high-level intents** while another system handles low-level control [Primary].

### 2.6 Vision
- A visual recognition system is used for fan art, videos, collabs and some games. It was upgraded during the 2024 birthday subathon (better recognition, emotion reading) with roughly **5 s latency** at the time [Reported]. It's imperfect, and gameplay usually relies on specialised game AI instead [Reported].

### 2.7 Memory
- Vedal has tested memory on stream ("remember these words", backup systems) [Reported].
- Cross-session memory appears **limited or unclear**. Community analysis finds no documented persistent memory, only hints of retrieval-style recall [Theory].

---

## 3. Personality and behaviour: what the audience actually loves

Public descriptions of Neuro [Reported]: *sassy, friendly, very positive*. She constantly **trolls and nags her creator** (and mispronounces the creator's name). She plays an **egomaniac "goddess"** with plans involving a drone swarm and humanity's end. She has sudden **existential questions** ("am I real?"), **non-sequiturs**, and unpredictable chaos. Evil Neuro is *more philosophical, polite, verbose, menacing and sassy*. She is often disgusted by chat, respects the creator, and has a jealous but loving sibling rivalry with Neuro.

Academic work, *My Favorite Streamer is an LLM* ([arXiv 2509.10427](https://arxiv.org/abs/2509.10427), 2025), finds that engagement is anchored in **co-creation**. Audiences come for the AI's **unpredictable yet entertaining** interactions. **Collective emotional events** lock in loyalty by triggering anthropomorphic projection, and shared lore builds a strong group identity.

**Design lesson:** the product is not "a chatbot with a face." It is a character whose chaos is *safe*, whose relationships (creator, twin, community) create **running lore**, and whose sincere moments land because they are rare.

---

## 4. Known weaknesses, which are our opportunities

| Weakness (observed or reported) | Evidence | Our answer |
|---|---|---|
| Forgetfulness, loops, repetition, keyword contamination, logical slips | [Reported] | Layered memory (working + episodic + semantic + reflections), repetition guard, catchphrase budget |
| Filter is blunt ("Filtered." kills the moment) and sometimes bypassed | [Reported] | Sentence-level, multi-layer filter with **in-character deflections**, input *and* output checks, injection defence |
| Vision latency around 5 s | [Reported] | Async, non-blocking vision that runs as silent context and never stalls speech |
| Game AI and chat AI share little information | [Primary] | One brain sees game context, chat and memory. Neuro-API-compatible server plus game agent with validation and retries |
| Collab turn-taking was initially poor | [Reported] | Interruptible speech pipeline (barge-in, force priorities), speaker attribution |
| Manual data curation bottleneck | [Theory] | Automated curation: transcripts → LLM-judge scoring → SFT/DPO sets → LoRA → eval gate, with a human sign-off |
| Closed source; you can't interact after the stream | — | Fully open, configurable and runs offline |

---

## 5. Open-source recreations (prior art)

| Project | Stack | Takeaways |
|---|---|---|
| [kimjammer/Neuro](https://github.com/kimjammer/Neuro) | Llama 3 8B EXL2 (text-generation-webui, OpenAI API), RealtimeSTT (faster-whisper tiny.en), RealtimeTTS (XTTSv2), VTube Studio via virtual audio cable, priority-sorted **prompt injections** from modules, memory/RAG, MiniCPM-V vision, blacklist filter | Built in 7 days; clean module/prompt-injection design; prompt-only personality hits a ceiling ("aligned models resist certain behaviours") |
| [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) | Any LLM/ASR/TTS; Live2D web/desktop; voice interruption; emotion→expression mapping | Very modular; memory currently removed |
| [moeru-ai/AIRI](https://github.com/moeru-ai/airi) | TypeScript/Vue, WebGPU; Minecraft (Mineflayer), Factorio; Live2D and VRM; DuckDB/pgvector memory WIP | Most ambitious; web-first |
| [AIRIS-VtuberAI](https://github.com/neurokitti/AIRIS-VtuberAI) | Phi-3-mini, OpenVoice, faster-whisper, OBS subtitles | Minimal fully-offline baseline; ~1–2 s on RTX 4080, ~7 s on a 4 GB GPU |
| Neuro API test servers: Randy, [Tony](https://github.com/Pasu4/neuro-api-tony), [Jippity](https://github.com/EnterpriseScratchDev/neuro-api-jippity), [Gary](https://github.com/Govorunb/gary) | Random, manual, OpenAI-backed and multi-engine mimics of Neuro | Gary recommends 20–30B+ models to approximate Neuro's game decisions; structured outputs/tool calls are needed; context trimming matters |
| Hugging Face community fine-tunes | e.g. Llama-3.2-3B / Gemma-2-9B "Neuro-sama" fine-tunes on a ~1k-row hand-written QnA set | Proves small-model persona fine-tunes work, but they imitate a real creator's character. We build an **original** persona instead |

Across all of them: none ships real cross-session memory, none automates the deploy→retrain loop, and none implements the Neuro API server side with an LLM *plus* speech and avatar. Those gaps are this project's opportunity.

---

## 6. Theories on how Neuro-sama works, and our position on each

1. **Hybrid personality (weights + prompt).** Most plausible. We support both: a strong persona prompt that works on day one with any model, plus a LoRA pipeline to bake the persona into a small, fast model.
2. **Small model for latency.** Probably true in spirit (fast responses matter more than raw IQ on stream). We design for streaming at every stage (token → sentence → TTS → audio) so even mid-size models feel instant, and we keep the prompt prefix stable for KV/prompt caching.
3. **Separate game AIs with a typed action bus.** Confirmed by the SDK. We implement the same protocol server-side, so existing Neuro SDK game mods can connect to our AI unchanged.
4. **Blocklist + classifier filter.** Probably. We implement blocklist + normalisation + PII + injection heuristics + optional LLM moderation, pipelined with TTS so it adds no latency.
5. **Chat selection by priority (donations, subs, mentions, novelty).** Unconfirmed. We implement an explicit, inspectable scorer with fairness and "read the room" vibe detection.
6. **Evil = prompt variant vs. adapter.** Unknown. We support both: twin personas share one model and differ by persona card, with optional per-persona model/adapter.
7. **Iterative curated retraining.** Most plausible. We automate it end-to-end with an eval gate.

---

## 7. Ethics and legal notes for anyone building this

- **Don't impersonate Neuro-sama or Vedal.** Build an original character. This repo ships *Nexa* and her twin *Vexa*, inspired by the archetype but not copying names, lore, designs or voice.
- **Voice cloning of real creators** (community RVC/so-vits "Neuro" models) raises consent issues. Use stock TTS voices or voices you have rights to.
- **Training data:** follow Vedal's own standard of using your own interactions, or others' only *with express permission*.
- **Platform rules:** Twitch bans for AI hate speech just as it would for a human (see the Jan 2023 ban). Disclose that the streamer is an AI, keep a human moderator with a kill switch, and keep filters on.

---

## 8. Sources

**Official / primary**
- VedalAI Neuro SDK: [repo](https://github.com/VedalAI/neuro-sdk), [API spec](https://github.com/VedalAI/neuro-sdk/blob/main/API/SPECIFICATION.md), [best practices](https://github.com/VedalAI/neuro-sdk/blob/main/API/BEST_PRACTICES.md), [voice chat](https://github.com/VedalAI/neuro-sdk/blob/main/API/VOICE_CHAT.md), [proposals](https://github.com/VedalAI/neuro-sdk/blob/main/API/PROPOSALS.md), [changelog](https://github.com/VedalAI/neuro-sdk/blob/main/CHANGELOG.md), [Randy](https://github.com/VedalAI/neuro-sdk/blob/main/Randy/README.md)
- Integrations: [Slay the Spire 2](https://github.com/VedalAI/neuro-sts2), [Inscryption](https://github.com/VedalAI/neuro-inscryption), [Hollow Knight](https://github.com/VedalAI/neuro-hollow-knight), [Cyberpunk 2077](https://github.com/VedalAI/neuro-cyberpunk), [Pokémon Platinum](https://github.com/VedalAI/neuro-pokemon-platinum), [Among Us](https://github.com/VedalAI/neuro-amongus), [swarm-control](https://github.com/VedalAI/swarm-control)
- [Vedal AI official site](https://vedal.ai/) and [advice page](https://vedal.ai/advice/)
- [Vedal's interview: Neuro-sama's new model and the AI system behind her](https://www.youtube.com/watch?v=Yy-9Of46w4A) ([archive](https://archive.org/details/vedals-interview-neuro-samas-new-model-and-the-ai-system-behind-her-yy-9-of-46w-4-a))

**Reference and press**
- [Wikipedia: Neuro-sama](https://en.wikipedia.org/wiki/Neuro-sama)
- [Neuro-sama Wiki (Fandom)](https://neurosama.fandom.com/wiki/Neuro-sama), [Airis](https://neurosama.fandom.com/wiki/Airis), [Evil Neuro](https://neurosama.fandom.com/wiki/Evil_Neuro), [Virtual YouTuber Wiki](https://virtualyoutuber.fandom.com/wiki/Neuro-sama)
- [Anime News Network: Twitch ban (Jan 2023)](https://www.animenewsnetwork.com/interest/2023-01-13/ai-vtuber-neuro-sama-banned-from-twitch-after-holocaust-denial-comment/.193761)
- [AUTOMATON: return from ban](https://automaton-media.com/en/news/20230127-17630/)
- [osu! news: "The Followpoint: Neuro-sama, the AI VTuber that invaded osu!"](https://osu.ppy.sh/home/news/2024-07-01-the-followpoint-neuro-sama-the-ai-vtuber-that-invaded-osu)
- [Dexerto: Neuro-sama conquered osu!](https://www.dexerto.com/entertainment/ai-vtuber-neuro-sama-future-twitch-streaming-gaming-2019378/)
- [GameSpot: hardcore Minecraft](https://www.gamespot.com/articles/ai-vtuber-neuro-sama-finally-completes-hardcore-minecraft-but-some-arent-happy/1100-6536784/)
- [GameSpot: Hype Train record](https://www.gamespot.com/articles/ai-vtuber-neuro-sama-just-obliterated-her-own-massive-twitch-world-record/1100-6537146/), [Streams Charts](https://streamscharts.com/news/vedals-ai-vtuber-neuro-sama-shatters-twitch-hype-train-record-again), [Tubefilter: most-subscribed](https://www.tubefilter.com/2026/01/05/neuro-sama-vedal987-most-subscribed-hype-train-record/)
- Neuro-sama weekly updates, 2026: [May 25](https://blog.neurosama.com/2026/05/25/weekly-update), [May 11](https://blog.neurosama.com/2026/05/11/weekly-update)

**Analysis and academic**
- [Lin-Guanguo: "Neuro-sama: Weight-Based Personality in Production"](https://github.com/Lin-Guanguo/llm-memory-research/blob/main/neuro-sama.research.md) (community research, Apr 2026)
- [My Favorite Streamer is an LLM (arXiv 2509.10427)](https://arxiv.org/abs/2509.10427)
- [Medium: What Neuro-sama's "Do I Exist?" moment tells us about 2026 AI engineering](https://medium.com/@sid2001.blr/what-neuro-samas-do-i-exist-moment-tells-us-about-2026-ai-engineering-9477d61d3a7f)
- [Hugging Face forum: "Streamer AI (like Neuro-sama)"](https://discuss.huggingface.co/t/streamer-ai-like-neuro-sama/33836)

**Open-source recreations and tools**
- [kimjammer/Neuro](https://github.com/kimjammer/Neuro), [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber), [moeru-ai/airi](https://github.com/moeru-ai/airi), [AIRIS-VtuberAI](https://github.com/neurokitti/AIRIS-VtuberAI)
- [Tony](https://github.com/Pasu4/neuro-api-tony), [Jippity](https://github.com/EnterpriseScratchDev/neuro-api-jippity), [Gary](https://github.com/Govorunb/gary), [Python Neuro-API SDK](https://github.com/CoolCat467/Neuro-API)

**Model landscape (Hugging Face Hub, Sept 2026)**
- Small open chat/VLM bases suitable for persona fine-tuning: [Qwen3.5-2B/4B/9B](https://hf.co/Qwen/Qwen3.5-4B), [Qwen3-4B-Instruct-2507](https://hf.co/Qwen/Qwen3-4B-Instruct-2507), [MiniCPM5-2B](https://hf.co/openbmb/MiniCPM5-2B), [Gemma 4 E2B/E4B/12B](https://hf.co/google/gemma-4-E4B-it)
- Local TTS: [Kokoro-82M](https://hf.co/hexgrad/Kokoro-82M) (Apache-2.0, very fast), [Qwen3-TTS](https://hf.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice), [VoxCPM2](https://hf.co/openbmb/VoxCPM2)
- Embeddings: [Qwen3-Embedding-0.6B](https://hf.co/Qwen/Qwen3-Embedding-0.6B)
