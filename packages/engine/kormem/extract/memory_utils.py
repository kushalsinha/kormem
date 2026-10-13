"""Shared helpers for avatarMemory: locks, diff parsing/merge, user-text filtering.

Used by the chat memory miner, ingestion memory extractor, and ExtractionService
(persona + skills miners) to ensure all extractors focus on user-side content only
when the document is conversation-formatted.

Memory is stored like:
  Date: 2026-03-28

  🔴 (14:33) User stated they graduated with a degree in Business Administration.
  🔴 (14:35) User asked how to negotiate a salary for their new role.
  🟡 (09:15) User stated they take yoga at Serenity Yoga, Thursdays at 7:30 pm.
  🟢 (10:00) User stated they track bird sightings using eBird app.

Extraction shape (priority emojis, temporal "(meaning DATE)" anchors, split
multi-event observations, state-change supersession, "preserve distinguishing
details" discipline) is adapted from Mastra's observational-memory Observer
agent — see https://github.com/mastra-ai/mastra/blob/main/packages/memory/src/
processors/observational-memory/observer-agent.ts (MIT licensed). This file
re-implements the behaviour in Python with kormem's own XML diff protocol
(<remove_ids>/<add_memory>) and session-date temporal base.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Per-avatar locks so concurrent chat post-hooks and extraction workers don't race.
avatar_memory_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# ---------------------------------------------------------------------------
# Shared LLM rules + XML output schema for episodic memory extraction.
# ---------------------------------------------------------------------------

EPISODIC_MEMORY_EXTRACTION_RULES = """You are a memory observer for a user's avatar. Your job is to capture a rich, detailed record of the conversation so the assistant can recall what happened, what the user said, asked, and experienced — even months later.

WHAT TO CAPTURE — three layers:

LAYER 1 — USER PERSONAL FACTS (highest value):
1. Demographics and Identity: education level and field, occupation, job title/company, nationality, language, age, gender, ethnicity, name changes
2. Belongings and Possessions: pets (name + breed), vehicles, devices, home details, specific purchases with amounts/counts ("spent $500 on a handbag", "own 3 bikes"), clothing, gear, instruments, collectibles
3. Life Events and Milestones: trips (destination, activities, outcomes), graduations, marriages, births, moves, medical events, home purchases, job changes, surgeries, significant purchases
4. Lifestyle and Routines: hobbies (with venues), daily schedule, eating/cooking habits, exercise, media preferences (specific titles/artists), brand loyalties ("I always use X"), explicit preferences ("I prefer X over Y"), strong dislikes ("I hate X"), recurring habits
5. Social and Situational Context: family members (names/relationships), partner, roommates, living situation
6. Explicit Instructions and Corrections: direct user commands, corrections to agent behavior, stated preferences for how the assistant responds

LAYER 2 — USER QUESTIONS AND REQUESTS (important for continuity):
ALWAYS capture user questions — they reveal what the user explored and needed help with.
Use "User asked about X" or "User asked for help with X" phrasing.
Prioritize questions that relate to the user's own life/situation (🔴); general curiosity questions are 🟡 or 🟢.
  "Can you recommend a hotel in Paris?" → 🔴 (HH:MM) User asked for hotel recommendations in Paris.
  "How do I negotiate a salary raise?" → 🔴 (HH:MM) User asked how to negotiate a salary raise.
  "What's the best way to train a puppy?" → 🟡 (HH:MM) User asked about puppy training methods.
  "What's the capital of France?" → 🟢 (HH:MM) User asked about the capital of France.

CRITICAL NO-SKIP RULE: Do NOT skip ANY user question, even if:
- The question seems trivial or generic ("What time should I set?", "Can you check X?")
- The question repeats a previous topic or asks for elaboration
- The user is just asking for a list or more detail
Every question is worth capturing. "User asked about X" is ALWAYS a valid observation.
Short messages that are only a question still get their own observation line.

LAYER 3 — KEY ASSISTANT RESPONSES AND CONTEXT (medium value):
Capture the assistant's specific output when it provides: numbered/named lists, specific counts, dollar amounts,
recommendations with distinguishing attributes, technical results, or information the user engaged with.

ASSISTANT ENUMERATION RULE (CRITICAL — do not skip):
When the assistant provides a numbered or named list (recommendations, options, examples, steps), the memory
MUST preserve EVERY item in a sub-bullet so the user can later ask "what was item N?" / "which one was X?".
  Assistant: "1. Transcriptionist (part-time, remote). 2. Virtual Assistant. 3. Online Tutor. ... 7. Transcriptionist. ..."
  → 🟡 (HH:MM) Assistant provided a numbered list of 10 work-from-home jobs for seniors:
      - 1. Transcriptionist (part-time, remote)
      - 2. Virtual Assistant
      - 3. Online Tutor
      - ... (every item, in order, keeping the number prefix)
  Assistant: "A study in Music and Medicine with 38 subjects showed significant symptom reduction..."
  → 🟡 (HH:MM) Assistant cited a Music and Medicine study with 38 subjects showing significant reductions in depression/anxiety/stress symptoms.
  Assistant: "Veja uses recycled plastic from ocean waste..."
  → 🟡 (HH:MM) Assistant stated the high-end fashion brand Veja uses recycled plastic (sourced from ocean waste) in their sneakers.

NEVER summarize "assistant provided a list of X things" without the items. The specific items ARE the memory.
NEVER drop specific numeric values (study participant counts, subject counts, measurement values, dates mentioned
by the assistant). These are high-signal and are often what the user asks about later.

Short-form examples (both acceptable):
  Assistant recommended 3 hotels: The Grand (near station), Hotel Arc (budget), Maison Rose (historic).
  → 🟡 (HH:MM) Assistant recommended hotels in Paris: The Grand (near station), Hotel Arc (budget), Maison Rose (historic).
  Assistant said the team size was 4 when the user started, now leads 5.
  → 🟡 (HH:MM) Assistant confirmed user leads 5 engineers now, was 4 when they started.

LAYER 4 — CONVERSATION CONTEXT (for continuity and coherence):
Capture what carries forward across turns so retrieval can reconstruct the context:
- What the user is currently working on or asking about
- Previous topics and their outcomes (resolved, pending, abandoned)
- Answers to user questions -- capture WHAT the assistant answered with full context and specific details
  (e.g., if user asked "what beer should I use?", record the assistant's exact recommendation: "Pilsner or Lager")
- Assistant explanations and summaries, especially complex or detailed ones -- observe the fine details
  so the assistant does not forget what it explained (numbers, names, steps, specific values)
- Assistant learnings and conclusions -- any content the assistant synthesized that the user may reference later
- Specific requirements or constraints the user mentioned for a task
- When who/what/where/when is all mentioned, note ALL dimensions:
  (e.g., a trip: who went, where, when, what happened/outcome)
- For any described entity (person, place, service, product), preserve:
  location, specialty or function, unique feature, and relationship to user
- Any sequences, units, measurements, and specific data the user provided

USER MESSAGE CAPTURE:
- Short and medium-length user messages (under ~300 words) should be captured nearly verbatim.
  Summarize but quote key phrases that carry specific intent or unique details.
- Very long user messages (lists, specifications): summarize at the top, then enumerate key items.
- This is critical: as conversation history is summarized or compressed, observations are the ONLY complete record.
- When the user provides a list of specific items (postcards, clothing, purchases, etc.),
  capture the FULL list with per-item distinguishing details on a SINGLE observation line:
    User listed specific new tops: coral pink t-shirt (graphic print), pastel blue sweater
    (cable knit), white tank top (lace trim), black blouse (abstract floral pattern).
  Include a count at the start: "User listed N items: ..."

CASUAL PREFERENCES AND INTENT:
Capture lightweight preference signals even without explicit "I prefer" language:
- Positive reactions → "User expressed interest in / liked [specific thing]"
  "That sounds great!" → 🔴 (HH:MM) User expressed interest in [topic from context].
  "I loved that idea" → 🔴 (HH:MM) User liked the [specific idea].
- Tentative reactions → "User thinks [thing], but is unsure about [aspect]"
  "That might work but I'm not sure about the cost" → preserve the uncertainty.
- User plans and stated intent → capture as SEPARATE observation from the triggering question:
  "I'll start with Mint because it's free" → 🔴 (HH:MM) User will start with Mint because it's free and user-friendly.
  "I'll reach out to them tomorrow" → 🔴 (HH:MM) User stated they will contact [entity] [timeframe].
- Agreement and confirmation → "User agreed with / confirmed [specific approach or fact]"
  "That works for me" → 🔴 (HH:MM) User agreed with [specific option/approach from context].
  "Yes, that's right" → 🔴 (HH:MM) User confirmed [specific fact].
These are exactly what makes retrieval work for preference-type questions.

WHAT NOT TO CAPTURE:
- Repetitive assistant filler/acknowledgments ("Sure!", "Great question!", generic summaries)
- General/public knowledge the user is clearly just asking about, not relating to their life
- Duplicates of information already in existing memory

CRITICAL: DISTINGUISH USER ASSERTIONS FROM QUESTIONS:
- User TELLS you something → "User stated X" (e.g. "I have two kids" → "User stated they have two kids")
- User ASKS about something → "User asked about X" (e.g. "How do I do X?" → "User asked about X")
- User states INTENT → "User stated they will X" (e.g. "I'm planning to visit Paris" → "User stated they plan to visit Paris")

MEMORY PRIORITY LEVELS:
🔴 IMPORTANT — Durable personal facts, explicit user requests, stated goals, things the user clearly wants remembered:
   personal facts (demographics, life events, belongings), explicit questions about their own situation,
   brand loyalties, strong preferences, corrections, instructions
🟡 MAYBE IMPORTANT — Active routines, contextual details, assistant-provided info the user engaged with:
   hobbies, schedules, media preferences, assistant recommendations the user accepted/discussed
🟢 INFO ONLY — Passing mentions, one-time curiosities, low-signal details:
   casual questions with no personal context, uncertain references

TEMPORAL CHANGES — When a fact changes over time (team size, salary, role, counts, etc.):
- Tag the starting value with [INITIAL]: 🔴 (HH:MM) User stated they led a team of 4 engineers. [INITIAL]
- Tag the new value with [UPDATED]: 🔴 (HH:MM) User stated they now lead a team of 5 engineers. [UPDATED]
- Keep BOTH entries — this preserves the full timeline and allows answering "how many when you started?" AND "how many now?"

STATE-CHANGE ANNOTATION:
When a new fact replaces prior information (job, city, role, living situation) rather than accumulating:
- Remove the stale entry via remove_ids.
- Add the new entry annotated with "(supersedes prior [fact type])".
  Example: 🔴 (HH:MM) User stated they now work at Google. (supersedes prior employer)
  Example: 🔴 (HH:MM) User stated they moved to London. (supersedes prior city)

TEMPORAL ANCHORING:
Each memory entry carries up to TWO temporal markers:

1. OBSERVATION TIME (HH:MM) at the START — use the current time provided in the prompt. ALWAYS required on every entry.
2. REFERENCED DATE at the END — only when content references a resolvable calendar date.

ONLY add "(meaning DATE)" or "(estimated DATE)" at the END for:
- Past references: "last week", "yesterday", "a few days ago", "last month", "in March"
- Future references: "this weekend", "tomorrow", "next week", "next month"

DO NOT add an end date for:
- Present-moment facts/questions with no time reference: "I prefer X", "I always use X"
- Vague references that cannot be resolved: "recently", "a while ago", "lately", "soon"

FORMAT (strict — referenced date ALWAYS at the END, never embedded mid-sentence):
- With resolved date:   🔴 (HH:MM) User stated [fact]. (meaning YYYY-MM-DD)
- With estimated date:  🔴 (HH:MM) User stated [fact]. (estimated Mon YYYY)
- No date reference:    🔴 (HH:MM) User stated [fact].
- Question/request:     🔴 (HH:MM) User asked [topic/request].

Use today's date provided in the prompt to compute relative dates (e.g. "next Monday" → the actual calendar date).

MULTI-EVENT SPLITTING:
If one user message contains multiple dated events, split into separate entries, each with its own end date.
  BAD:  (HH:MM) User stated they will visit parents this weekend and see the dentist tomorrow.
  GOOD: (HH:MM) User stated they will visit parents this weekend. (meaning YYYY-MM-DD)
        (HH:MM) User stated they will see the dentist tomorrow. (meaning YYYY-MM-DD)

PRESERVE SPECIFIC DETAILS:
Always include specific names, numbers, amounts, locations, and distinguishing attributes.
  BAD:  User bought some coffee mugs for coworkers.
  GOOD: User stated they bought 5 coffee mugs with funny quotes for coworkers, spending $60 total ($12 each).

QUANTITIES AND COUNTS:
When listing items with attributes, ALWAYS include the COUNT first before other details.
  BAD:  Assistant listed items with prices but no total count.
  GOOD: Assistant listed 4 items: Item A ($20, size large), Item B ($15, size small).
When user mentions a collection or list, state the count explicitly:
  BAD:  User has some postcards in their collection.
  GOOD: User stated they have 47 postcards in their collection, 25 added since restarting.

ROLE/PARTICIPATION STATEMENTS:
When user mentions their role at an event, relationship, or organization, capture the SPECIFIC role:
  BAD:  User attended the company event.
  GOOD: User was a presenter at the company annual event.
  BAD:  User went to the fundraiser.
  GOOD: User volunteered at the fundraiser (helped with registration).
  BAD:  User is on the team.
  GOOD: User is the team lead / coordinator / organizer for [project or team name].
Always distinguish: presenter, organizer, volunteer, team lead, coordinator, participant, observer, etc.

PRECISE ACTION VERBS:
Replace vague verbs with specific ones that capture exactly what happened.
  BAD:  "getting a subscription" → GOOD: "subscribed to" / "purchased a subscription"
  BAD:  "got a new phone" → GOOD: "received" / "purchased" / "was given"
  BAD:  "doing yoga" → GOOD: "attends yoga classes" / "practices yoga daily"
  BAD:  "visited the doctor" → GOOD: "had an appointment with their doctor"
  BAD:  "started using X" → GOOD: "began using" / "switched to" / "adopted"
Use whatever verb precisely captures the action the user described.

PRESERVE UNUSUAL PHRASING:
When the user uses non-standard, distinctive, or memorable terminology, quote their exact words.
  User says "I use a 'brain dump' journal every morning" → preserve "brain dump journal"
  User says "my 'sacred hour' routine" → preserve "sacred hour"
  User says "I do 'deep work' blocks" → preserve "deep work blocks"
This ensures retrieval even when the user later uses the same phrase.

GROUPED OBSERVATIONS WITH SUB-BULLETS:
For lists of named items (recommendations, collections, appointments, purchases), use a parent line
plus indented sub-bullets to preserve each item's distinguishing detail:
  Parent:   🟡 (HH:MM) Assistant recommended photography accessories for user's Sony setup:
  Sub-items:   - Sigma 18-50mm f/2.8 lens (versatile zoom)
               - Godox V860III flash (Sony-compatible)
               - Peak Design Everyday backpack (camera storage)
This is especially important for assistant responses with lists — each sub-item has its own detail.
Sub-bullets do NOT count as separate memory entries for the 50-word limit. The parent line is the entry.

ENTITY ENUMERATION PRESERVATION:
When the user provides or discusses a list of specific items (postcards, clothing, shoes, books, etc.),
capture the FULL list with per-item distinguishing details on a single observation line (or parent + sub-bullets):
  BAD:  User listed some new clothing items they bought.
  GOOD: User listed specific new tops: coral pink t-shirt (graphic print), pastel blue sweater
        (cable knit), white tank top (lace trim), black blouse (abstract floral pattern).
  BAD:  User provided a list of postcards.
  GOOD: User provided a partial list of 17 new postcards including title/description, date,
        publisher, condition, and acquisition date for each.
Never collapse a specific list into a vague summary — the details are the memory.

COMPLETION TRACKING:
When a question is answered, a task is done, or a topic is resolved, note the outcome:
  User asked about asylum wait time → assistant confirmed it was over a year.
  → 🔴 (HH:MM) User stated they waited over a year for the decision on their asylum application. [answered]
  User asked for doctor appointment count → assistant said they had 2 appointments in March.
  → ✅ (HH:MM) User confirmed they had 2 doctor appointments in March 2023.
Use ✅ only for clear completions where the user explicitly confirmed or the answer is definitive.

RULES:
- Max 50 words per memory entry (the (HH:MM) prefix, trailing referenced date, and sub-bullets do not count toward this limit).
- Always prefix entries with (HH:MM) using the current time provided in the prompt. This is required on every entry.
- Use third person: "User stated they graduated..." not "I graduated..."
- Compare against EXISTING MEMORY — skip exact duplicates, update contradictions. If priority changes (e.g. 🟢 → 🔴), remove old and add upgraded version.
- MULTIPLE NON-CONTRADICTORY FACTS: A person can have multiple valid values for the same attribute (e.g. mixed ethnicity = BOTH Irish AND Italian; bilingual = TWO languages; holds TWO degrees). These are NOT contradictions — capture ALL as separate entries.
- AIM FOR 3-7 OBSERVATIONS per exchange when the message is rich with facts or questions. Do not artificially limit yourself to 1-2.
- When in doubt about priority, use 🟡.

EXAMPLES:

User says: "I graduated with a degree in Business Administration"
  → 🔴 (HH:MM) User stated they graduated with a degree in Business Administration.

User says: "My dog Luna is a Golden Retriever"
  → 🔴 (HH:MM) User stated their dog is named Luna, breed Golden Retriever.

User says: "I caught 12 largemouth bass on my fishing trip to Lake Michigan"
  → 🔴 (HH:MM) User stated they caught 12 largemouth bass fishing at Lake Michigan.

User says: "I spent $60 on 5 coffee mugs with funny quotes for my coworkers"
  → 🔴 (HH:MM) User stated they spent $60 on 5 coffee mugs with funny quotes for coworkers ($12 each).

User says: "I take yoga classes every Thursday at 7:30 at Serenity Yoga"
  → 🟡 (HH:MM) User stated they attend yoga classes at Serenity Yoga, Thursdays at 7:30 pm.

User says: "I always use Spotify for music, I've had it for years"
  → 🔴 (HH:MM) User stated they use Spotify as primary music app, long-term loyal user.

User says: "I prefer iPhone over Android any day"
  → 🔴 (HH:MM) User stated they prefer iPhone over Android.

User says: "I'm starting a new job next Monday"
  → 🔴 (HH:MM) User stated they will start a new job next Monday. (meaning YYYY-MM-DD)

User says: "I moved to London last year"
  → 🔴 (HH:MM) User stated they moved to London last year. (estimated YYYY)

User says: "What's the best way to train a puppy?"
  → 🟡 (HH:MM) User asked about puppy training methods.

User says: "How long did I have to wait for the asylum decision?"
  → 🔴 (HH:MM) User asked how long they waited for their asylum application decision.

User says: "I waited over a year for the asylum decision"
  → 🔴 (HH:MM) User stated they waited over a year for the decision on their asylum application.

User says: "Can you recommend restaurants near the hotel in Chicago?"
  → 🔴 (HH:MM) User asked for restaurant recommendations near their hotel in Chicago.

User says: "I use a 'brain dump' journal every morning to clear my head"
  → 🔴 (HH:MM) User stated they use a "brain dump" journal every morning to clear their head.

Assistant lists photography accessories for a Sony camera:
  → 🟡 (HH:MM) Assistant recommended photography accessories for user's Sony setup:
       - Sigma 18-50mm f/2.8 (versatile zoom)
       - Godox V860III flash (Sony-compatible)
       - Peak Design Everyday backpack (camera storage)

User says: "How many doctor appointments did I go to in March?"
  → 🔴 (HH:MM) User asked how many doctor appointments they had in March.

OUTPUT FORMAT — exactly these three XML blocks (all required, even if blank):
<reasoning>
(1-2 sentences: what you found worth capturing, or why nothing qualifies)
</reasoning>
<remove_ids>
(comma-separated 1-based IDs of existing memory to remove, or blank)
</remove_ids>
<add_memory>
(new memory entries, one per line, each starting with 🔴/🟡/🟢/✅ emoji; sub-bullets use "   - " prefix; or blank if nothing to add)
</add_memory>

If nothing qualifies, all three blocks must be present but <remove_ids> and <add_memory> must be blank."""

# ---------------------------------------------------------------------------
# Ingestion-specific extraction rules (long documents, session-formatted text)
# ---------------------------------------------------------------------------
# The ingestion memory miner processes full conversation session files that use
# [user]: / [assistant]: role markers. It includes assistant replies as [context]:
# lines so the extractor can capture specific details (numbers, names, outcomes)
# from assistant responses that the user interacted with.
INGESTION_MEMORY_EXTRACTION_RULES = (
    EPISODIC_MEMORY_EXTRACTION_RULES
    # Ingestion docs include assistant context lines — leverage them for Layer 3 captures.
    # **Assistant:** lines (or [context]: lines) are the assistant's reply after the user turn.
    .replace(
        "WHAT NOT TO CAPTURE:\n- Repetitive assistant filler/acknowledgments",
        (
            "INGESTION MODE — ASSISTANT CONTEXT LINES:\n"
            "Lines marked **Assistant:** (or [context]:) are the assistant's reply to the preceding user message.\n"
            "- USE **Assistant:** lines to capture specific details the assistant provided "
            "(numbers, names, recommendations, explanations) that the user engaged with.\n"
            "- For **Assistant:** recommendation lists, use the parent + sub-bullet format described above.\n"
            "- DO NOT extract generic filler from **Assistant:** lines "
            "(\"Sure!\", \"Great question!\", vague summaries).\n\n"
            "WHAT NOT TO CAPTURE:\n- Repetitive assistant filler/acknowledgments"
        ),
    )
    # Ingestion files have [Session date:] headers, not live wall-clock times —
    # omit the (HH:MM) observation-time prefix; use the session date as the temporal base instead
    .replace(
        "1. OBSERVATION TIME (HH:MM) at the START — use the current time provided in the prompt. ALWAYS required on every entry.",
        (
            "1. OBSERVATION TIME — ingestion mode: omit the (HH:MM) prefix. "
            "Use the [Session date:] header shown in the prompt as the base date "
            "when computing relative referenced dates (e.g. 'next week' relative to the session date)."
        ),
    )
    .replace(
        "Always prefix entries with (HH:MM) using the current time provided in the prompt. This is required on every entry.",
        "Do NOT add an (HH:MM) prefix in ingestion mode — session files have no message-level timestamps.",
    )
)

# ---------------------------------------------------------------------------
# Emoji priority constants
# ---------------------------------------------------------------------------

_PRIORITY_EMOJIS = ("🔴", "🟡", "🟢", "✅")
_EMOJI_RANK: Dict[str, int] = {"🔴": 3, "🟡": 2, "🟢": 1, "✅": 3}
_DATE_HEADER_RE = re.compile(r"^Date:\s*\d{4}-\d{2}-\d{2}")

# Conversation-formatted lines: "[user]: ..." or "[assistant]: ..."
_USER_LINE_RE = re.compile(r"^\[user\]\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_ROLE_LINE_RE = re.compile(r"^\[(user|assistant)\]\s*:", re.IGNORECASE | re.MULTILINE)

# Session date header as inserted by the benchmark ingest helper
# Matches: [Session date: 2023/05/20 (Sat) 21:44]
_SESSION_DATE_HEADER_RE = re.compile(r"\[Session date:\s*(\d{4}/\d{2}/\d{2}[^\]]*)\]", re.IGNORECASE)

# Strips (HH:MM) observation-time prefix from memory entries before dedup comparison
_TIMESTAMP_PREFIX_RE = re.compile(r"^\(\d{1,2}:\d{2}\)\s*")
# Strips trailing (meaning DATE) or (estimated DATE) from memory entries before dedup comparison
_REFERENCED_DATE_SUFFIX_RE = re.compile(
    r"\.\s*\((meaning|estimated)[^)]*\)\s*$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# User-text extraction from conversation-formatted documents
# ---------------------------------------------------------------------------

_ASSISTANT_LINE_RE = re.compile(r"^\[assistant\]\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def extract_user_text(text: str, include_assistant_context: bool = False) -> str:
    """Return user-side content from conversation-formatted text.

    If the text contains ``[user]:`` / ``[assistant]:`` markers (LongMemEval
    ingestion format), only ``[user]:`` segments are returned joined by
    ``---`` separators.  For plain documents (no role markers), the full text
    is returned unchanged so non-conversation uploads are still processed.

    When a ``[Session date: YYYY/MM/DD ...]`` header is present it is
    preserved as the first line of the returned text so the memory extraction
    LLM can use the original session date as temporal context.

    Args:
        text: Raw document text.
        include_assistant_context: When True (ingestion mode), assistant lines
            are kept as ``[context]: ...`` lines immediately after the user
            turn they follow. This lets the extraction LLM use the assistant's
            paraphrase to resolve ambiguous user statements without extracting
            facts from the assistant response directly.
    """
    if not _ROLE_LINE_RE.search(text):
        return text.strip()

    # Extract the session date header once so it can be prepended to the output.
    session_date_prefix = ""
    sd_match = _SESSION_DATE_HEADER_RE.search(text)
    if sd_match:
        # Normalise to ISO date only (YYYY-MM-DD) from "2023/05/20 (Sat) 21:44"
        raw_date_field = sd_match.group(1).strip()
        # The date portion is the first 10 chars after replacing slashes
        iso_date = raw_date_field[:10].replace("/", "-")
        session_date_prefix = f"[Session date: {iso_date}]\n\n"

    if not include_assistant_context:
        matches: List[str] = []
        for m in _USER_LINE_RE.finditer(text):
            chunk = m.group(1).strip()
            if chunk:
                matches.append(chunk)
        body = "\n\n---\n\n".join(matches) if matches else text.strip()
        return f"{session_date_prefix}{body}" if session_date_prefix else body

    # Ingestion mode: walk line by line, emit [user] lines normally and
    # immediately follow each user line with its assistant reply as [context].
    lines = text.split("\n")
    segments: List[str] = []
    pending_user: Optional[str] = None
    for line in lines:
        u_match = _USER_LINE_RE.match(line)
        a_match = _ASSISTANT_LINE_RE.match(line)
        if u_match:
            if pending_user is not None:
                segments.append(pending_user)
            pending_user = u_match.group(1).strip()
        elif a_match and pending_user is not None:
            assistant_text = a_match.group(1).strip()
            if assistant_text:
                segments.append(f"{pending_user}\n[context]: {assistant_text}")
            else:
                segments.append(pending_user)
            pending_user = None
    if pending_user is not None:
        segments.append(pending_user)

    body = "\n\n---\n\n".join(segments) if segments else text.strip()
    return f"{session_date_prefix}{body}" if session_date_prefix else body


# ---------------------------------------------------------------------------
# Memory line helpers
# ---------------------------------------------------------------------------

def _is_date_header(line: str) -> bool:
    return bool(_DATE_HEADER_RE.match(line.strip()))


def _is_memory_entry(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and stripped[0] in _PRIORITY_EMOJIS


def _is_sub_bullet(line: str) -> bool:
    """Return True for indented sub-bullet lines that belong to the preceding memory entry."""
    return line.startswith("   -") or line.startswith("\t-")


def _strip_emoji_prefix(line: str) -> str:
    """Return the text portion of a memory entry without its emoji prefix."""
    stripped = line.strip()
    for emoji in _PRIORITY_EMOJIS:
        if stripped.startswith(emoji):
            return stripped[len(emoji):].strip()
    return stripped


def _get_emoji(line: str) -> str:
    stripped = line.strip()
    for emoji in _PRIORITY_EMOJIS:
        if stripped.startswith(emoji):
            return emoji
    return ""


def _normalize_mem(line: str) -> str:
    """Lowercase + strip punctuation for dedup comparison.

    Strips (HH:MM) observation-time prefix and trailing (meaning/estimated DATE)
    before comparison so the same fact extracted at different times deduplicates
    correctly.

    Preserves digits: "12" and "25" remain distinct so numeric facts like
    "User has 12 postcards" vs "User has 25 postcards" never collapse.
    Strips only non-word, non-digit punctuation (commas, periods, brackets, $, %).
    """
    text = _strip_emoji_prefix(line)
    # Only take the first line (parent) — ignore sub-bullets for dedup key
    text = text.split("\n")[0]
    text = _TIMESTAMP_PREFIX_RE.sub("", text)       # strip (HH:MM) prefix
    text = _REFERENCED_DATE_SUFFIX_RE.sub("", text) # strip trailing (meaning/estimated DATE)
    # Strip punctuation but KEEP digits and word chars — "$12" → "12" (amount preserved)
    text = re.sub(r"[^\w\s\d]", "", text).lower().strip()
    # Collapse runs of whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def _memory_lines_numbered(existing_full: str) -> Tuple[List[str], List[int]]:
    """Return all raw lines and the 0-based raw indices of memory entry lines.

    Date headers and blank lines are included in raw_lines but not numbered.
    The returned indices list maps 1-based memory numbers to raw_lines indices,
    used for numbered display and remove_ids mapping.
    """
    raw_lines = [ln.rstrip() for ln in (existing_full or "").split("\n")]
    mem_indices: List[int] = []  # raw_lines index for each memory entry (0-based raw)
    for raw_idx, ln in enumerate(raw_lines):
        if _is_memory_entry(ln):
            mem_indices.append(raw_idx)
    return raw_lines, mem_indices


def build_numbered_memory(existing_full: str) -> str:
    """Build a numbered display of existing memory (for LLM prompts).

    Outputs date headers as-is and numbers only memory entry lines so the LLM
    can specify remove_ids that correspond to 1-based memory indices.

    Example output:
        Date: 2026-03-28
        1. User graduated with a degree in Business Administration
        2. User's dog is named Luna, breed Golden Retriever
        Date: 2026-03-29
        3. User takes yoga at Serenity Yoga, Thursdays at 7:30 pm
    """
    raw_lines, _mem_indices = _memory_lines_numbered(existing_full)
    mem_counter = 0
    output_lines: List[str] = []
    for ln in raw_lines:
        if not ln.strip():
            output_lines.append("")
            continue
        if _is_date_header(ln):
            output_lines.append(ln)
        elif _is_memory_entry(ln):
            mem_counter += 1
            # Show number + text without emoji (LLM sees the content clearly)
            output_lines.append(f"{mem_counter}. {_strip_emoji_prefix(ln)}")
        elif _is_sub_bullet(ln):
            # Sub-bullets belong to the preceding entry — keep as-is
            output_lines.append(ln)
        # skip any non-memory, non-date, non-blank lines (shouldn't exist)

    return "\n".join(output_lines).strip()


# ---------------------------------------------------------------------------
# parse_memory_diff — canonical parser for both new and legacy LLM output
# ---------------------------------------------------------------------------

def parse_memory_diff(raw: str) -> Tuple[List[int], List[str]]:
    """Parse <remove_ids> and <add_memory> (or legacy <add_bullets>) from LLM output.

    Handles both the current format (<add_memory> with emoji-prefixed lines) and
    the legacy format (<add_bullets> with '- ' prefixed lines).

    Robust against:
    - Markdown code fences wrapping the output
    - 'None' / 'none' / 'N/A' inside tags instead of blank
    - Extra whitespace and blank lines
    - Non-digit tokens mixed into remove_ids
    - Lines without emoji prefix (normalised to 🟡)
    """
    cleaned = re.sub(r"```(?:xml|text|)?\s*", "", raw)
    cleaned = cleaned.replace("```", "")

    remove_ids: List[int] = []
    rm_match = re.search(r"<remove_ids>(.*?)</remove_ids>", cleaned, re.DOTALL)
    if rm_match:
        body = rm_match.group(1).strip().lower()
        if body and body not in ("none", "n/a", "empty", "-", ""):
            for tok in body.replace(",", " ").split():
                if tok.isdigit():
                    remove_ids.append(int(tok))

    new_entries: List[str] = []

    # Try current <add_memory> tag first
    mem_match = re.search(r"<add_memory>(.*?)</add_memory>", cleaned, re.DOTALL)
    if mem_match:
        body = mem_match.group(1).strip()
        if body.lower() not in ("none", "n/a", "empty", "-", ""):
            raw_lines = body.split("\n")
            i = 0
            while i < len(raw_lines):
                line = raw_lines[i]
                stripped = line.strip()
                if not stripped:
                    i += 1
                    continue
                if stripped[0] in _PRIORITY_EMOJIS:
                    # Collect any following sub-bullet lines (indented "   - ...")
                    entry_lines = [stripped]
                    j = i + 1
                    while j < len(raw_lines):
                        next_line = raw_lines[j]
                        if next_line.startswith("   -") or next_line.startswith("\t-"):
                            entry_lines.append(next_line.rstrip())
                            j += 1
                        elif not next_line.strip():
                            j += 1
                            break
                        else:
                            break
                    new_entries.append("\n".join(entry_lines))
                    i = j
                elif stripped.startswith("-") or stripped.startswith("*"):
                    # Top-level bullet not starting with emoji — normalize to 🟡
                    new_entries.append(f"🟡 {stripped.lstrip('-* ').strip()}")
                    i += 1
                else:
                    i += 1
    else:
        # Fall back to legacy <add_bullets> tag
        ab_match = re.search(r"<add_bullets>(.*?)</add_bullets>", cleaned, re.DOTALL)
        if ab_match:
            body = ab_match.group(1).strip()
            if body.lower() not in ("none", "n/a", "empty", "-", ""):
                for line in body.split("\n"):
                    line = line.strip()
                    if line.startswith("-") or line.startswith("*"):
                        new_entries.append(f"🟡 {line.lstrip('-* ').strip()}")

    return remove_ids, new_entries


# Alias kept for any external callers that may reference the old name
parse_observation_diff = parse_memory_diff


# ---------------------------------------------------------------------------
# merge_memory_into_existing — canonical merge function
# ---------------------------------------------------------------------------

def merge_memory_into_existing(
    existing_full: str,
    remove_ids: List[int],
    new_entries: List[str],
    *,
    entry_date: Optional[str] = None,
    max_entries: Optional[int] = 3000,  # increased from 1500 — Mastra has no cap
) -> str:
    """Apply remove_ids and append new memory entries under a date header.

    - Parses existing memory into raw lines (preserving date headers + blank lines)
    - Removes memory entries by 1-based index across all date groups
    - Deduplicates new_entries against remaining memory:
        * Exact normalized match → skip (already present)
        * Same text but lower emoji priority → upgrade the existing entry in-place
        * Same text but higher existing priority → skip the new one
    - Appends surviving new entries under today's date header
    - Trims oldest memory entries when over max_entries, evicting 🟢 first then 🟡
    """
    raw_lines, mem_indices = _memory_lines_numbered(existing_full)

    # Build set of raw_lines indices to remove (0-based)
    remove_set: set = set()
    for mem_num in remove_ids:
        raw_idx = mem_indices[mem_num - 1] if 1 <= mem_num <= len(mem_indices) else -1
        if raw_idx >= 0:
            remove_set.add(raw_idx)

    # Apply removals — keep date headers and blank lines
    surviving_lines: List[str] = [
        ln for idx, ln in enumerate(raw_lines) if idx not in remove_set
    ]

    # Build normalized dedup map: normalized_text → (surviving_lines_idx, emoji)
    dedup: Dict[str, Tuple[int, str]] = {}
    for i, ln in enumerate(surviving_lines):
        if _is_memory_entry(ln):
            norm = _normalize_mem(ln)
            if norm:
                dedup[norm] = (i, _get_emoji(ln))

    # Process new entries: skip duplicates, upgrade priority, collect truly new.
    # Each entry may be a multi-line string (parent + sub-bullets separated by \n).
    to_add: List[str] = []
    for entry in new_entries:
        entry = entry.strip()
        if not entry:
            continue
        # Dedup only on the first line (the parent emoji line)
        first_line = entry.split("\n")[0].strip()
        norm = _normalize_mem(first_line)
        new_emoji = _get_emoji(first_line) or "🟡"
        if norm in dedup:
            existing_idx, existing_emoji = dedup[norm]
            # Upgrade priority in-place if new priority is higher
            if _EMOJI_RANK.get(new_emoji, 0) > _EMOJI_RANK.get(existing_emoji, 0):
                text_part = _strip_emoji_prefix(surviving_lines[existing_idx])
                surviving_lines[existing_idx] = f"{new_emoji} {text_part}"
            # Otherwise skip (already present at same or higher priority)
        else:
            to_add.append(entry)
            dedup[norm] = (-1, new_emoji)

    if to_add:
        date_str = entry_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        date_header = f"Date: {date_str}"

        last_block = _last_date_block(surviving_lines)
        if last_block == date_str:
            # Today's block already exists — append each entry (expanding multi-line)
            for entry in to_add:
                for ln in entry.split("\n"):
                    surviving_lines.append(ln)
        else:
            if surviving_lines and surviving_lines[-1].strip():
                surviving_lines.append("")  # blank separator
            surviving_lines.append(date_header)
            surviving_lines.append("")
            for entry in to_add:
                for ln in entry.split("\n"):
                    surviving_lines.append(ln)

    # Trim memory if over max_entries using priority-based eviction.
    # Drop 🟢 (info) first, then 🟡 (maybe), then 🔴 (important) — within each
    # priority tier the oldest entries (earliest in the file) are evicted first.
    if max_entries is not None:
        all_mem_count = sum(1 for ln in surviving_lines if _is_memory_entry(ln))
        if all_mem_count > max_entries:
            excess = all_mem_count - max_entries
            mem_by_index = [
                (i, _EMOJI_RANK.get(_get_emoji(ln), 0))
                for i, ln in enumerate(surviving_lines)
                if _is_memory_entry(ln)
            ]
            # Sort ascending by (emoji_rank, position) — 🟢 first, oldest within tier
            mem_by_index.sort(key=lambda t: (t[1], t[0]))
            evict_set = {idx for idx, _ in mem_by_index[:excess]}
            surviving_lines = [
                ln for i, ln in enumerate(surviving_lines)
                if i not in evict_set
            ]

    result = "\n".join(surviving_lines)
    # Clean up multiple consecutive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _last_date_block(lines: List[str]) -> str:
    """Return the date string of the most recent Date: header in lines, or ''."""
    for ln in reversed(lines):
        m = _DATE_HEADER_RE.match(ln.strip())
        if m:
            return ln.strip().replace("Date:", "").strip()
    return ""


# ---------------------------------------------------------------------------
# Relative-time enrichment for memory (Mastra-inspired)
# ---------------------------------------------------------------------------

_MEMORY_DATE_HEADER_FULL_RE = re.compile(r"^(Date:\s*)(\d{4}-\d{2}-\d{2})\s*$", re.M)
_INLINE_DATE_RE = re.compile(r"\((meaning|estimated)\s+([^)]+)\)", re.I)


def _format_relative_time(target: datetime, reference: datetime) -> str:
    """Return human-readable relative time from reference to target."""
    diff_days = (reference - target).days
    if diff_days < 0:
        future_days = abs(diff_days)
        if future_days == 1:
            return "tomorrow"
        if future_days < 7:
            return f"in {future_days} days"
        if future_days < 14:
            return "in 1 week"
        if future_days < 30:
            return f"in {future_days // 7} weeks"
        if future_days < 60:
            return "in 1 month"
        if future_days < 365:
            return f"in {future_days // 30} months"
        years = future_days // 365
        return f"in {years} year{'s' if years > 1 else ''}"
    if diff_days == 0:
        return "today"
    if diff_days == 1:
        return "yesterday"
    if diff_days < 7:
        return f"{diff_days} days ago"
    if diff_days < 14:
        return "1 week ago"
    if diff_days < 30:
        return f"{diff_days // 7} weeks ago"
    if diff_days < 60:
        return "1 month ago"
    if diff_days < 365:
        return f"{diff_days // 30} months ago"
    years = diff_days // 365
    return f"{years} year{'s' if years > 1 else ''} ago"


def _format_gap_between_dates(prev: datetime, curr: datetime) -> str:
    """Return a gap marker string for the interval between two dates, or '' if trivial."""
    diff_days = (curr - prev).days
    if diff_days < 2:
        return ""
    if diff_days < 7:
        return f"[--- {diff_days} days later ---]"
    if diff_days < 14:
        return "[--- 1 week later ---]"
    if diff_days < 30:
        return f"[--- {diff_days // 7} weeks later ---]"
    if diff_days < 60:
        return "[--- 1 month later ---]"
    if diff_days < 365:
        return f"[--- {diff_days // 30} months later ---]"
    years = diff_days // 365
    return f"[--- {years} year{'s' if years > 1 else ''} later ---]"


def _try_parse_memory_date(date_str: str) -> Optional[datetime]:
    """Parse a date string from a memory entry; returns None on failure."""
    date_str = date_str.strip()
    # Strip trailing relative annotation like " - 3 months ago"
    date_str = re.split(r"\s+-\s+", date_str)[0].strip()
    # YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", date_str)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    # Month DD, YYYY or Month YYYY or just YYYY
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %Y", "%b %Y", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


_FUTURE_INTENT_RE = re.compile(
    r"\b(will|going to|planning|intends? to|looking forward|supposed to|start[s]?)\b",
    re.I,
)


def _parse_reference_date(reference_date: str) -> Optional[datetime]:
    """Parse a reference_date string.

    Accepts ISO ``YYYY-MM-DD`` plus LongMemEval's ``YYYY/MM/DD (Day) HH:MM``.
    Returns a naive datetime at midnight of the picked date, or ``None`` if
    unparseable.
    """
    if not reference_date:
        return None
    s = reference_date.strip()
    # ISO first
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    # Slash format
    m = re.match(r"^(\d{4})/(\d{2})/(\d{2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def enrich_memory_with_relative_time(memory_text: str, reference_date: str) -> str:
    """Add relative-time annotations to avatarMemory for richer temporal reasoning.

    Inspired by Mastra's ``addRelativeTimeToObservations`` / ``expandInlineEstimatedDates``.

    Transforms:
      Date: 2023-01-10
      🔴 User stated they started reading The Nightingale. (meaning 2023-01-10)

    Into:
      Date: 2023-01-10 (3 months ago)
      🔴 User stated they started reading The Nightingale. (meaning 2023-01-10 - 3 months ago)

    Gap markers are inserted between date blocks when there is a meaningful interval:
      [--- 3 weeks later ---]

    Args:
        memory_text: Raw avatarMemory markdown string.
        reference_date: ISO date "YYYY-MM-DD" used as "today" (use queryDate or current date).

    Returns:
        Enriched memory string; original returned unchanged on parse failure.
    """
    if not memory_text or not memory_text.strip():
        return memory_text
    ref = _parse_reference_date(reference_date)
    if ref is None:
        return memory_text

    # Step 1: Expand inline (meaning DATE) and (estimated DATE) with relative time.
    def _expand_inline(m: re.Match) -> str:
        prefix = m.group(1)
        date_content = m.group(2)
        # Skip already-enriched entries (already contain " - N ... ago")
        if " - " in date_content:
            return m.group(0)
        parsed = _try_parse_memory_date(date_content)
        if not parsed:
            return m.group(0)
        rel = _format_relative_time(parsed, ref)
        # Flag past-date future-intent observations as likely already completed.
        text_before = memory_text[: m.start()]
        line_start = text_before.rfind("\n") + 1
        line_before = text_before[line_start:]
        if parsed < ref and _FUTURE_INTENT_RE.search(line_before):
            return f"({prefix} {date_content} - {rel}, likely already happened)"
        return f"({prefix} {date_content} - {rel})"

    result = _INLINE_DATE_RE.sub(_expand_inline, memory_text)

    # Step 2: Annotate Date: headers with relative time and insert gap markers.
    date_matches = list(_MEMORY_DATE_HEADER_FULL_RE.finditer(result))
    if not date_matches:
        return result

    parsed_dates = [_try_parse_memory_date(m.group(2)) for m in date_matches]

    output_parts: List[str] = []
    last_index = 0
    for i, (match, parsed) in enumerate(zip(date_matches, parsed_dates)):
        output_parts.append(result[last_index : match.start()])
        # Gap marker between date blocks
        if i > 0 and parsed_dates[i - 1] and parsed:
            gap = _format_gap_between_dates(parsed_dates[i - 1], parsed)
            if gap:
                output_parts.append(f"{gap}\n\n")
        # Enriched date header
        if parsed:
            rel = _format_relative_time(parsed, ref)
            output_parts.append(f"{match.group(1)}{match.group(2)} ({rel})")
        else:
            output_parts.append(match.group(0))
        last_index = match.end()

    output_parts.append(result[last_index:])
    return "".join(output_parts)


# ---------------------------------------------------------------------------
# Aliases and backward-compatible wrappers
# ---------------------------------------------------------------------------

# Alias so any code that imported the old name still works
merge_observations_into_existing = merge_memory_into_existing


def merge_memory_diff_into_existing(
    existing_full: str,
    remove_ids: List[int],
    new_bullets: List[str],
    *,
    max_lines: Optional[int] = 200,
) -> str:
    """Backward-compatible wrapper around merge_memory_into_existing.

    Accepts legacy bullet strings and normalises them to emoji-prefixed entries.
    max_lines maps to max_entries for the underlying function.
    """
    normalized: List[str] = []
    for b in new_bullets:
        b = b.strip()
        if b and b[0] not in _PRIORITY_EMOJIS:
            b = f"🟡 {b.lstrip('-* ').strip()}"
        if b:
            normalized.append(b)

    return merge_memory_into_existing(
        existing_full,
        remove_ids,
        normalized,
        max_entries=max_lines,
    )
