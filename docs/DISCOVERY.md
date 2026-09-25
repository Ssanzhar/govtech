# DISCOVERY.md — Analyst / operator interviews (PLAN_2026-09 C7)

_Purpose: 3–5 structured conversations with the people who would actually use Level 2 or
feed it — a bank fraud desk, a hotline operator, an Antifraud-centre or police cyber-unit
analyst — so that the analyst backlog (C6 feedback loop, queue, exports) is shaped by
what they do, not by what we imagine. Outcome: anonymised notes below + **ADR D16**
recording what changed in the backlog. Interviews are not a gate for the partner API or
the privacy architecture (those are unconditional)._

## Ground rules

- 45 minutes, one interviewer, one note-taker. Recorded only with explicit consent; notes
  are anonymised (role + organisation type, never names) before they are committed.
- We do **not** demo first. The demo comes in the last 10 minutes, after their story.
- We ask about their work, not about our product. "Would you use X?" questions are banned.
- No call content is collected during an interview. If they show us a case, we take notes
  on the *shape* (channel, tactic, timeline), never numbers or text.

## Script

**1. Their day (10 min)**
1. Walk me through the last suspected-scam case you handled, start to finish.
2. Where did it come from (customer call, internal alert, another institution, police)?
3. What did you have to look at before deciding anything? What was missing?
4. How many of these per day / per week? What share turn out to be real?

**2. Linking and prioritising (10 min)**
5. How do you decide which case to work first?
6. When do you conclude that two cases are the same operation? What evidence counts
   (number reuse, script, timing, money flow)?
7. How often does a genuinely new scheme appear, and how do you first notice it?
8. Who else needs to know when you find one, and how do you tell them today?

**3. Data and constraints (10 min)**
9. What can you legally share outside your organisation — and in what form (hashed number,
   tactic label, transcript)? Who signs that off?
10. What would make you refuse a tool: where it runs, what it stores, who can see what?
11. Do you keep outcome labels (confirmed fraud / not)? Could a sample be shared for
    evaluation under the intake protocol (`docs/DATA_INTAKE.md`)?
12. Kazakh-language calls: how are they handled today? Who transcribes or translates?

**4. Reaction to the concept (10 min, demo)**
Show: the citizen page on a phone (on-device), then the analyst queue and one drill-down.
13. What here would you act on tomorrow? What would you ignore?
14. When the model says "novel scheme", what would you need to believe it?
15. If you could confirm / dismiss / merge organisations, would that change what you do
    the next day? (This is C6 — listen for whether feedback is a habit or a chore.)

**5. Close (5 min)**
16. Who else should we talk to? May we come back with a pilot proposal (PLAN §7 item 7)?

## Notes template (commit one file per interview under `docs/discovery/`)

```
# Interview <n> — <role>, <organisation type>, <date>
Consent: recorded / notes only
Volume & mix: …
How they link cases today: …
How they prioritise: …
How new schemes surface: …
Sharing constraints (legal, form, sign-off): …
Reaction to the queue / drill-down / novelty flag: …
Feedback loop (C6): would use / would not, why: …
Data offer (A3): yes / maybe / no — what, how many, under which basis: …
Quotes (anonymised, ≤ 3): …
Backlog implications: …
```

## What we are listening for (backlog hooks)

| Signal in the interview | Backlog item it shapes |
|---|---|
| "Same operation" = number reuse | keep the number graph primary (C8 finding); C9 signals-only placement |
| "Same operation" = script/timing | text overlay or timeline features; revisit the C8 text-overlay row |
| New schemes noticed via volume spikes | C10 novelty = support + growth, not distance alone |
| Confirm/dismiss is natural | C6 feedback loop as designed (dismissed decay in `rank.py`) |
| Confirm/dismiss is a chore | C6 becomes implicit feedback (opened cases, exports) |
| Legal can share hashed numbers + tactic labels only | partner API stays signals-first (D19); transcripts optional forever |
| Kazakh handled by translation | on-device ASR (B7) matters more than expected; KK eval rows first |

## ADR D16 rule

After the third interview, write `DECISIONS.md` **D16 — Discovery outcome**: what we
heard (3–5 bullets), what changed in the backlog (items added, reprioritised, dropped),
and what did not change and why. Interviews four and five append to it.
