# Csound Programming Notes

Purpose: Keep short, general coding conventions for Csound/Cabbage work in this repo.

## Core Conventions
- Scalars: use `snake_case` (example: `k_step_changed`).
- Prefer no underscore immediately after rate prefix: use `kchanged` instead of `k_changed` when practical.
- Tables/arrays: use `camelCase` (example: `kThis_step[]`, `giProgTables`).
- Prefer clear, consistent naming over abbreviations when possible.

## GUI Update Rule (Queue Mode)
- In `guiMode("queue")`, use `cabbageSet` or `cabbageSetValue` for GUI/widget updates.
- Use trigger-based (k-rate) updates when changing properties like `visible`, `text`, `active`, or `value`.

## Common Syntax Reminder
- Use assignment when storing `changed(...)` output.
- Correct:

```csound
ktrig = changed(kval)
```

- Incorrect:

```csound
ktrig changed(kval)
```

## Channel Read Rate Rule (`chnget`)
- A control channel defined/updated at `k`-rate can still be read safely at `i`-rate with `chnget`.
- Do not assume an `i`-rate read is invalid just because the same channel is also read at `k`-rate elsewhere.
- Prefer `i`-rate reads for one-shot instrument initialization and `k`-rate reads for continuously polled logic.

## Init-Pass Branch Rule (`goto` vs `kgoto`)
- If a branch can run during init pass, do not use `goto` to jump past later `chnget.k` initialization code.
- Use `kgoto` for k-rate early exits so init pass can complete setup and avoid `chnget.k: not initialised` PERF errors.
- Typical pattern: after `kchanged changed ...`, use `if kchanged < 0.5 then kgoto done endif`.

## Instrument Stop Rule (Negative Instrument Number)
- Using a negative instrument number in an `event` call (for example `event "i", -50, ...`) only turns off running instances that were started with indefinite duration (`p3 = -1`).
- If an instrument was started with a finite duration, do not rely on negative instrument events for lifecycle control.

## Widget Default Initialization Rule
- For GUI toggles/checkboxes, prefer setting a default `value(...)` on the widget and driving behavior from `changed(...)` logic in a long-running controller instrument.
- Avoid extra startup-only branches when the same state transition can be handled by the existing change-trigger path.

## OSCsend Trigger Behavior (`kwhen`)
- `OSCsend` is trigger-driven by its `kwhen` argument.
- Do not use a static value like `1` for `kwhen` when you want send-on-change behavior.
- Prefer a trigger signal from `changed(...)`, and call `OSCsend` directly with that trigger.

Example:

```csound
khex_size_x chnget "hexgrid_size_x"
ktrig_hex_size_x changed khex_size_x
OSCsend ktrig_hex_size_x, "127.0.0.1", 9801, "/hex_size_x", "f", khex_size_x
```

Likewise for layout updates:

```csound
klayout_hex chnget "hexgrid_layout"
ktrig_hex_layout changed klayout_hex
OSCsend ktrig_hex_layout, "127.0.0.1", 9801, "/hex_layout", "f", klayout_hex
```

## Table Lookup Rate Rule
- If the table number/index source is `k`-rate (dynamic at control-rate), use `tablekt` instead of `table`.
- Use `table` only when table selection is `i`-rate/static.
- Example fix: replace `kval table kndx, k_table_id` with `kval tablekt kndx, k_table_id` when `k_table_id` can change at `k`-rate.

## Nesting Widgets Inside a Groupbox

Cabbage supports child-widget nesting inside a `groupbox` using `{ }` syntax. All widget positions inside the braces are **relative to the groupbox origin**, not the form.

```
groupbox bounds(X, Y, W, H), channel("myGroup"), text("Title"), ... {
    button bounds(Xrel, Yrel, w, h), channel("myBtn"), ...
    rslider ...
}
```

Steps to nest existing widgets:
1. Add ` {` at the end of the `groupbox` line.
2. Subtract the groupbox's X from every child widget's X.
3. Subtract the groupbox's Y from every child widget's Y.
4. Place `}` on its own line after the last child widget.

Widgets outside the braces (e.g. `csoundoutput`) keep their absolute form coordinates.

## Collapsing/Expanding a Groupbox at Runtime

Use a form-level toggle button and `cabbageSet` to resize the groupbox height. Collapsing to ~20px shows only the title bar; child widgets are clipped and effectively hidden.

### GUI
```
; Always-visible toggle above the groupbox
button bounds(5, 6, 20, 18), channel("mod_collapse"), text("▶","▼"), value(1),
    colour:0(30,30,60), colour:1(30,30,60), fontColour("white")

; Give the groupbox a channel so cabbageSet can target it
groupbox bounds(5, 27, 1250, 268), channel("modGroup"), text("Title"), ... {
    ...
}
```

- `value(1)` → starts expanded (▼ shown).
- Must use `channel("name")`, **not** `identChannel` — in `guiMode("queue")`, `cabbageSet` targets widgets by channel name.

### DSP (instr 1)
```csound
kmod_col  chnget "mod_collapse"
ktrig_col changed kmod_col
if ktrig_col == 1 then
    if kmod_col > 0.5 then
        cabbageSet ktrig_col, "modGroup", "bounds(5, 27, 1250, 268)"   ; expand
    else
        cabbageSet ktrig_col, "modGroup", "bounds(5, 27, 1250, 20)"    ; collapse
    endif
endif
```

## guiMode("queue") Widget Targeting

| Goal | Correct |
|------|---------|
| Read widget value | `chnget "channelName"` |
| Write widget value | `cabbageSetValue "channelName", kval, ktrig` |
| Change properties (bounds, colour, …) | `cabbageSet ktrig, "channelName", "property(value)"` |
| Make widget targetable by `cabbageSet` | `channel("name")` — NOT `identChannel("name")` |

---

## Shared State Rule (Prefer Channels Over Globals)
- Prefer `chnset`/`chnget` channels for scalar shared state between instruments when functionality is unchanged.
- Keep globals for arrays/tables and truly global data structures that are awkward or inefficient to move into channels.
- For event-time coordination (for example MIDI timing), write timestamps to channels in the event instrument and read them from always-on controller instruments.

Example pattern for phrase selection with MIDI silence gating:

```csound
; in MIDI input instr (note-on)
ktime_now times
chnset ktime_now, "last_midi_event_time"

; in always-on selector instr
klast_evt chnget "last_midi_event_time"
ksilence = times - klast_evt
if ksilence > 1.0 then
	; select/update phrase mode and publish via channel
	chnset knew_mode, "gen_phrase_sel"
endif

; in phrase generator instr
itraj_mode chnget "gen_phrase_sel"
```

## Lightweight Checklist
- Variable naming follows scalar/table-array convention.
- Queue-mode GUI updates use `cabbageSet`/`cabbageSetValue` with proper triggers.
- `changed(...)` calls use `=` assignment.
- `tablekt` is used when table selection is `k`-rate.
- Early k-rate exits that skip work use `kgoto` when needed to preserve init-pass setup.
- File compiles cleanly (no parser/perf errors).
