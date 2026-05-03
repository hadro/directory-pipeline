---
name: uxreview
description: |
  Conducts a structured expert UX/HCI review of a data-browser explorer HTML file.
  Trigger when the user asks for a "UX review", "HCI review", "usability review",
  "interface feedback", or "how can I improve the explorer". The skill guides a
  multi-pass review combining source-code analysis with user-supplied screenshots
  or observations, and produces prioritised, actionable recommendations.
---

When this skill is invoked, follow these steps. Steps 1–3 are setup; Step 4 is the
core review loop.

---

## Step 1: Identify the target

Ask the user which explorer to review if not already clear:
- Path to the HTML file (e.g. `output/.../green_book_entries_explorer.html`)
- Or a screenshot they want to analyse

If a file path is given, use the Read tool to load the source.

---

## Step 2: Gather context (ask once, then proceed)

Ask the user for a brief answer to each of these — a sentence or two is enough:

1. **Audience**: Who will use this explorer? (Researchers? General public? Internal team?)
2. **Primary task**: What's the one thing most users come here to do?
3. **Known pain points**: Anything you already know is awkward or broken?
4. **Scope**: Full review, or focus on a specific area (filters, detail panel, mobile, accessibility)?

If the user says "just do it" or gives no answers, assume: researchers and
general public, primary task = find a specific listing by name/location/year,
no known pain points, full review.

---

## Step 3: Source-code analysis (always run this)

Read the HTML file and analyse it against the checklist below. You are looking for
**structural and behavioural issues** — things that are hard to see in a screenshot
but are clear in code.

### Checklist A — Information architecture
- [ ] Is the page title / heading unambiguous about what this tool does?
- [ ] Is the number of results visible at all times?
- [ ] Is the current filter state always visible (what's active, how to clear it)?
- [ ] Is there a clear empty state (message when zero results match)?
- [ ] Are column headers and filter labels written in plain language, not schema names?

### Checklist B — Interaction model
- [ ] Can filters be combined? Can they be cleared individually and all-at-once?
- [ ] Does search give feedback on what it searches (all fields? name only?)?
- [ ] Is the detail panel open/close gesture obvious (click row? click button?)?
- [ ] Is URL state preserved so users can bookmark / share a view?
- [ ] Is there a way to get back to "start" without reloading the page?

### Checklist C — Progressive disclosure
- [ ] Is the initial view clean (no overwhelming number of controls)?
- [ ] Are advanced or rare controls secondary (collapsed, smaller, lower)?
- [ ] Does opening a detail panel not disrupt the list scroll position?
- [ ] Are image thumbnails only loaded on demand (not all upfront)?

### Checklist D — Keyboard and accessibility (WCAG 2.1 AA)

**Perceivable**
- [ ] **1.4.3 Contrast (Minimum)** — Normal text ≥ 4.5:1, large text (≥ 18 pt / 14 pt bold) ≥ 3:1.
  Check body text, small labels (meta chips, facet counts, year chips), placeholder text,
  and any text on coloured backgrounds (active chip labels on #1a6ebd, etc.).
- [ ] **1.4.11 Non-text Contrast** — UI components (button borders, input borders, focus rings,
  chart elements) have ≥ 3:1 contrast against their background.
- [ ] **1.4.4 Resize Text** — Page is usable and readable when browser text is scaled to 200%.
- [ ] **1.4.10 Reflow** — Content reflows without horizontal scrolling at 320 CSS px width
  (equivalent to 400% zoom on a 1280 px screen).
- [ ] **1.3.1 Info and Relationships** — Structure conveyed visually (headings, lists, tables)
  is also conveyed programmatically (correct HTML elements or ARIA roles).
- [ ] **1.3.3 Sensory Characteristics** — No instruction relies solely on shape, colour,
  size, or position ("click the blue chip", "the button on the right").
- [ ] **1.4.1 Use of Color** — Colour is never the *only* visual means of conveying
  information (e.g. active-chip state must have a non-colour indicator too).

**Operable**
- [ ] **2.1.1 Keyboard** — All functionality (filter, search, open detail, close detail,
  about modal, export) is operable via keyboard alone.
- [ ] **2.1.2 No Keyboard Trap** — Focus can always move away from any component (modal
  traps focus *within* it while open, but Escape releases it).
- [ ] **2.4.3 Focus Order** — Tab order follows a logical reading sequence.
- [ ] **2.4.7 Focus Visible** — Every focusable element has a clearly visible focus indicator.
  Check that `:focus-visible` styles are not overridden to `outline: none` without a replacement.
- [ ] **2.4.4 Link Purpose** — Link and button text is meaningful out of context
  ("Export CSV" ✓, "click here" ✗, "Item page ↗" is borderline — needs surrounding context).
- [ ] **2.5.3 Label in Name** — Where a control has a visible text label, the accessible
  name contains that text.

**Understandable**
- [ ] **3.1.1 Language of Page** — `<html lang="...">` is set.
- [ ] **3.2.2 On Input** — Changing a filter or search field does not trigger unexpected
  context changes (e.g. page navigation or a modal opening).
- [ ] **3.3.1 Error Identification** — If input can be invalid (search syntax, URL params),
  errors are identified in text.

**Robust**
- [ ] **4.1.2 Name, Role, Value** — All custom interactive components (chips, detail panel,
  modal, facet checkboxes) expose name, role, and state to assistive technology via
  correct HTML semantics or ARIA attributes.
- [ ] **4.1.3 Status Messages** — Status updates (result count changes, filter applied)
  are programmatically determinable without focus moving to them (use `aria-live` regions).

**How to check contrast without a browser plugin:**
For any hex colour pair, ask: "What is the WCAG 2.1 contrast ratio between #XXXXXX
and #YYYYYY?" — Claude can compute this directly from the colour values in the CSS.

### Checklist E — Performance and resilience
- [ ] Is the dataset size embedded in JS (may cause slow parse on mobile)?
- [ ] Does the table re-render the full DOM on every filter change?
- [ ] Is there a loading or progress indicator for any async work?
- [ ] Does the page function with JavaScript disabled (graceful degradation)?

### Checklist F — Mobile and small screens
- [ ] Does the layout reflow usably at ≤ 640 px?
- [ ] Are touch targets at least 44 × 44 px?
- [ ] Is horizontal scrolling avoided on small screens?
- [ ] Does the detail panel occupy the full screen (not overlap with a still-visible table)?

Report findings as:
- **Critical** — breaks core task, likely affects most users
- **High** — significantly degrades experience, affects many users
- **Medium** — noticeable friction, worth fixing when time allows
- **Low / Nice-to-have** — polish, affects power users or edge cases

---

## Step 4: Screenshot analysis (run if the user provides images)

If the user pastes one or more screenshots, analyse each against the same checklist
categories **plus** these visual / perceptual criteria that code analysis cannot cover:

- **Visual hierarchy**: Is the most important information (entry name, year, city) the
  most visually prominent? Or is secondary info competing for attention?
- **Proximity and grouping**: Do related controls cluster together? Does white space
  separate distinct regions?
- **Scannability**: Can a user skim the results list and find what they want in
  under 5 seconds?
- **Affordances**: Do buttons look like buttons? Do clickable rows signal they are
  clickable (hover cursor, hover state)?
- **Feedback**: Is it clear which year chips / facets are currently active?
- **Density**: Is the information density appropriate for the audience, or does it
  feel cramped / sparse?
- **First impressions**: What does a new user see first? Is that the right thing?

Ask the user to provide these screenshots if they haven't already:
1. Initial page load (no filters applied)
2. A filtered state with several facets active
3. A detail panel open alongside the results
4. The page at mobile width (≤ 480 px) if relevant

---

## Step 5: Prioritised recommendations

Produce a ranked recommendation list. Structure each item as:

```
### [Priority] Short title
**Problem**: One sentence describing the issue and why it matters.
**Evidence**: Where you saw it (code line, screenshot region, or heuristic violated).
**Recommendation**: Concrete, specific suggestion — not "improve X" but "change Y to Z".
**Effort**: S / M / L (roughly: < 1 hr / half-day / multi-day)
```

Group by priority tier (Critical → High → Medium → Low).

At the end, include a **"Quick wins"** section: the three highest-impact, lowest-effort
items a developer could ship in an hour.

---

## Step 6: Long-term directions (optional, ask first)

If the user wants strategic suggestions beyond the current interface, discuss:

- **Faceted search evolution**: As the dataset grows, do the current filter patterns
  scale? (Facet value counts, search-within-facet, hierarchical filters.)
- **Progressive enhancement**: IIIF deep-links, Content State links to a viewer,
  map integration — which add the most value for the audience?
- **Saved searches / permalinks**: URL hash state is a good start; is there a case
  for named saved searches or sharing presets?
- **Data quality surface**: Should the UI surface data confidence or completeness
  (e.g. "OCR confidence low", "no bounding box available")?
- **Comparison and export**: Can users select multiple entries to compare or export
  a filtered subset?
- **Guided onboarding**: For a public-facing tool, is there a case for a brief
  interactive tour or example queries?

---

## Notes on getting the best review

- **More screenshots = better feedback.** Code analysis finds structural issues;
  screenshots reveal visual hierarchy, density, and affordance problems that are
  invisible in source.
- **Re-run after major changes.** A 15-minute re-review after each significant
  iteration catches regressions and validates fixes.
- **Invite a real user.** Even one 10-minute session with someone unfamiliar with
  the tool will surface issues no code review can find.
