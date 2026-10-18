# Design Brief Document (Web Playground & Open-Source Identity)

## 1. Aesthetic Direction
Modern, developer-centric dashboard with paired **Night** (default) and **Day** themes (inspired by Vercel & Linear). A header toggle switches modes, persists the choice, and is applied before first paint.

## 2. Color Palette
- **Background Primary:** `#09090B` (Zinc 950)
- **Background Card/Surface:** `#18181B` (Zinc 900) with `#27272A` borders
- **Primary Accent:** `#10B981` (Emerald 500 - signaling active system health)
- **Secondary Accent:** `#6366F1` (Indigo 500 - AI synthesis actions)
- **Text Primary:** `#FAFAFA` | **Text Muted:** `#A1A1AA`
- **Status Colors:** Success `#22C55E`, Warning `#F59E0B`, Error `#EF4444`

### Day Theme (Light Mode) — Muted Slate, no white glare
- **Background Primary:** `#DCE3EC` (muted slate-gray) with emerald/indigo aurora at ~40% intensity
- **Surface Card:** `rgba(241,245,249,.86–.92)` (slate-100, never pure white), `#CBD5E1` borders, soft `rgba(15,23,42,.30)` shadow
- **Inner panels / sub-cards:** `#E6ECF3`–`#E2E8F0`; **form fields** (`input`, `select`, `textarea`): `#E8EEF5` with `#CBD5E1` borders and `#1E293B` text
- **Chips & badges:** `#E2E8F0` background with `#334155` text (muted slate, not white pills)
- **Primary Accent:** `#047857` (emerald 700) for text/links, `#10B981` for controls
- **Secondary Accent:** `#4338CA` (indigo 700) for model chips and code accents
- **Text Primary:** `#0F172A` | **Text Muted:** `#52606D` | **Reader Body:** `#334155`
- The theme toggle in the header switches `html.light` / `html.dark`, persists in `localStorage` (`pd.theme`), and is applied before first paint to avoid a flash. The Markdown reader, error banner, badges, fields and toggles all have dedicated light variants.
- **Chevron rule:** the custom `<select>` arrow is drawn with `background-image` + `background-repeat: no-repeat` + `background-position` + `background-size`. Any theme rule that paints a field must use `background-color` (never the `background` shorthand), otherwise the shorthand resets the longhands and the chevron tiles across the whole control. Both `select.field` and `html.light select.field` restate all four longhands for safety.

## 3. Typography & UI Components
- **Font:** `Inter` for UI, `JetBrains Mono` for token metrics, model names, and JSON logs, `Vazirmatn` for Persian/Arabic (`dir="auto"`) content.
- **Core UI Components (Single-Page `/`):**
  1. **Model Router card:** Provider dropdown (all 8), provider API key stored **per provider** with show/hide toggle, an active-key line (`Active key: UI Override (sk-or-…4a2b)` / `Server .env (…)`) with a `× Use server .env instead` action, Custom Base URL (visible only for `custom`), **Discover Live Models** (Live Model Discovery via `fetch_available_models`, default fallback `openrouter/auto`; OpenRouter keys verified against `/auth/key`), live model search (`🔍 Search 400+ models by name or ID…`), quick-filter chips (`All`, `⚡ Free`, `GPT`, `Claude`, `Gemini`, `DeepSeek`, `Llama`), `Showing X of Y models` counter and the filtered model dropdown with the first match auto-selected.
  2. **Integrations & API Keys Vault:** Tavily key, Resend key + from-email (default `onboarding@resend.dev`) and Discord webhook URL, each with a badge reflecting server `.env` availability; **Save keys** / **Clear saved keys** persist to `localStorage`.
  3. **New Digest form:** Topic textarea (`dir="auto"`), **Output Language** select (`🌐 Auto (Same as Topic Language)` default, `Persian (فارسی)`, `English`, `Arabic (العربية)`, `German`, `French`), Recipient Email, animated Email/Discord toggle switches, and **Run autonomous digest** with an animated three-step pipeline (`Searching Web via Tavily` → `Synthesizing with AI` → `Dispatching Alerts`).
  4. **Result reader:** sanitized Markdown with `dir="auto"`, `Copy Markdown`, `Download .md`, latency, token breakdown, USD cost badge and delivery/source pills; backend `detail` errors render in a styled alert with troubleshooting tips.
  5. **Telemetry & Updates bar:** total requests, tokens, cumulative USD cost, deliveries, and **Check updates** with per-package installed → latest PyPI rows.
- **Surface treatment:** near-black `#08080B` canvas with radial emerald/indigo aurora, masked grid overlay, glassmorphic cards (`backdrop-blur`, zinc-800 borders, inset highlight), emerald primary + indigo secondary actions, pulsing system-health dot.