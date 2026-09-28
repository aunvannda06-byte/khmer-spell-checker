// netlify/functions/check-khmer.js
// Secure proxy between the Khmer Spell Checker frontend and the Gemini API.
// The API key is read from the Netlify environment variable GEMINI_API_KEY
// and is NEVER sent to the browser.

const MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;

// Netlify's default sync-function limit is 10s, so we abort a little earlier
// and return a clean JSON error instead of a platform-level 502.
const GEMINI_TIMEOUT_MS = 8500;

// Input limits (keeps requests small and cheap; blocks abuse)
const MAX_ITEMS = 25;
const MAX_WORD_LEN = 60;
const MAX_CONTEXT_LEN = 240;
const MAX_BODY_BYTES = 20000;
const MAX_SUGGESTIONS = 5;
const MAX_RESULTS = 40;
const ALLOWED_TYPES = ['spelling', 'grammar', 'context', 'proper_noun', 'valid'];

const HEADERS = {
  'Content-Type': 'application/json; charset=utf-8',
  'Cache-Control': 'no-store',
};

const json = (statusCode, body) => ({ statusCode, headers: HEADERS, body: JSON.stringify(body) });

const SYSTEM_PROMPT = `You are a careful Khmer spelling and grammar assistant.

The user message is JSON: {"items":[{"word":"...","context":"..."}]}.
Each item is a whole word that a dictionary check did NOT fully recognise, with the sentence it appears in.

Rules:
1. Return exactly one result for every item, with "word" copied exactly as given.
2. "suggestions": up to ${MAX_SUGGESTIONS} correct Khmer replacements for the WHOLE word (each one replaces the entire word), best first. Use [] if none.
3. "type" must be one of:
   - "spelling": misspelled word
   - "grammar": grammatical error
   - "context": a real word but wrong for this context (e.g. homophone / wrong word choice)
   - "proper_noun": a name, place, foreign or technical term that is probably fine
   - "valid": the word is actually correct
4. "reason": one short sentence in Khmer explaining the issue.
5. If a context sentence contains ANOTHER clearly wrong word that is not in items, add a result for it. Its "word" must be one single token copied exactly from that context.
6. Never add results for correct words. Never invent words. Keep meaning unchanged.`;

const RESPONSE_SCHEMA = {
  type: 'OBJECT',
  properties: {
    results: {
      type: 'ARRAY',
      items: {
        type: 'OBJECT',
        properties: {
          word: { type: 'STRING' },
          suggestions: { type: 'ARRAY', items: { type: 'STRING' } },
          type: { type: 'STRING', enum: ALLOWED_TYPES },
          reason: { type: 'STRING' },
        },
        required: ['word', 'suggestions', 'type', 'reason'],
      },
    },
  },
  required: ['results'],
};

const clean = (v, max) => (typeof v === 'string' ? v.normalize('NFC').trim().slice(0, max) : '');

exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return json(405, { error: 'method_not_allowed' });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error('GEMINI_API_KEY is not set');
    return json(500, { error: 'server_not_configured' });
  }

  // ── Validate input ──────────────────────────────────────────
  if (event.body && event.body.length > MAX_BODY_BYTES) {
    return json(413, { error: 'payload_too_large' });
  }
  let payload;
  try {
    payload = JSON.parse(event.body || '{}');
  } catch (_) {
    return json(400, { error: 'invalid_json' });
  }

  const seen = new Set();
  const items = (Array.isArray(payload.items) ? payload.items : [])
    .map((it) => ({ word: clean(it && it.word, MAX_WORD_LEN), context: clean(it && it.context, MAX_CONTEXT_LEN) }))
    .filter((it) => it.word && !seen.has(it.word) && seen.add(it.word))
    .slice(0, MAX_ITEMS);

  if (items.length === 0) {
    return json(200, { results: [] });
  }

  // ── Call Gemini (with timeout) ──────────────────────────────
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GEMINI_TIMEOUT_MS);

  try {
    const res = await fetch(GEMINI_URL, {
      method: 'POST',
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': apiKey, // header, not URL, so it never lands in logs
      },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents: [{ role: 'user', parts: [{ text: JSON.stringify({ items }) }] }],
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 2048,
          responseMimeType: 'application/json',
          responseSchema: RESPONSE_SCHEMA,
        },
      }),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      console.error('Gemini HTTP', res.status, detail.slice(0, 300));
      return json(res.status === 429 ? 429 : 502, { error: res.status === 429 ? 'rate_limited' : 'upstream_error' });
    }

    const data = await res.json();
    const text =
      data && data.candidates && data.candidates[0] && data.candidates[0].content &&
      data.candidates[0].content.parts
        ? data.candidates[0].content.parts.map((p) => p.text || '').join('')
        : '';

    let parsed;
    try {
      parsed = JSON.parse(text.replace(/^```json\s*|```$/g, '').trim());
    } catch (_) {
      console.error('Gemini returned non-JSON output');
      return json(502, { error: 'bad_upstream_response' });
    }

    // ── Sanitize model output ─────────────────────────────────
    const sentWords = new Set(items.map((i) => i.word));
    const allContext = items.map((i) => i.context).join('\n');

    const results = (Array.isArray(parsed && parsed.results) ? parsed.results : [])
      .map((r) => ({
        word: clean(r && r.word, MAX_WORD_LEN),
        suggestions: (Array.isArray(r && r.suggestions) ? r.suggestions : [])
          .map((s) => clean(s, MAX_WORD_LEN))
          .filter(Boolean)
          .slice(0, MAX_SUGGESTIONS),
        type: ALLOWED_TYPES.includes(r && r.type) ? r.type : 'spelling',
        reason: clean(r && r.reason, 300),
      }))
      // keep only words we sent, or extra words that really appear in the supplied context
      .filter((r) => r.word && (sentWords.has(r.word) || allContext.includes(r.word)))
      .slice(0, MAX_RESULTS);

    return json(200, { results });
  } catch (err) {
    if (err && err.name === 'AbortError') {
      return json(504, { error: 'timeout' });
    }
    console.error('check-khmer error:', err && err.message);
    return json(502, { error: 'upstream_error' });
  } finally {
    clearTimeout(timer);
  }
};
