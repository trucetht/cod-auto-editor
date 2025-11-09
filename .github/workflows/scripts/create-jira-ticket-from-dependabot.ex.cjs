#!/usr/bin/env node
/**
 * Dependabot PR → Jira Bug (Site or EX mode)
 * - Parses Dependabot PRs (single & grouped) to extract upgrades
 * - Calls GitHub Models to draft Jira BUG content (forces/repairs JSON)
 * - Searches Jira for duplicates (new /search/jql API), else creates an issue
 */

const https = require('https');
const fs = require('fs');

const {
  DEBUG,

  // EX gateway (optional)
  JIRA_EX_BASE,
  JIRA_CLOUD_ID,

  // Site mode (preferred)
  JIRA_API_BASE,

  // Force mode: "ex" or "site"
  JIRA_MODE,

  // Jira auth/fields
  JIRA_EMAIL,
  JIRA_API_TOKEN,
  JIRA_PROJECT_KEY,
  JIRA_ISSUE_TYPE,
  JIRA_DEFAULT_LABELS = '',
  JIRA_STORY_POINTS_FIELD_ID = '',
  JIRA_STORY_POINTS_VALUE = '',
  JIRA_BROWSE_BASE,

  // PR context
  PR_NUMBER,
  PR_TITLE,
  PR_BODY = '',
  PR_HTML_URL,
  REPO,

  // Models
  GH_MODELS_TOKEN,
  GITHUB_TOKEN,

  USE_LLM = 'true',
  PREFERRED_MODEL: RAW_PREFERRED_MODEL
} = process.env;

const PREFERRED_MODEL = (RAW_PREFERRED_MODEL && RAW_PREFERRED_MODEL.trim()) || 'openai/gpt-5-nano';
const MODELS_TOKEN = GH_MODELS_TOKEN || GITHUB_TOKEN;

const REQ_TIMEOUT = 20000;
const b64 = (s) => Buffer.from(s, 'utf8').toString('base64');

function httpJson({ method, host, path, headers = {} }, body) {
  return new Promise((resolve, reject) => {
    const req = https.request({ method, hostname: host, path, headers }, (res) => {
      let data = '';
      res.on('data', (c) => (data += c));
      res.on('end', () => {
        let parsed = {};
        try { parsed = data ? JSON.parse(data) : {}; } catch { parsed = { raw: data }; }
        const ok = res.statusCode >= 200 && res.statusCode < 300;
        if (ok) return resolve({ status: res.statusCode, data: parsed });
        const err = new Error(`HTTP ${res.statusCode}`);
        err.status = res.statusCode;
        err.body = parsed;
        return reject(err);
      });
    });
    req.on('error', (e) => {
      const err = new Error(e.message || 'request_error');
      err.code = e.code;
      return reject(err);
    });
    req.setTimeout(REQ_TIMEOUT, () => req.destroy(new Error('request_timeout')));
    if (body) req.write(typeof body === 'string' ? body : JSON.stringify(body));
    req.end();
  });
}

function pickMode() {
  const mode = (JIRA_MODE || '').toLowerCase();
  if (mode === 'site') return 'site';
  if (mode === 'ex') return 'ex';
  if (JIRA_EX_BASE && JIRA_CLOUD_ID) return 'ex';
  if (JIRA_API_BASE) return 'site';
  throw new Error('Missing Jira config: (JIRA_EX_BASE + JIRA_CLOUD_ID) or JIRA_API_BASE.');
}

function jiraHostAndPathBuilder() {
  const mode = pickMode();
  if (mode === 'ex') {
    const host = new URL(JIRA_EX_BASE).hostname;
    return {
      mode,
      host,
      path: (p) => `/ex/jira/${encodeURIComponent(JIRA_CLOUD_ID)}/rest/api/3${p.startsWith('/') ? p : `/${p}`}`
    };
  } else {
    const host = new URL(JIRA_API_BASE).hostname;
    return {
      mode,
      host,
      path: (p) => `/rest/api/3${p.startsWith('/') ? p : `/${p}`}`
    };
  }
}

/** Parse Dependabot PRs (title + body; handles grouped PR “Updates `pkg` from X to Y” table) */
function extractPackages(prTitle, prBody) {
  const pkgs = new Set();
  const upgrades = [];
  const ver = '([0-9A-Za-z.+-]+)';

  // "Bump foo from X to Y"
  const t = prTitle?.match(new RegExp(`Bump\\s+([@\\w\\/.-]+)\\s+from\\s+${ver}\\s+to\\s+${ver}`, 'i'));
  if (t) { pkgs.add(t[1]); upgrades.push({ name: t[1], from: t[2], to: t[3] }); }

  if (prBody) {
    // bullet format in body
    const reBullets = new RegExp(String.raw`(?<=\* )([@\w\/.-]+)\s+from\s+${ver}\s+to\s+${ver}`, 'gi');
    for (const m of prBody.matchAll(reBullets)) {
      pkgs.add(m[1]); upgrades.push({ name: m[1], from: m[2], to: m[3] });
    }
    // grouped “Updates `pkg` from X to Y”
    const reUpdates = new RegExp(String.raw`Updates\s+\`([@\w\/.-]+)\`\s+from\s+${ver}\s+to\s+${ver}`, 'gi');
    for (const m of prBody.matchAll(reUpdates)) {
      pkgs.add(m[1]); upgrades.push({ name: m[1], from: m[2], to: m[3] });
    }
  }
  return { packages: Array.from(pkgs), upgrades };
}

function sanitizeLabel(s) {
  return String(s).toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9_.-]/g, '-')
    .replace(/-+/g, '-')
    .slice(0, 200);
}

/** Try to force JSON; if the model still returns prose, coerce to JSON safely */
async function generateWithGitHubModels(ctx) {
  if (!MODELS_TOKEN) {
    const e = new Error('missing_token');
    e.code = 'missing_token';
    throw e;
  }

  const baseHeaders = {
    'User-Agent': 'gh-models-jira-bot',
    'Authorization': `Bearer ${MODELS_TOKEN}`,
    'Accept': 'application/json',
    'X-GitHub-Api-Version': '2022-11-28',
    'Content-Type': 'application/json'
  };

  const commonMsg = [
    {
      role: "system",
      content: "You are a release/QA engineer. Return strictly JSON with the requested schema. No prose."
    },
    {
      role: "user",
      content: [
        { type: "text", text:
`Repo: ${ctx.repo}
PR: ${ctx.prUrl}
Title: ${ctx.prTitle}

Parsed upgrades:
${ctx.upgrades.map(u => `- ${u.name}: ${u.from} → ${u.to}`).join('\n')}

PR body (truncated):
${(ctx.prBody || '').slice(0, 3000)}` },
        { type: "text", text:
`Return STRICT JSON:
{
  "title": string,
  "description": string,
  "acceptance_criteria": [string, ...],
  "definition_of_done": [string, ...],
  "labels": [string, ...]
}` }
      ]
    }
  ];

  // Attempt 1: ask for JSON output formally
  let res;
  try {
    res = await httpJson(
      { method: 'POST', host: 'models.github.ai', path: '/inference/chat/completions', headers: baseHeaders },
      {
        model: PREFERRED_MODEL,
        messages: commonMsg,
        // "nano" doesn’t accept temperature override; omit it
        // Try to hint JSON mode – if the model ignores, we’ll still coerce
        response_format: { type: "json_object" }
      }
    );
  } catch (err) {
    // If /catalog/models helps debugging, print it
    try {
      const cat = await httpJson(
        { method: 'GET', host: 'models.github.ai', path: '/catalog/models', headers: baseHeaders }
      );
      console.error('Catalog models visible to token:', JSON.stringify(cat.data, null, 2));
    } catch {}
    throw err;
  }

  const content = res?.data?.choices?.[0]?.message?.content;
  if (!content) {
    const e = new Error('no_content_from_models_api');
    e.status = res?.status;
    e.body = res?.data;
    throw e;
  }

  // Try strict JSON parse first
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0]);
      return normalizeModelJson(parsed);
    } catch {
      // fall through to coercion
    }
  }

  // Coerce free-text to JSON (robust against the sample you posted)
  const coerced = coerceFreeTextToJson(content, ctx);
  return normalizeModelJson(coerced);
}

function normalizeModelJson(obj) {
  if (typeof obj !== 'object' || obj === null) throw new Error('Model response is not an object');
  const out = {
    title: String(obj.title || '').trim(),
    description: String(obj.description || '').trim(),
    acceptance_criteria: Array.isArray(obj.acceptance_criteria) ? obj.acceptance_criteria.filter(s => typeof s === 'string' && s.trim()) : [],
    definition_of_done: Array.isArray(obj.definition_of_done) ? obj.definition_of_done.filter(s => typeof s === 'string' && s.trim()) : [],
    labels: Array.isArray(obj.labels) ? obj.labels.filter(s => typeof s === 'string' && s.trim()) : []
  };
  if (!out.title) out.title = 'Dependency upgrades QA validation';
  if (!out.title.toLowerCase().startsWith('[dependabot')) {
    out.title = `[Dependabot] ${out.title}`;
  }
  return out;
}

/** Turn the model’s prose (like in your log) into our JSON shape */
function coerceFreeTextToJson(text, ctx) {
  const lines = String(text).split(/\r?\n/);
  const result = { title: '', description: '', acceptance_criteria: [], definition_of_done: [], labels: [] };

  let section = 'desc';
  for (const raw of lines) {
    const line = raw.trim();

    if (/^summary:/i.test(line)) {
      result.title = line.replace(/^summary:\s*/i, '').trim();
      continue;
    }
    if (/^acceptance criteria:/i.test(line)) { section = 'ac'; continue; }
    if (/^definition of done:/i.test(line)) { section = 'dod'; continue; }
    if (/^labels?:/i.test(line)) {
      const lbls = line.replace(/^labels?:\s*/i, '').split(/[,\s]+/).map(s => s.trim()).filter(Boolean);
      result.labels.push(...lbls);
      continue;
    }
    if (/^references:$/i.test(line)) { section = 'desc'; continue; }

    if (section === 'ac' && line) {
      const item = line.replace(/^[-*]\s*/, '').trim();
      if (item) result.acceptance_criteria.push(item);
      continue;
    }
    if (section === 'dod' && line) {
      const item = line.replace(/^[-*]\s*/, '').trim();
      if (item) result.definition_of_done.push(item);
      continue;
    }
    // Default: accumulate into description
    if (line) result.description += (result.description ? '\n' : '') + raw;
  }

  if (!result.title) {
    // Rough title from upgrades or PR title
    const { upgrades } = extractPackages(ctx.prTitle, ctx.prBody);
    const top = upgrades.map(u => u.name).slice(0, 3).join(', ') || ctx.prTitle || 'Dependency upgrades';
    result.title = `Regression risk after dependency upgrades (${top}${upgrades.length > 3 ? ', …' : ''})`;
  }
  return result;
}

function jiraClient() {
  const { host, path } = jiraHostAndPathBuilder();
  const auth = `Basic ${b64(`${JIRA_EMAIL}:${JIRA_API_TOKEN}`)}`;
  const base = { host, headers: { 'Authorization': auth, 'Accept': 'application/json', 'Content-Type': 'application/json' } };
  return {
    async get(p)  { return httpJson({ method: 'GET',  ...base, path: path(p) }); },
    async post(p, body) { return httpJson({ method: 'POST', ...base, path: path(p) }, body); }
  };
}

/** Use new /search/jql; if 400 due to shape mismatch, retry with 'query' instead of 'jql' */
async function jiraSearchByText(jira, text) {
  if (!text || !text.trim()) return null;
  const safe = text.replace(/["\\]/g, '\\$&');
  const jql = `project = ${JIRA_PROJECT_KEY} AND text ~ "${safe}" ORDER BY created DESC`;

  // First try: correct shape (use 'jql')
  let body = { queries: [ { jql, startAt: 0, maxResults: 1, fields: ['key'] } ] };
  try {
    const res = await jira.post('/search/jql', body);
    const issue = res.data?.results?.[0]?.issues?.[0];
    return issue ?? null;
  } catch (e) {
    if (e.status === 400) {
      // Fallback attempt with 'query' (some tenants/sandboxes briefly expected this)
      try {
        body = { queries: [ { query: jql, startAt: 0, maxResults: 1, fields: ['key'] } ] };
        const res2 = await jira.post('/search/jql', body);
        const issue2 = res2.data?.results?.[0]?.issues?.[0];
        return issue2 ?? null;
      } catch (e2) {
        throw e2; // bubble up 400 so caller can decide
      }
    }
    throw e;
  }
}

async function jiraCreateIssue(jira, fields) {
  const res = await jira.post('/issue', { fields });
  return res.data;
}

function setOutput(name, value) {
  const outFile = process.env.GITHUB_OUTPUT;
  if (!outFile) {
    console.error('GITHUB_OUTPUT not set; cannot export step outputs.');
    return;
  }
  fs.appendFileSync(outFile, `${name}=${value}\n`, { encoding: 'utf8' });
}

function buildBrowseUrl(issueKey) {
  const base = (JIRA_BROWSE_BASE || '').replace(/\/$/, '');
  if (!base) return `https://atlassian.net/browse/${issueKey}`;
  return `${base}/browse/${issueKey}`;
}

(async () => {
  try {
    // Required env
    for (const [k, v] of Object.entries({ JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY })) {
      if (!v) throw new Error(`Missing required env: ${k}`);
    }
    if (USE_LLM === 'true' && !MODELS_TOKEN) {
      throw new Error('Missing required env: GH_MODELS_TOKEN or GITHUB_TOKEN');
    }

    pickMode();
    const jira = jiraClient();

    // Verify auth
    try {
      const me = await jira.get('/myself');
      if (DEBUG === 'true') console.log('Authenticated Jira user:', me.data && (me.data.displayName || me.data.name || 'ok'));
    } catch (e) {
      console.error('Jira auth/scope check failed.');
      if (DEBUG === 'true') {
        console.error('Status:', e.status);
        console.error('Response:', JSON.stringify(e.body, null, 2));
      }
      throw e;
    }

    const { packages, upgrades } = extractPackages(PR_TITLE, PR_BODY);

    // De-dup by PR URL using new search API
    let existing = null;
    if (PR_HTML_URL && PR_HTML_URL.trim()) {
      try { existing = await jiraSearchByText(jira, PR_HTML_URL); }
      catch (e) {
        console.warn('Jira search failed; continuing to create.', DEBUG === 'true' ? e : '');
      }
    }
    if (existing) {
      const url = buildBrowseUrl(existing.key);
      setOutput('jira_key', existing.key);
      setOutput('jira_url', url);
      console.log(`Found existing Jira issue ${existing.key} -> ${url}`);
      return;
    }

    // Generate content (never fail the whole job because of “non-JSON” replies)
    let gen;
    if (USE_LLM === 'true') {
      try {
        gen = await generateWithGitHubModels({
          repo: REPO,
          prUrl: PR_HTML_URL || '',
          prTitle: PR_TITLE || '',
          prBody: PR_BODY || '',
          upgrades
        });
      } catch (e) {
        console.error(`LLM request failed for preferred model: ${PREFERRED_MODEL}`);
        if (DEBUG === 'true') {
          console.error('Status:', e.status || '');
          if (e.code) console.error('Error code:', e.code);
          console.error('Response body:', JSON.stringify(e.body || e, null, 2));
        }
        throw new Error('Aborting ticket creation because the LLM could not be used with the preferred model.');
      }
    } else {
      throw new Error('Aborting ticket creation because USE_LLM=false is not supported for this workflow.');
    }

    const defaults = (JIRA_DEFAULT_LABELS || '').split(',').map(s => s.trim()).filter(Boolean);
    const parsedPkgLabels = packages.map(sanitizeLabel);
    const modelLabels = Array.isArray(gen.labels) ? gen.labels.map(sanitizeLabel) : [];
    const labels = Array.from(new Set(['dependabot', ...defaults, ...parsedPkgLabels, ...modelLabels]));

    const ac = (gen.acceptance_criteria || []).map(i => `- ${i}`).join('\n');
    const dod = (gen.definition_of_done || []).map(i => `- ${i}`).join('\n');
    const extras = `\n\n*Acceptance Criteria*\n${ac}\n\n*Definition of Done*\n${dod}\nPR: ${PR_HTML_URL}\n`;

    const fields = {
      project: { key: JIRA_PROJECT_KEY },
      issuetype: { name: JIRA_ISSUE_TYPE || 'Bug' },
      summary: gen.title || `[Dependabot] ${PR_TITLE}`,
      description: {
        type: 'doc',
        version: 1,
        content: [
          { type: 'paragraph', content: [{ type: 'text', text: (gen.description || '').slice(0, 60000) }] },
          { type: 'paragraph', content: [{ type: 'text', text: extras.slice(0, 60000) }] }
        ]
      },
      labels
    };

    if (JIRA_STORY_POINTS_FIELD_ID && JIRA_STORY_POINTS_VALUE !== '') {
      const sp = Number(JIRA_STORY_POINTS_VALUE);
      if (Number.isFinite(sp)) fields[JIRA_STORY_POINTS_FIELD_ID] = sp;
      else console.warn(`Ignored non-numeric Story Points: "${JIRA_STORY_POINTS_VALUE}"`);
    }

    const created = await jiraCreateIssue(jira, fields);
    const key = created.key;
    const browseUrl = buildBrowseUrl(key);
    setOutput('jira_key', key);
    setOutput('jira_url', browseUrl);
    console.log(`Created Jira issue ${key} -> ${browseUrl}`);
  } catch (err) {
    console.error('Failed to create Jira issue.');
    if (DEBUG === 'true') {
      if (err && (err.body || err.status || err.code)) {
        if (err.status) console.error('Status:', err.status);
        if (err.code) console.error('Error code:', err.code);
        console.error('Response body:', JSON.stringify(err.body || err, null, 2));
      } else {
        console.error('Error details:', err);
      }
    }
    process.exit(1);
  }
})();
