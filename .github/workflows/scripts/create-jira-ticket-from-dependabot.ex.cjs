#!/usr/bin/env node
/**
 * Dependabot PR → Jira Bug (Site mode or EX mode)
 * - Parses Dependabot PRs (single & grouped) to extract upgrades
 * - Calls GitHub Models via models.github.ai to draft Jira BUG content
 * - Searches Jira for duplicates (new /search/jql API), else creates a new issue
 * - Fails the job if the LLM cannot be used with the preferred model
 */

const https = require('https');
const fs = require('fs');

const {
  DEBUG,

  // EX gateway (optional alternative to site)
  JIRA_EX_BASE,           // e.g. https://api.atlassian.com/ex/jira
  JIRA_CLOUD_ID,

  // Site API (preferred in your setup)
  JIRA_API_BASE,

  // optional explicit mode override: "ex" or "site"
  JIRA_MODE,

  // Common Jira
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

// Hard default even if the env var comes in as an empty string
const PREFERRED_MODEL = (RAW_PREFERRED_MODEL && RAW_PREFERRED_MODEL.trim()) || 'openai/gpt-5-nano';
const MODELS_TOKEN = GH_MODELS_TOKEN || GITHUB_TOKEN;

const REQUEST_TIMEOUT_MS = 20000;
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
    req.setTimeout(REQUEST_TIMEOUT_MS, () => {
      req.destroy(new Error('request_timeout'));
    });
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
  throw new Error('Missing Jira config: provide (JIRA_EX_BASE + JIRA_CLOUD_ID) for ex mode or JIRA_API_BASE for site mode.');
}

function jiraHostAndPathBuilder() {
  const mode = pickMode();
  if (mode === 'ex') {
    const host = new URL(JIRA_EX_BASE).hostname;
    return {
      mode,
      host,
      path: (p) => {
        const clean = p.startsWith('/') ? p : `/${p}`;
        return `/ex/jira/${encodeURIComponent(JIRA_CLOUD_ID)}/rest/api/3${clean}`;
      }
    };
  } else {
    const host = new URL(JIRA_API_BASE).hostname;
    return {
      mode,
      host,
      path: (p) => {
        const clean = p.startsWith('/') ? p : `/${p}`;
        return `/rest/api/3${clean}`;
      }
    };
  }
}

/** Extract upgrades from Dependabot PRs (single AND grouped formats) */
function extractPackages(prTitle, prBody) {
  const pkgs = new Set();
  const upgrades = [];
  const ver = '([0-9A-Za-z.+-]+)';

  // Single bump in title: "Bump foo from X to Y"
  const t = prTitle?.match(new RegExp(`Bump\\s+([@\\w\\/.-]+)\\s+from\\s+${ver}\\s+to\\s+${ver}`, 'i'));
  if (t) { pkgs.add(t[1]); upgrades.push({ name: t[1], from: t[2], to: t[3] }); }

  if (prBody) {
    // Old bullet format: "* foo from X to Y"
    const reBullets = new RegExp(String.raw`(?<=\* )([@\w\/.-]+)\s+from\s+${ver}\s+to\s+${ver}`, 'gi');
    for (const m of prBody.matchAll(reBullets)) {
      pkgs.add(m[1]);
      upgrades.push({ name: m[1], from: m[2], to: m[3] });
    }

    // Grouped PR lines: "Updates `pkg` from X to Y"
    const reUpdates = new RegExp(String.raw`Updates\s+\`([@\w\/.-]+)\`\s+from\s+${ver}\s+to\s+${ver}`, 'gi');
    for (const m of prBody.matchAll(reUpdates)) {
      pkgs.add(m[1]);
      upgrades.push({ name: m[1], from: m[2], to: m[3] });
    }
  }

  return { packages: Array.from(pkgs), upgrades };
}

function sanitizeLabel(s) {
  return String(s)
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9_.-]/g, '-')
    .replace(/-+/g, '-')
    .slice(0, 200);
}

/** Call GitHub Models; if it fails, throw with rich diagnostics */
async function generateWithGitHubModels(ctx) {
  if (!MODELS_TOKEN) {
    const e = new Error('missing_token');
    e.code = 'missing_token';
    throw e;
  }

  const body = {
    model: PREFERRED_MODEL,
    messages: [
      {
        role: "system",
        content: "You are a release/QA engineer. Produce crisp Jira BUG ticket content from dependency upgrade PRs."
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text:
`Repo: ${ctx.repo}
PR: ${ctx.prUrl}
Title: ${ctx.prTitle}

Parsed upgrades:
${ctx.upgrades.map(u => `- ${u.name}: ${u.from} → ${u.to}`).join('\n')}

PR body (truncated):
${(ctx.prBody || '').slice(0, 3000)}`
          },
          {
            type: "text",
            text:
`Return STRICT JSON:
{
  "title": string,
  "description": string,
  "acceptance_criteria": [string, ...],
  "definition_of_done": [string, ...],
  "labels": [string, ...]
}`
          }
        ]
      }
    ]
    // Note: do NOT set temperature for nano; some models only accept default.
  };

  let res;
  try {
    res = await httpJson(
      {
        method: 'POST',
        host: 'models.github.ai',
        path: '/inference/chat/completions',
        headers: {
          'User-Agent': 'gh-models-jira-bot',
          'Authorization': `Bearer ${MODELS_TOKEN}`,
          'Accept': 'application/json',
          'X-GitHub-Api-Version': '2022-11-28',
          'Content-Type': 'application/json'
        }
      },
      body
    );
  } catch (err) {
    // Enrich failure with catalog (what models are visible?)
    try {
      const cat = await httpJson(
        {
          method: 'GET',
          host: 'models.github.ai',
          path: '/catalog/models',
          headers: {
            'User-Agent': 'gh-models-jira-bot',
            'Authorization': `Bearer ${MODELS_TOKEN}`,
            'Accept': 'application/json'
          }
        }
      );
      console.error('Catalog models visible to token:', JSON.stringify(cat.data, null, 2));
    } catch (catErr) {
      if (DEBUG === 'true') {
        console.error('Failed to retrieve /catalog/models');
        console.error('Status:', catErr.status || '');
        console.error('Response body:', JSON.stringify(catErr.body || catErr, null, 2));
      }
    }
    throw err;
  }

  const content = res?.data?.choices?.[0]?.message?.content;
  if (!content) {
    const e = new Error('no_content_from_models_api');
    e.status = res?.status;
    e.body = res?.data;
    throw e;
  }

  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    const e = new Error('no_json_object_found_in_model_response');
    e.status = res?.status;
    e.body = { raw: content };
    throw e;
  }

  let parsed;
  try { parsed = JSON.parse(jsonMatch[0]); }
  catch {
    const e = new Error('malformed_json_in_model_response');
    e.status = res?.status;
    e.body = { raw: content };
    throw e;
  }

  validateModelResponseSchema(parsed);
  return parsed;
}

function validateModelResponseSchema(obj) {
  if (typeof obj !== 'object' || obj === null) throw new Error('Model response is not an object');
  for (const k of ['title', 'description']) {
    if (typeof obj[k] !== 'string' || !obj[k].trim()) {
      throw new Error(`Missing/invalid key: ${k}`);
    }
  }
  for (const k of ['acceptance_criteria', 'definition_of_done', 'labels']) {
    if (!Array.isArray(obj[k])) obj[k] = [];
    obj[k] = obj[k].filter((x) => typeof x === 'string' && x.trim());
  }
  if (!obj.title.trim().toLowerCase().startsWith('[dependabot')) {
    obj.title = `[Dependabot] ${obj.title.trim()}`;
  }
}

function jiraClient() {
  const { mode, host, path } = jiraHostAndPathBuilder();
  const auth = `Basic ${b64(`${JIRA_EMAIL}:${JIRA_API_TOKEN}`)}`;

  return {
    mode,
    async get(p) {
      return httpJson(
        { method: 'GET', host, path: path(p), headers: { 'Authorization': auth, 'Accept': 'application/json' } }
      );
    },
    async post(p, body) {
      return httpJson(
        { method: 'POST', host, path: path(p), headers: { 'Authorization': auth, 'Accept': 'application/json', 'Content-Type': 'application/json' } },
        body
      );
    }
  };
}

// ✅ Use the new multi-query endpoint to avoid 410
async function jiraSearchByText(jira, text) {
  if (!text || !text.trim()) return null;
  const safe = text.replace(/["\\]/g, '\\$&');
  const jql = `project = ${JIRA_PROJECT_KEY} AND text ~ "${safe}" ORDER BY created DESC`;
  const body = {
    queries: [
      { query: jql, startAt: 0, maxResults: 1, fields: ['key'] }
    ]
  };
  const res = await jira.post('/search/jql', body);
  const first = res.data?.results?.[0];
  const issue = first?.issues?.[0];
  return issue ?? null;
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

    // Quick auth check (already done in workflow, but nice here too)
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

    // De-dupe by PR URL using new /search/jql
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

    // Generate with LLM (hard fail if we cannot use it)
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
    const extras =
      `\n\n*Acceptance Criteria*\n${ac}\n\n*Definition of Done*\n${dod}\nPR: ${PR_HTML_URL}\n`;

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
      if (Number.isFinite(sp)) {
        fields[JIRA_STORY_POINTS_FIELD_ID] = sp;
      } else {
        console.warn(`Ignored non-numeric Story Points: "${JIRA_STORY_POINTS_VALUE}"`);
      }
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
