#!/usr/bin/env node
/**
 * Dependabot PR → Jira Bug (SITE mode example)
 * - Parses packages from PR
 * - Calls GitHub Models for JSON (aborts if model fails or not JSON)
 * - Uses new Jira search endpoint (/search/jql) with GET query params
 * - Resolves issue type ID via createmeta if name given
 * - Creates issue with ADF description
 */

const https = require('https');
const fs = require('fs');

const env = process.env;

// Required env
const {
  JIRA_API_BASE,
  JIRA_EMAIL,
  JIRA_API_TOKEN,
  JIRA_PROJECT_KEY,
  JIRA_BROWSE_BASE,

  // Issue type: prefer ID. If missing, we resolve ID from name.
  JIRA_ISSUE_TYPE_ID,
  JIRA_ISSUE_TYPE,

  // Optional SPs
  JIRA_STORY_POINTS_FIELD_ID = '',
  JIRA_STORY_POINTS_VALUE = '',

  // PR info
  PR_NUMBER,
  PR_TITLE,
  PR_BODY = '',
  PR_HTML_URL,
  REPO,

  // LLM
  USE_LLM = 'true',
  GITHUB_TOKEN,

  DEBUG = 'false'
} = env;

// Prefer explicit model env only if non-empty; otherwise default to nano.
const PREFERRED_MODEL = (env.PREFERRED_MODEL && env.PREFERRED_MODEL.trim())
  ? env.PREFERRED_MODEL.trim()
  : 'openai/gpt-5-nano';

const MODELS_TOKEN = GITHUB_TOKEN;           // 'models:read' permission required
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
    req.setTimeout(REQUEST_TIMEOUT_MS, () => req.destroy(new Error('request_timeout')));
    if (body) req.write(typeof body === 'string' ? body : JSON.stringify(body));
    req.end();
  });
}

function jiraClient() {
  const host = new URL(JIRA_API_BASE).hostname;
  const auth = `Basic ${b64(`${JIRA_EMAIL}:${JIRA_API_TOKEN}`)}`;
  const basePath = '/rest/api/3';

  return {
    async get(pWithQs) {
      return httpJson(
        { method: 'GET', host, path: `${basePath}${pWithQs}`, headers: { 'Authorization': auth, 'Accept': 'application/json' } }
      );
    },
    async post(p, body) {
      return httpJson(
        { method: 'POST', host, path: `${basePath}${p}`, headers: { 'Authorization': auth, 'Accept': 'application/json', 'Content-Type': 'application/json' } },
        body
      );
    }
  };
}

function extractPackages(prTitle, prBody) {
  const pkgs = new Set();
  const upgrades = [];
  const ver = '([0-9A-Za-z.+-]+)';
  const t = prTitle?.match(new RegExp(`Bump\\s+([@\\w\\/.-]+)\\s+from\\s+${ver}\\s+to\\s+${ver}`, 'i'));
  if (t) { pkgs.add(t[1]); upgrades.push({ name: t[1], from: t[2], to: t[3] }); }
  if (prBody) {
    const re = new RegExp(String.raw`(?<=\|\s)([@\w\/.-]+)\`\s*\|\s*\`(${ver})\`\s*\|\s*\`(${ver})\``, 'gi');
    for (const m of prBody.matchAll(re)) {
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

/**
 * New Jira search endpoint (GET) to avoid old /search payload 400s.
 * https://your-site/rest/api/3/search/jql?jql=...&maxResults=1&fields=key
 */
async function jiraSearchByText(jira, text) {
  if (!text || !text.trim()) return null;
  const safe = text.replace(/["\\]/g, '\\$&');
  const jql = `project = ${JIRA_PROJECT_KEY} AND text ~ "${safe}" ORDER BY created DESC`;
  const qs = `?jql=${encodeURIComponent(jql)}&maxResults=1&fields=key`;
  const res = await jira.get(`/search/jql${qs}`);
  const issue = res.data?.issues?.[0];
  return issue ?? null;
}

/** Resolve issue type id from ID or Name via createmeta */
async function resolveIssueTypeId(jira) {
  // If ID provided, trust it
  if (JIRA_ISSUE_TYPE_ID && String(JIRA_ISSUE_TYPE_ID).trim()) {
    return String(JIRA_ISSUE_TYPE_ID).trim();
  }

  // Otherwise, fetch and match by name (case-insensitive)
  const qs = `?projectKeys=${encodeURIComponent(JIRA_PROJECT_KEY)}&expand=projects.issuetypes.fields`;
  const meta = await jira.get(`/issue/createmeta${qs}`);
  const its = meta.data?.projects?.[0]?.issuetypes || [];
  if (!its.length) throw new Error(`No createmeta issuetypes available for project ${JIRA_PROJECT_KEY}`);

  if (JIRA_ISSUE_TYPE && JIRA_ISSUE_TYPE.trim()) {
    const want = JIRA_ISSUE_TYPE.trim().toLowerCase();
    const hit = its.find(it => (it.name || '').toLowerCase() === want);
    if (hit?.id) return hit.id;
    if (DEBUG === 'true') {
      console.warn(`Issue type name "${JIRA_ISSUE_TYPE}" not valid for ${JIRA_PROJECT_KEY}. Available: ${its.map(i => `${i.name}(${i.id})`).join(', ')}`);
    }
  }

  // Fallback: first available type
  return its[0].id;
}

function setOutput(name, value) {
  const outFile = process.env.GITHUB_OUTPUT;
  if (!outFile) return;
  fs.appendFileSync(outFile, `${name}=${value}\n`, { encoding: 'utf8' });
}

function buildBrowseUrl(issueKey) {
  const base = (JIRA_BROWSE_BASE || '').replace(/\/$/, '');
  if (!base) return `https://atlassian.net/browse/${issueKey}`;
  return `${base}/browse/${issueKey}`;
}

async function generateWithGitHubModels(ctx) {
  if (!MODELS_TOKEN) {
    const e = new Error('missing_token');
    e.code = 'missing_token';
    throw e;
  }

  const body = {
    model: PREFERRED_MODEL,
    // Some small models ignore JSON mode; hammer it via instructions
    messages: [
      { role: "system", content: "You are a release/QA engineer. ONLY return a single JSON object, no prose, no code fences." },
      {
        role: "user",
        content:
`Repo: ${ctx.repo}
PR: ${ctx.prUrl}
Title: ${ctx.prTitle}

Parsed upgrades:
${ctx.upgrades.map(u => `- ${u.name}: ${u.from} → ${u.to}`).join('\n')}

Return EXACT JSON with these keys:
{"title": string, "description": string, "acceptance_criteria": [string], "definition_of_done": [string], "labels": [string]}
NO extra keys. NO additional text.`
      }
    ]
    // DO NOT send temperature; some models only support default
  };

  const res = await httpJson(
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

  const content = res?.data?.choices?.[0]?.message?.content;
  if (!content || typeof content !== 'string') {
    const e = new Error('no_content_from_models_api');
    e.status = res?.status;
    e.body = res?.data;
    throw e;
  }

  // Extract first JSON object
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

  // Basic schema
  if (typeof parsed.title !== 'string' || !parsed.title.trim()) throw new Error('Missing/invalid key: title');
  if (typeof parsed.description !== 'string' || !parsed.description.trim()) throw new Error('Missing/invalid key: description');

  for (const k of ['acceptance_criteria', 'definition_of_done', 'labels']) {
    if (!Array.isArray(parsed[k])) parsed[k] = [];
    parsed[k] = parsed[k].filter((s) => typeof s === 'string' && s.trim());
  }
  if (!parsed.title.toLowerCase().startsWith('[dependabot')) {
    parsed.title = `[Dependabot] ${parsed.title.trim()}`;
  }
  return parsed;
}

(async () => {
  try {
    for (const [k, v] of Object.entries({ JIRA_API_BASE, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY })) {
      if (!v) throw new Error(`Missing required env: ${k}`);
    }
    if (USE_LLM === 'true' && !MODELS_TOKEN) {
      throw new Error('Missing required env: GITHUB_TOKEN (with models:read)');
    }

    const jira = jiraClient();

    // Verify Jira auth
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

    // Look for an existing issue referencing this PR
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

    // Build content with LLM (abort if fails — per your requirement)
    if (USE_LLM !== 'true') throw new Error('USE_LLM must be true for this workflow.');
    let gen;
    try {
      const { packages, upgrades } = extractPackages(PR_TITLE, PR_BODY);
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

    // Resolve issue type id
    let issueTypeId;
    try {
      issueTypeId = await resolveIssueTypeId(jira);
    } catch (e) {
      console.error('Failed to resolve issue type id.');
      if (DEBUG === 'true') console.error(e);
      throw e;
    }

    // Labels
    const defaults = [];
    const parsedPkgLabels = (extractPackages(PR_TITLE, PR_BODY).packages || []).map(sanitizeLabel);
    const modelLabels = Array.isArray(gen.labels) ? gen.labels.map(sanitizeLabel) : [];
    const labels = Array.from(new Set(['dependabot', ...defaults, ...parsedPkgLabels, ...modelLabels]));

    const ac = (gen.acceptance_criteria || []).map(i => `- ${i}`).join('\n') || '- ';
    const dod = (gen.definition_of_done || []).map(i => `- ${i}`).join('\n') || '- ';
    const extras = `\n\n*Acceptance Criteria*\n${ac}\n\n*Definition of Done*\n${dod}\nPR: ${PR_HTML_URL}\n`;

    const fields = {
      project:   { key: JIRA_PROJECT_KEY },
      issuetype: { id: issueTypeId }, // IMPORTANT: send ID, not name
      summary:   gen.title || `[Dependabot] ${PR_TITLE}`,
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
      } else if (DEBUG === 'true') {
        console.warn(`Ignored non-numeric Story Points: "${JIRA_STORY_POINTS_VALUE}"`);
      }
    }

    const created = await jira.post('/issue', { fields });
    const key = created.data?.key || created.key;
    const browseUrl = buildBrowseUrl(key);

    setOutput('jira_key', key);
    setOutput('jira_url', browseUrl);

    console.log(`Created Jira issue ${key} -> ${browseUrl}`);
  } catch (err) {
    console.error('Failed to create Jira issue.');
    if (DEBUG === 'true') {
      if (err && (err.body || err.status || err.code)) {
        if (err.status) console.error('Status:', err.status);
        if (err.code)   console.error('Error code:', err.code);
        console.error('Response body:', JSON.stringify(err.body || err, null, 2));
      } else {
        console.error('Error details:', err);
      }
    }
    process.exit(1);
  }
})();
