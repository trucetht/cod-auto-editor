#!/usr/bin/env node
/**
 * Dependabot PR → Jira Bug
 * - Extracts packages from PR title/body
 * - Uses GitHub Models to draft Title/Description/AC/DoD
 * - Searches Jira for existing issue mentioning the PR URL
 * - Creates Jira issue with labels and optional Story Points
 *
 * Notes:
 * - Uses scoped token against ex gateway for API calls
 * - Uses JIRA_BROWSE_BASE for URL
 * - Sets step outputs via $GITHUB_OUTPUT
 */

const https = require('https');
const fs = require('fs');

const {
  DEBUG,

  // EX gateway
  JIRA_EX_BASE,     // e.g. https://api.atlassian.com/ex/jira
  JIRA_CLOUD_ID,

  // Site API 
  JIRA_API_BASE,

  // optional explicit mode override: "ex" or "site"
  JIRA_MODE,

  // Common
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

  // GitHub Models
  GITHUB_TOKEN,

  USE_LLM = 'true',
  PREFERRED_MODEL = 'openai/gpt-5-nano'
} = process.env;

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
    req.on('error', reject);
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

function extractPackages(prTitle, prBody) {
  const pkgs = new Set();
  const upgrades = [];

  const ver = '([0-9A-Za-z.+-]+)';
  const t = prTitle?.match(new RegExp(`Bump\\s+([@\\w\\/.-]+)\\s+from\\s+${ver}\\s+to\\s+${ver}`, 'i'));
  if (t) { pkgs.add(t[1]); upgrades.push({ name: t[1], from: t[2], to: t[3] }); }

  if (prBody) {
    const re = new RegExp(String.raw`(?<=\* )([@\w\/.-]+)\s+from\s+${ver}\s+to\s+${ver}`, 'gi');
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

async function generateWithGitHubModels(ctx) {
  const body = {
    model: PREFERRED_MODEL,
    messages: [
      {
        role: "system",
        content:
          "You are a release/QA engineer. Produce crisp Jira BUG ticket content from dependency upgrade PRs."
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
    ],
    temperature: 0.2
  };

  const res = await httpJson(
    {
      method: 'POST',
      host: 'models.github.ai',
      path: '/inference/chat/completions',
      headers: {
        'User-Agent': 'gh-models-jira-bot',
        'Authorization': `Bearer ${GITHUB_TOKEN}`,
        'Accept': 'application/json',
        'X-GitHub-Api-Version': '2022-11-28',
        'Content-Type': 'application/json'
      }
    },
    body
  );

  const content = res.data?.choices?.[0]?.message?.content;
  if (!content) throw new Error('No content from models API');

  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) throw new Error('No JSON object found in model response');
  let parsed;
  try { parsed = JSON.parse(jsonMatch[0]); }
  catch { throw new Error('Malformed JSON in model response'); }

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

// New search via POST /search/jql
async function jiraSearchByText(jira, text) {
  if (!text || !text.trim()) return null;
  const safe = text.replace(/["\\]/g, '\\$&');
  const jql = `project = ${JIRA_PROJECT_KEY} AND text ~ "${safe}" ORDER BY created DESC`;
  const body = {
    queries: [
      { query: jql, startAt: 0, maxResults: 1, fields: ["key"] }
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

function fallbackTicket(ctx) {
  const title = `[Dependabot] ${ctx.prTitle || 'Dependency update'}`.slice(0, 255);
  const lines = [];
  if (ctx.upgrades && ctx.upgrades.length) {
    lines.push('Upgrades:');
    for (const u of ctx.upgrades) lines.push(`- ${u.name}: ${u.from} → ${u.to}`);
  }
  const desc = [
    `Automated ticket generated without LLM due to model access.`,
    lines.join('\n'),
    `PR: ${ctx.prUrl}`
  ].filter(Boolean).join('\n\n');
  return {
    title,
    description: desc,
    acceptance_criteria: [],
    definition_of_done: [],
    labels: []
  };
}

(async () => {
  try {
    for (const [k, v] of Object.entries({ JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY, GITHUB_TOKEN })) {
      if (!v) throw new Error(`Missing required env: ${k}`);
    }
    pickMode();

    const jira = jiraClient();

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

    let gen;
    if (USE_LLM === 'true') {
      if (!PREFERRED_MODEL) {
        console.error('Missing PREFERRED_MODEL.');
        process.exit(1);
      }
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
        if (e && (e.body || e.status)) {
          if (e.status) console.error('Status:', e.status);
          console.error('Response body:', JSON.stringify(e.body || {}, null, 2));
        } else {
          console.error('Error details:', e);
        }
        console.error('Aborting ticket creation because the LLM could not be used with the preferred model.');
        process.exit(1);
      }
    } else {
      gen = fallbackTicket({
        repo: REPO,
        prUrl: PR_HTML_URL || '',
        prTitle: PR_TITLE || '',
        prBody: PR_BODY || '',
        upgrades
      });
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
      if (err && (err.body || err.status)) {
        console.error('Status:', err.status);
        console.error('Response body:', JSON.stringify(err.body, null, 2));
      } else {
        console.error('Error details:', err);
      }
    }
    process.exit(1);
  }
})();
