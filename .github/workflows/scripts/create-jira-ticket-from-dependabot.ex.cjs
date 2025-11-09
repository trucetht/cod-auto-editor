name: Dependabot → Jira Bug

on:
  pull_request:
    types: [opened, reopened, synchronize, edited]
  workflow_dispatch: {}

permissions:
  contents: read
  pull-requests: read
  issues: write
  models: read

jobs:
  create-jira-bug:
    # Run for PRs authored by Dependabot, or when manually dispatched
    if: >
      (github.event_name == 'pull_request' && github.event.pull_request.user.login == 'dependabot[bot]')
      || (github.event_name == 'workflow_dispatch')
    runs-on: ubuntu-latest
    steps:
      - name: Debug event context
        run: |
          echo "event_name: ${{ github.event_name }}"
          echo "actor: ${{ github.actor }}"
          echo "sender.login: ${{ github.event.sender.login }}"
          echo "pr.user.login: ${{ github.event.pull_request.user.login }}"

      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install model client
        run: npm i openai@latest

      - name: Jira auth check (site mode)
        shell: bash
        env:
          JIRA_API_BASE: ${{ secrets.JIRA_API_BASE }}
          JIRA_EMAIL: ${{ secrets.JIRA_EMAIL }}
          JIRA_API_TOKEN: ${{ secrets.JIRA_API_TOKEN }}
        run: |
          set -euo pipefail
          if [[ -z "${JIRA_API_BASE:-}" || -z "${JIRA_EMAIL:-}" || -z "${JIRA_API_TOKEN:-}" ]]; then
            echo "::error::Missing JIRA_API_BASE, JIRA_EMAIL, or JIRA_API_TOKEN"
            exit 1
          fi
          host=$(node -e "console.log(new URL(process.env.JIRA_API_BASE).hostname)")
          path="/rest/api/3/myself"
          code=$(curl -sS -o resp.json -w "%{http_code}" \
            -H "Accept: application/json" \
            -u "${JIRA_EMAIL}:${JIRA_API_TOKEN}" \
            "https://${host}${path}")
          echo "HTTP ${code}"
          if [ "$code" -ne 200 ]; then
            echo "::error title=Jira auth/scope failed::See response below"
            cat resp.json
            exit 1
          fi
          echo "OK: Token + scopes valid"

      - name: Create Jira ticket from Dependabot PR (site mode)
        env:
          # Jira
          JIRA_API_BASE: ${{ secrets.JIRA_API_BASE }}
          JIRA_EMAIL: ${{ secrets.JIRA_EMAIL }}
          JIRA_API_TOKEN: ${{ secrets.JIRA_API_TOKEN }}
          JIRA_PROJECT_KEY: ${{ secrets.JIRA_PROJECT_KEY }}
          JIRA_BROWSE_BASE: ${{ secrets.JIRA_BROWSE_BASE }}

          # One of these; ID preferred
          JIRA_ISSUE_TYPE_ID: ${{ secrets.JIRA_ISSUE_TYPE_ID }}
          JIRA_ISSUE_TYPE:     ${{ secrets.JIRA_ISSUE_TYPE }}

          # Optional
          JIRA_STORY_POINTS_FIELD_ID: ${{ secrets.JIRA_STORY_POINTS_FIELD_ID }}
          JIRA_STORY_POINTS_VALUE:    ${{ secrets.JIRA_STORY_POINTS_VALUE }}

          # LLM
          USE_LLM: 'true'
          PREFERRED_MODEL: ${{ vars.PREFERRED_MODEL != '' && vars.PREFERRED_MODEL || 'openai/gpt-5-nano' }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

          # PR context
          PR_NUMBER:   ${{ github.event.pull_request.number }}
          PR_TITLE:    ${{ github.event.pull_request.title }}
          PR_BODY:     ${{ github.event.pull_request.body }}
          PR_HTML_URL: ${{ github.event.pull_request.html_url }}
          REPO:        ${{ github.repository }}

          DEBUG: 'true'
        run: node .github/workflows/scripts/create-jira-ticket-from-dependabot.ex.cjs

      - name: Comment PR with Jira link
        if: always()
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          if [[ -f "$GITHUB_OUTPUT" ]]; then
            key=$(grep '^jira_key=' "$GITHUB_OUTPUT" | cut -d= -f2- || true)
            url=$(grep '^jira_url=' "$GITHUB_OUTPUT" | cut -d= -f2- || true)
            if [[ -n "$key" && -n "$url" ]]; then
              gh pr comment "${{ github.event.pull_request.number }}" --body "Jira bug created: **${key}** – ${url}"
            fi
          fi
