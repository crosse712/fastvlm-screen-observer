# GitHub Actions Workflows

This repository uses GitHub Actions to automatically sync changes to Hugging Face Spaces.

## Workflows

### 1. sync-to-huggingface.yml
- **Trigger**: Push to main branch or manual dispatch
- **Method**: Direct git push to Hugging Face Space
- **Speed**: Fast, simple approach

### 2. sync-hf-space.yml
- **Trigger**: Push to main, PR to main, or manual dispatch
- **Method**: Uses huggingface_hub Python library
- **Features**: More control over file uploads, ignore patterns

## Setup Instructions

### Step 1: Get Hugging Face Token
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `github-actions`
4. Role: `write`
5. Copy the token

### Step 2: Add Token to GitHub Secrets
1. Go to your GitHub repo: https://github.com/crosse712/fastvlm-screen-observer
2. Navigate to: Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `FASTVLM_API`
5. Value: Paste your Hugging Face token
6. Click "Add secret"

### Step 3: Enable Workflow
The workflows will automatically run when you:
- Push to the main branch
- Create a pull request
- Manually trigger from Actions tab

## Manual Trigger
1. Go to: https://github.com/crosse712/fastvlm-screen-observer/actions
2. Select a workflow
3. Click "Run workflow"
4. Select branch and run

## Monitoring
- Check workflow runs: https://github.com/crosse712/fastvlm-screen-observer/actions
- View Space build logs: https://huggingface.co/spaces/crosse712/fastvlm-screen-observer