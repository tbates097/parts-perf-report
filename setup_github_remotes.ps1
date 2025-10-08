# A one-time script to set up a new local repo with both Bitbucket and GitHub remotes.

# --- YOUR GITHUB USERNAME HERE ---
$githubUsername = "tbates097" # IMPORTANT: Confirm this is your GitHub username

# Get the name of the current folder
$repo_name = (Get-Item .).Name
$currentBranch = "main" # Assumes the default branch is 'main', change if needed

Write-Host "Setting up new repository: $repo_name" -ForegroundColor Yellow

# --- STEP 1: MANUAL BITBUCKET SETUP (MANDATORY) ---
Write-Host @"
-------------------------------------------------------------------
ACTION REQUIRED: You must create the Bitbucket repository manually.
1. Go to your Bitbucket server.
2. Create a NEW, EMPTY repository named '$repo_name'.
3. DO NOT add a README or .gitignore file from the Bitbucket UI.
-------------------------------------------------------------------
"@ -ForegroundColor Cyan

$bitbucketUrl = Read-Host "https://scm2.aerotech.com/profile"

if (-not $bitbucketUrl) {
    Write-Host "Bitbucket URL cannot be empty. Aborting." -ForegroundColor Red
    exit
}

# --- STEP 2: AUTOMATED SETUP ---

# Initialize a local Git repo if it doesn't exist
if (-not (Test-Path ".git")) {
    Write-Host " -> Initializing new local Git repository..."
    git init
    git branch -M $currentBranch # Set default branch name to 'main'
    git add .
    git commit -m "Initial commit"
}

# Add the Bitbucket remote as 'origin'
Write-Host " -> Adding Bitbucket remote as 'origin'..."
git remote add origin $bitbucketUrl

# Create the GitHub repo and add it as the 'github' remote
Write-Host " -> Creating GitHub repository and adding remote 'github'..."
gh repo create $repo_name --public --source=. --remote=github

# --- STEP 3: INITIAL PUSH ---

Write-Host " -> Performing initial push to both remotes..."
# Push to Bitbucket and set it as the upstream for the branch
git push -u origin $currentBranch

# Push to GitHub
git push github $currentBranch

Write-Host "Repository '$repo_name' has been successfully set up and pushed to both remotes. ✨"