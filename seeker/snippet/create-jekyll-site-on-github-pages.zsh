#date: 2025-09-09T16:54:04Z
#url: https://api.github.com/gists/1f74cf6b9fd3428ffcc2632918b6ae11
#owner: https://api.github.com/users/maulomelin

#!/bin/zsh
# SPDX-FileCopyrightText: (c) 2025 Mauricio Lomelin <maulomelin@gmail.com>
# SPDX-License-Identifier: MIT
# SPDX-FileComment: This script creates a Jekyll site hosted on GitHub Pages.
# It is written as a reference for copying commands into a terminal, reviewing
# the output, and fixing any issues. It is not intended to be run directly, as
# it lacks the error handling or logging required for unsupervised execution.
# Written for Zsh on macOS, it assumes a preconfigured environment.

# Script configuration.
readonly _USERNAME=$(gh api user -q ".login")
readonly _REPO_NAME="${_USERNAME}.github.io"
readonly _CONTACT_INFO="Mauricio Lomelin &lt;maulomelin@gmail.com&gt;"
readonly _DEV_ROOT="${HOME}/_dev"
readonly _WORKSPACE_DIR="${_DEV_ROOT}/repos/github.com/${_USERNAME}"
readonly _PROJECT_DIR="${_WORKSPACE_DIR}/${_REPO_NAME}"
readonly _TMP_DIR="_tmp"
readonly _JEKYLL_GITIGNORE="https://raw.githubusercontent.com/github/gitignore/HEAD/Jekyll.gitignore"
readonly _MACOS_GITIGNORE="https://raw.githubusercontent.com/github/gitignore/HEAD/Global/macOS.gitignore"
readonly _THEME_REPO="https://github.com/pages-themes/primer/"
readonly _THEME_REPO_DTSLUG="${${${_THEME_REPO:t3}// /}//\//--}--$(date +%Y%m%dT%H%M%S)"

# Verify environment.
if [ -z "${ZSH_NAME}" ]; then   # Fail fast if not running in zsh.
    echo "This is a pseudo-zsh script. Adjust for other shells."
    exit 1
fi

# Install Homebrew to install Jekyll and Bundler.
if ( ! brew -v ); then          # Install Homebrew.
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    brew update                 # Update Homebrew.
    brew upgrade                # Install the newest versions of all packages.
fi

# Install Jekyll and Bundler to run local Jekyll builds.
if ( jekyll -v ); then          # Update Jekyll and Bundler.
    gem update jekyll           # Update Jekyll gem to the latest version.
    gem update bundler          # Update Bundler gem to the latest version.
else                            # Install Jekyll and Bundler.
    brew install chruby         # Install Ruby version manager.
    brew install ruby-install   # Install Ruby installer.
    ruby-install ruby 3.4.1     # Install Ruby for Jekyll.
    # Update the zsh configuration file to use chruby by default.
    echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.zshrc
    echo "source $(brew --prefix)/opt/chruby/share/chruby/auto.sh" >> ~/.zshrc
    echo "chruby ruby-3.4.1" >> ~/.zshrc
    source ~/.zshrc             # Apply updates (alternate: `exec zsh`).
    gem install jekyll          # Install the latest Jekyll gem.
    gem install bundler         # Install the latest Bundler gem.
fi

# Check versions for compatibility.
ruby -v                         # Check version: gem "ruby", "3.4.1"
jekyll -v                       # Check version: gem "jekyll", ">=4.4.1"
bundler -v                      # Check version: gem "bundler", ">=2.6.9"

# Run Homebrew diagnostics to check for issues.
brew doctor                     # Run Homebrew diagnostics, as an FYI.

# Move to the workspace directory to create a repo.
mkdir -p "${_WORKSPACE_DIR}"    # Create intermediate directories, if needed.
cd "${_WORKSPACE_DIR}"          # Move to the directory.

# Create a repo for the GitHub Pages site and clone it locally.
gh repo create ${_REPO_NAME} --public --gitignore Jekyll --clone

# Move to the project directory.
cd "${_PROJECT_DIR}"

# Update the `.gitignore` file with custom + Jekyll + macOS entries.
(   echo "### Source: Custom\n";                 echo "${_TMP_DIR}/"; \
    echo "\n### Source: ${_JEKYLL_GITIGNORE}\n"; curl -fsSL ${_JEKYLL_GITIGNORE}; \
    echo "\n### Source: ${_MACOS_GITIGNORE}\n";  curl -fsSL ${_MACOS_GITIGNORE} \
) > GITIGNORE
mv GITIGNORE .gitignore

# Confirm the remote GitHub fetch/push URLs.
git remote -v

# Create a `README.md` file with copyright and licensing terms.
cat <<CONTENT > README.md
# About

This is my personal blog - a collection of notes, tutorials, and code snippets.

It's built with Jekyll and hosted on GitHub Pages - see posts for more details.

## Copyright and Licensing

The content in the \`_posts/\` and \`images/\` directories of this project is the copyright of ${_CONTACT_INFO}. Do not use without permission.

All other files and content are licensed under the MIT License.
CONTENT

# Create a `Gemfile` file to match local builds with GitHub Pages builds.
cat <<CONTENT > Gemfile
source "https://rubygems.org"
gem "github-pages", group: :jekyll_plugins    
CONTENT

# Create a minimal `_config.yml` file to configure Jekyll for GitHub Pages.
cat <<CONTENT > _config.yml
title: Personal Documents
description: A collection of notes, tutorials, and code snippets.
repository: ${_USERNAME}/${_REPO_NAME}
CONTENT

# Add the Jekyll theme "Primer" to `_config.yml` to give a minimal UX to the site.
cat <<CONTENT >> _config.yml
theme: jekyll-theme-primer
CONTENT

# Export the Jekyll theme "Primer" to a local folder, to use as reference for template files.
git clone --depth=1 "${_THEME_REPO}" "${_TMP_DIR}/${_THEME_REPO_DTSLUG}"
rm -rf ./"${_TMP_DIR}/${_THEME_REPO_DTSLUG}"/.git

# Create a `_posts/` directory for all blog posts.
mkdir "_posts"

# Create an initial stub post.
cat <<CONTENT > _posts/1993-09-26--my-first-post.md
---
layout: post
title: My 1st Post
---
This is my first post.
CONTENT

# Create an `index.html` file to serve as homepage and list all posts.
cat <<CONTENT > index.html
---
layout: default
---
<ul>
    {% for post in site.posts %}
    <li>
        [{{ post.date | date: "%Y-%m-%d" }}]
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
</ul>
CONTENT

# Install the project gems and run a local server to build and preview the site.
bundle install                  # Install all project gems.
bundle exec jekyll serve        # Build and serve the site locally. 

# Stage all changes, commit them locally, and push them to GitHub.
git add .                                   # Stage all files.
git commit -m "Initial commit from local"   # Commit changes to local repo.
git push origin                             # Push changes to remote repo.

# Configure GitHub Pages for this repo:
echo "Configure GitHub Pages for this repo:"
echo "  1. Go to the GitHub Pages Settings for this repo:"
echo "     https://github.com/${_USERNAME}/${_REPO_NAME}/settings/pages"
echo "  2. Select [ Source = \"Deploy from a branch\" ]."
echo "  3. Select [ Branch > Select branch = \"main\" ]."
echo "  4. Select [ Branch > Select folder = \"/ (root)\" ]."
echo "  5. Click on \"Save\"."
open "https://github.com/${_USERNAME}/${_REPO_NAME}/settings/pages"

# View the published site.
if [[ "${_REPO_NAME}" == "${_USERNAME}.github.io" ]]; then
    open "https://${_USERNAME}.github.io/"                  # User site.
else
    open "https://${_USERNAME}.github.io/${_REPO_NAME}/"    # Project site.
fi
