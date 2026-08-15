#!/bin/sh
# Make the data directory writable by the app, then drop privileges.
#
# WHY THIS EXISTS. The image chowns /data at build time, but a platform that
# mounts a volume over /data replaces that directory — and its ownership — at
# runtime. Railway bind-mounts it root-owned, so an app running as a non-root
# user gets "unable to open database file" on a perfectly healthy deploy.
# Docker *named* volumes hide this, because they inherit ownership from the
# image, which is why it only shows up once deployed.
#
# Fixing it needs root, but running the app as root does not: so chown here and
# hand off to the unprivileged user. Platforms that already start the container
# as a non-root user (Hugging Face Spaces uses uid 1000) fall through to the
# plain exec — the mount is theirs to own and there is nothing to fix.
set -e

if [ "$(id -u)" = "0" ]; then
    mkdir -p /data
    chown -R scholar:scholar /data
    exec gosu scholar "$@"
fi

exec "$@"
