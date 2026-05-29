#!/bin/bash

# Manual fix when git operations fail with VS Code socket permission errors:
# bash .devcontainer/fix_vscode_git_socket_acl_until_done.sh
# Check logs:
# tail -f /tmp/fix_vscode_git_socket_acl_until_done.log

set -u
set -o pipefail

LOG_FILE="/tmp/fix_vscode_git_socket_acl_until_done.log"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

apply_acl_once() {
    shopt -s nullglob

    local remote_socks=(/tmp/vscode-remote-containers-ipc-*.sock)
    local git_socks=(/tmp/vscode-git-*.sock)

    local found_remote=0
    local found_git=0
    local failed_any=0
    local sock

    # Remote container IPC sockets
    for sock in "${remote_socks[@]}"; do
        found_remote=1
        if setfacl --modify "user:${USERNAME}:rw" "$sock" 2>/dev/null; then
            log "Remote container IPC socket permission setup successful: $(basename "$sock")"
        else
            failed_any=1
            log "Failed to set permission on remote container IPC socket: $(basename "$sock")"
        fi
    done

    # Git operation sockets
    for sock in "${git_socks[@]}"; do
        found_git=1
        if setfacl --modify "user:${USERNAME}:rw" "$sock" 2>/dev/null; then
            log "Git ops socket permission setup successful: $(basename "$sock")"
        else
            failed_any=1
            log "Failed to set permission on git ops socket: $(basename "$sock")"
        fi
    done

    shopt -u nullglob

    # Success only when BOTH groups are present and all ACL updates succeeded.
    if [ "$found_remote" -eq 1 ] && [ "$found_git" -eq 1 ] && [ "$failed_any" -eq 0 ]; then
        return 0
    fi

    return 1
}

main() {
    echo "Its running"
    # Never block startup or manual workflow.
    if [ -z "${USERNAME:-}" ]; then
        log "USERNAME is empty, exiting without failure."
        return 0
    fi

    if ! command -v setfacl >/dev/null 2>&1; then
        log "setfacl not available, exiting without failure."
        return 0
    fi

    log "Socket ACL watcher started for USERNAME=${USERNAME}"

    while true; do
        if apply_acl_once; then
            log "Both socket groups ready and permission setup successful. Exiting watcher."
            return 0
        fi
        sleep "$SLEEP_SECONDS"
    done
}

main || true
exit 0
