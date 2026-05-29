#!/bin/bash
set -e


# ====================
# USER AND GROUP SETUP
# ====================

# Check if the user already exists
if id --user "$USERNAME" >/dev/null 2>&1; then
    echo "user:$USERNAME already exists, skipping user setup ..."
else
    # Check if group exists before creating it
    if ! getent group "$USERNAME" >/dev/null; then
        echo "Creating group: $USERNAME with GID: $USER_GID ..."
        groupadd --gid "$USER_GID" "$USERNAME"
    fi

    # Create user
    echo "Creating user: $USERNAME with UID: $USER_UID and GID: $USER_GID ..."
    useradd --uid "$USER_UID" --gid "$USER_GID" --create-home "$USERNAME"

    # Add sudoers entry only if it doesn't exist
    if ! grep -q "^$USERNAME ALL=(ALL) NOPASSWD:ALL" /etc/sudoers; then
        echo "Adding sudoers entry for $USERNAME ..."
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    fi
fi

# =====================
# ACL PERMISSIONS SETUP
# =====================

echo "Setting ACL permissions for $USERNAME"

# Check if /root exists and if /root has ACL permissions for the user, if not set them

if [ -d "/root" ] ; then
    if ! getfacl /root 2>/dev/null | grep -q "user:$USERNAME:--x"; then
        echo "Setting ACL permissions for $USERNAME on /root"
        setfacl --modify user:"$USERNAME":--x /root
    else
        echo "ACL permissions for $USERNAME on /root already set, skipping ..."
    fi
else
    echo "Warning: /root does not exist"
fi

# Check if .vscode-server exists and if it has ACL permissions for the user, if not set them
if [ -d "/root/.vscode-server" ]; then
    if ! getfacl /root/.vscode-server 2>/dev/null | grep -q "user:$USERNAME:rwx" || \
       ! getfacl --default /root/.vscode-server 2>/dev/null | grep -q "user:$USERNAME:rwx"; then
        echo "Setting ACL permissions for $USERNAME on /root/.vscode-server"
        setfacl --recursive --modify user:"$USERNAME":rwX /root/.vscode-server
        setfacl --recursive --default --modify user:"$USERNAME":rwX /root/.vscode-server
    else
        echo "ACL permissions for $USERNAME on /root/.vscode-server already set, skipping ..."
    fi
else
    echo "Warning: /root/.vscode-server does not exist"
fi

# Check if /nlp-venv exists and if it has ACL permissions for the user, if not set them
# if [ -d "/nlp-venv" ]; then
#     if ! getfacl /nlp-venv 2>/dev/null | grep -q "user:$USERNAME:rwx" || \
#        ! getfacl --default /nlp-venv 2>/dev/null | grep -q "user:$USERNAME:rwx"; then
#         echo "Setting ACL permissions for $USERNAME on /nlp-venv"
#         setfacl --recursive --modify user:"$USERNAME":rwX /nlp-venv
#         setfacl --recursive --default --modify user:"$USERNAME":rwX /nlp-venv
#     else
#         echo "ACL permissions for $USERNAME on /nlp-venv already set, skipping ..."
#     fi
# else
#     echo "Warning: /nlp-venv does not exist"
# fi

# ====
# DONE
# ====

echo "Setup complete for user: $USERNAME"
exit 0
