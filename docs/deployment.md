# Password and deployment

## Password and access

<!-- help: Password and access -->

To create the password hash, see [Set a password](install.md#set-a-password).

## Access from another computer

### Single user

Keep the default `--host 127.0.0.1` and reach the app through an SSH tunnel
or a private network such as Tailscale:

```bash
ssh -L 8501:localhost:8501 you@server
```

### Team access over the internet

Keep the app on `127.0.0.1` and put a reverse proxy with HTTPS in front of
it. For example, with this Caddyfile:

```
tbook.example.com {
    reverse_proxy 127.0.0.1:8501
}
```

Without HTTPS, the password is sent over the network in plain text.

## Security

### Password

Only a salted scrypt hash of the password is stored, in an environment
variable. Anyone who can read the process environment on the server can
see the hash, but the hash does not reveal the password. Each wrong attempt
waits one second. To change the password, generate a new hash and restart
the app. Open sessions end on restart.

### Files

Files can only be downloaded from a logged-in session, through a random,
short-lived link. Paths are checked to be inside the runs folder, so
symlinks that point elsewhere are refused. Streamlit's static file serving
is enabled only for the fonts bundled with the package. It never serves the
runs folder. Do not point another web server at the runs folder.

### TensorBoard

TensorBoard has no password. TensorBoard instances started from the app
therefore listen on `127.0.0.1` by default, and their links work on the
machine that runs the app. When the app runs on a server, forward the
TensorBoard port as well. The panel in the app shows the command, for
example:

```bash
ssh -L 6006:localhost:6006 you@server
```

`--tensorboard-host 0.0.0.0` makes the instances reachable from the
network. This exposes the selected runs to anyone who can reach that port,
so use it only on a trusted network.

### Large files

File previews read at most 300 KB. Zip downloads are limited to 1 GB, which
can be changed in **Manage → Settings**. A single file download is read into
memory, so copy very large checkpoints with `scp` or `rsync` instead.
