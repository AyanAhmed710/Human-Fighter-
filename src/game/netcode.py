"""LAN 1v1 networking -- minimal replicated-input netcode, not a client-server
authoritative simulation.

Both laptops run the FULL game (identical src/game/match.py Match, identical
RoundMatch, identical HUD/rendering) -- only the two players' raw actions
("punch"/"kick"/"shoot", plus the "restart" control signal) get replicated
over the wire. Each machine applies its own local action to its own
match.try_action() the instant it happens (and sends it to the peer), and
applies the peer's action to the OTHER side's Player the instant it arrives.
match.py's state machine is a deterministic function of the action sequence
plus wall-clock timers, not physics/frame-count dependent, so two machines
fed the same two action-streams converge to matching health/round/win state
on their own -- only a few ms of skew (real LAN latency) between when a
given hit visibly lands on each screen, no full state snapshot ever needs to
cross the wire.

Why not host-authoritative state broadcast instead: scripts/play_game.py's
Game/HUD/render code is written entirely around owning ONE local Match
object it freely reads every frame (health, current_action, timers,
banners, ...). Splitting that into "sim on host, render-only mirror on
client" would mean teaching every HUD/animation call site to read from a
network snapshot instead of local state. Replicating the small, infrequent
action stream instead lets every existing line of Game/RoundMatch/HUD code
stay untouched -- both machines just ARE the authority for their own
player's actions.

Fighter picking is NOT synchronized/split between machines -- whoever hosts
picks both fighters in the existing single-machine CharacterSelect screen
(same as local play), and that choice is sent to the joining client in the
"start" message. Keeps this file (and the menu flow around it) from needing
a second character-select screen and a pick-sync protocol on top of the
actual gameplay networking.

Transport: plain TCP, newline-delimited single-line JSON messages, one
background reader thread per socket appending decoded messages to a
thread-safe queue so the ursina main thread never blocks -- Game.update()
only ever does a non-blocking drain, same shape as the existing
PlayerCameraInput.get_action_nowait().
"""
import json
import queue
import socket
import threading

PORT = 5763  # arbitrary unassigned-range port -- nothing else on a home LAN uses it


def local_ip() -> str:
    """Best-effort LAN IP to show the host so they can read it out to
    whoever's joining. Opens a UDP socket toward a public IP -- UDP connect()
    never actually sends a packet, it just makes the OS pick the right local
    route/interface -- so this works on multi-NIC machines instead of
    guessing from the hostname (which can resolve to 127.0.0.1 on Windows)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class _NetSession:
    """Shared plumbing for both NetHost and NetClient once a socket is
    connected -- one reader thread appending decoded messages to self.inbox,
    send() writes a newline-framed JSON line straight to the socket. send()
    is only ever called from the ursina main thread (single writer), so no
    lock is needed on the socket itself."""

    role = None  # "host" | "client", set by the subclass

    def __init__(self, sock: socket.socket):
        # Nagle's algorithm (TCP's default) batches/delays small outgoing
        # packets waiting for either more data or the previous packet's ACK
        # -- fine for bulk transfer, but this connection only ever sends
        # small, infrequent messages (one action every second or so), which
        # is exactly the pattern Nagle adds up to ~40ms of pure waiting to.
        # Disabling it (TCP_NODELAY) sends each action the instant it's
        # queued -- on a real LAN that turns "network delay" into low-single-
        # digit ms (WiFi ping to the router/AP), not a perceptible input lag.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = sock
        self.inbox: "queue.Queue[dict]" = queue.Queue()
        self.connected = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        buf = b""
        try:
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line:
                        continue
                    try:
                        self.inbox.put(json.loads(line.decode("utf-8")))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass  # drop a corrupt line rather than kill the connection
        except OSError:
            pass
        finally:
            self.connected = False

    def send(self, msg: dict):
        if not self.connected:
            return
        try:
            self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
        except OSError:
            self.connected = False

    def poll(self):
        """Drains every message received since the last poll -- usually
        empty or length 1; only longer if the frame that just ran was
        unusually slow. Order-preserving (FIFO)."""
        msgs = []
        while True:
            try:
                msgs.append(self.inbox.get_nowait())
            except queue.Empty:
                break
        return msgs

    def send_action(self, action: str):
        self.send({"type": "action", "action": action})

    def send_restart(self):
        self.send({"type": "restart"})

    def close(self):
        self.connected = False
        try:
            self.sock.close()
        except OSError:
            pass


class NetHost(_NetSession):
    """Opens a listening socket immediately; accept + the client's hello are
    handled on a background thread so the menu screen can keep polling
    is_connected() every frame without blocking. Does NOT call
    _NetSession.__init__ until a client has actually connected -- poll
    is_connected() before calling send()/poll()/send_action()."""

    role = "host"

    def __init__(self, port: int = PORT):
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("0.0.0.0", port))
        self._listener.listen(1)
        self.port = port
        self.peer_username = None
        self._accepted = threading.Event()
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        try:
            conn, _addr = self._listener.accept()
        except OSError:
            return  # listener closed (host backed out) before anyone connected
        try:
            self._listener.close()
        except OSError:
            pass
        # read the client's one-line hello synchronously on this background
        # thread before flipping _accepted, so peer_username is already
        # populated the instant the main thread's poll sees is_connected().
        buf = b""
        hello = {}
        try:
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
            if b"\n" in buf:
                line, _rest = buf.split(b"\n", 1)
                hello = json.loads(line.decode("utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            hello = {}
        self.peer_username = hello.get("username") or "PLAYER 2"
        _NetSession.__init__(self, conn)
        self._accepted.set()

    def is_connected(self) -> bool:
        return self._accepted.is_set() and self.connected

    def send_match_start(self, model_p1: str, model_p2: str, username_p1: str):
        self.send({"type": "start", "model_p1": model_p1, "model_p2": model_p2,
                   "username_p1": username_p1})

    def close(self):
        try:
            self._listener.close()
        except OSError:
            pass
        super().close()


class NetClient(_NetSession):
    """Connects immediately (blocking, with a timeout) -- construction
    raises OSError/socket.timeout on failure, which the caller (menu flow)
    catches to show a "couldn't connect" status instead of crashing."""

    role = "client"

    def __init__(self, host_ip: str, username: str, port: int = PORT, timeout: float = 5.0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host_ip, port))
        sock.settimeout(None)  # back to blocking for the reader thread's recv loop
        sock.sendall((json.dumps({"username": username}) + "\n").encode("utf-8"))
        super().__init__(sock)
        self.start_info = None  # (model_p1, model_p2, username_p1) once the host's "start" arrives

    def poll_start(self):
        """Non-blocking check for the host's match-start signal -- called
        from the "connecting..." menu screen's per-frame poll. Any
        non-"start" message that arrives first (shouldn't normally happen,
        but the wire doesn't guarantee it) is put back so Game's own poll()
        still sees it in order once the match actually begins."""
        for msg in self.poll():
            if msg.get("type") == "start":
                self.start_info = (msg["model_p1"], msg["model_p2"], msg["username_p1"])
            else:
                self.inbox.put(msg)
        return self.start_info
