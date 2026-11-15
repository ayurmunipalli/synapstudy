// One WebSocket to /ws, reconnecting on its own. Each server message is re-emitted on the bus as
// "ws:<type>". Connection changes are emitted as "socket:open" and "socket:close" (a separate prefix, so
// they can never be mistaken for a server message).

import { bus } from './bus.js';

const RETRY_MIN_MS = 500;
const RETRY_MAX_MS = 5000;

let socket = null;
let open = false;
let attempts = 0;
let retryTimer = null;

function url() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${scheme}://${location.host}/ws`;
}

function scheduleReconnect() {
  clearTimeout(retryTimer);
  const delay = Math.min(RETRY_MAX_MS, RETRY_MIN_MS * 2 ** attempts);
  attempts += 1;
  retryTimer = setTimeout(connect, delay);
}

function connect() {
  clearTimeout(retryTimer);
  let s;
  try {
    s = new WebSocket(url());
  } catch {
    scheduleReconnect();
    return;
  }
  socket = s;

  s.onopen = () => {
    if (s !== socket) return;
    open = true;
    attempts = 0;
    bus.emit('socket:open');
  };

  s.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    if (msg && typeof msg.type === 'string') bus.emit(`ws:${msg.type}`, msg);
  };

  s.onclose = () => {
    if (s !== socket) return;
    const wasOpen = open;
    open = false;
    socket = null;
    if (wasOpen) bus.emit('socket:close');
    scheduleReconnect();
  };

  // An error is always followed by close in browsers; closing here just makes that prompt.
  s.onerror = () => {
    try {
      s.close();
    } catch {
      /* already closed */
    }
  };
}

export const ws = {
  /** Send one message. Returns false (and drops it) when the socket is not open: nothing is queued,
   *  so a stale simulated key press can never be delivered late. */
  send(obj) {
    if (!open || !socket || socket.readyState !== WebSocket.OPEN) return false;
    try {
      socket.send(JSON.stringify(obj));
      return true;
    } catch {
      return false;
    }
  },

  connected() {
    return open;
  },
};

connect();
