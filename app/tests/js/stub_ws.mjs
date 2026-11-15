// Test stub for app/web/js/ws.js (no socket).
export const sent = [];
export const ws = { send: (o) => { sent.push(o); return true; }, connected: () => true };
