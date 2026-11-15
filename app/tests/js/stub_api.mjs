// Test stub for app/web/js/api.js (no network).
export class ApiError extends Error {}
export const api = {
  get: async () => ({}),
  post: async () => ({ id: 's1', summary: {} }),
  put: async () => ({ ok: true }),
  del: async () => ({}),
};
