// JSON over HTTP to the local server. Every failure becomes an ApiError {status, message}.

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

// Paths may be given as "/api/sessions" or "/sessions"; both reach the same endpoint.
function resolve(path) {
  const p = path.startsWith('/') ? path : `/${path}`;
  return p === '/api' || p.startsWith('/api/') ? p : `/api${p}`;
}

async function request(method, path, body) {
  const init = { method, headers: { Accept: 'application/json' } };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }

  let res;
  try {
    res = await fetch(resolve(path), init);
  } catch {
    throw new ApiError(0, 'The app server is not responding. Is it still running?');
  }

  let data = null;
  const text = await res.text().catch(() => '');
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }

  if (!res.ok) {
    const detail = data && (data.error ?? data.detail);
    const message = typeof detail === 'string' && detail ? detail : `The server answered ${res.status} ${res.statusText}`.trim();
    throw new ApiError(res.status, message);
  }
  return data;
}

export const api = {
  get: (path) => request('GET', path),
  post: (path, body) => request('POST', path, body === undefined ? {} : body),
  put: (path, body) => request('PUT', path, body === undefined ? {} : body),
  del: (path) => request('DELETE', path),
};
