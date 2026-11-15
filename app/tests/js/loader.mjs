// Node module hook: load the real frontend modules, but swap api.js and ws.js for the stubs.
export async function resolve(specifier, context, nextResolve) {
  const r = await nextResolve(specifier, context);
  if (r.url.endsWith('/app/web/js/api.js')) return { ...r, url: new URL('./stub_api.mjs', import.meta.url).href };
  if (r.url.endsWith('/app/web/js/ws.js')) return { ...r, url: new URL('./stub_ws.mjs', import.meta.url).href };
  return r;
}
