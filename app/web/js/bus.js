// A tiny publish/subscribe hub shared by every module.

const listeners = new Map();

export const bus = {
  /** Subscribe; returns a function that unsubscribes. */
  on(type, fn) {
    if (typeof fn !== 'function') throw new TypeError('bus.on needs a function');
    let set = listeners.get(type);
    if (!set) listeners.set(type, (set = new Set()));
    set.add(fn);
    return () => {
      set.delete(fn);
      if (!set.size && listeners.get(type) === set) listeners.delete(type);
    };
  },

  emit(type, payload) {
    const set = listeners.get(type);
    if (!set) return;
    // Copy first: a listener may unsubscribe itself (or others) while we iterate.
    for (const fn of [...set]) {
      try {
        fn(payload);
      } catch (err) {
        console.error(`bus listener for "${type}" failed`, err);
      }
    }
  },
};
