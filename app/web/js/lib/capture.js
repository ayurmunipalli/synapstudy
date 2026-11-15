// Optional lapse screenshots for the video view: one tab-capture stream, one frame per lapse.

const MAX_WIDTH_PX = 960;
// The server accepts up to 2 MB; 2,000,000 base64 characters decode to about 1.5 MB.
const MAX_DATA_URL_CHARS = 2_000_000;

let stream = null;
let video = null;
const endListeners = new Set();

function release() {
  if (stream) {
    for (const track of stream.getTracks()) {
      track.onended = null;
      track.stop();
    }
  }
  if (video) {
    video.pause();
    video.srcObject = null;
  }
  stream = null;
  video = null;
}

export const capture = {
  supported() {
    return Boolean(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia);
  },

  active() {
    return Boolean(stream && stream.active && video);
  },

  // Must be called from a user gesture. Resolves { ok, reason }; never rejects.
  async request() {
    if (this.active()) return { ok: true, reason: null };
    if (!this.supported()) return { ok: false, reason: 'this browser cannot capture the window' };
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({ video: true, preferCurrentTab: true });
    } catch (err) {
      stream = null;
      const refused = err && (err.name === 'NotAllowedError' || err.name === 'AbortError');
      return { ok: false, reason: refused ? 'sharing was not allowed' : `capture failed (${err.name || 'error'})` };
    }
    const track = stream.getVideoTracks()[0];
    if (!track) {
      release();
      return { ok: false, reason: 'the shared source has no video' };
    }
    // Frames are cropped with this tab's own coordinates, so anything but a tab would save the wrong pixels.
    const surface = track.getSettings().displaySurface;
    if (surface && surface !== 'browser') {
      release();
      return { ok: false, reason: 'a window or screen was shared instead of this tab' };
    }
    track.onended = () => {
      release();
      for (const fn of [...endListeners]) fn();
    };
    video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    try {
      await video.play();
    } catch (err) {
      release();
      return { ok: false, reason: `capture failed (${err.name || 'error'})` };
    }
    return { ok: true, reason: null };
  },

  // Called when the user stops sharing from the browser's own bar. Returns off().
  onEnded(fn) {
    endListeners.add(fn);
    return () => endListeners.delete(fn);
  },

  // One JPEG data URL of the shared tab cropped to `el`, or null when no frame can be taken.
  async grab(el) {
    if (!this.active() || !video.videoWidth || !video.videoHeight) return null;
    const rect = el.getBoundingClientRect();
    const left = Math.max(0, rect.left);
    const top = Math.max(0, rect.top);
    const width = Math.min(window.innerWidth, rect.right) - left;
    const height = Math.min(window.innerHeight, rect.bottom) - top;
    if (width < 2 || height < 2) return null;

    // The stream shows the tab's viewport, usually at device-pixel resolution.
    const sx = video.videoWidth / window.innerWidth;
    const sy = video.videoHeight / window.innerHeight;
    const scale = Math.min(1, MAX_WIDTH_PX / (width * sx));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(width * sx * scale));
    canvas.height = Math.max(1, Math.round(height * sy * scale));
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(video, left * sx, top * sy, width * sx, height * sy, 0, 0, canvas.width, canvas.height);

    for (const quality of [0.82, 0.65, 0.45]) {
      const url = canvas.toDataURL('image/jpeg', quality);
      if (url.length <= MAX_DATA_URL_CHARS) return url;
    }
    return null;
  },

  stop() {
    release();
    endListeners.clear();
  },
};
