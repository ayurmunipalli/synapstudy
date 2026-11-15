"""Test 8: live deep-sleep detector that plays quiet cues.

  python sleep_cue.py --model results/nap-XXXX_model.npz --tap-threshold T --volume 0.15 --minutes 90
  python sleep_cue.py --threshold 0.70 --beta-threshold 0.012 --tap-threshold T      # rule instead of model
  python sleep_cue.py ... --cues cues/      # folder of .wav files, one per word

Every 5 s it looks at the last 30 s of forehead EEG (AF7/AF8) and decides "deep" with one of (see detector.py):
  --model           personal model trained by stage_nap.py on an earlier nap: P(N3) > 0.5 held --persist s
  --threshold       rule: delta 0.5-2 Hz share > threshold AND (if given) beta 15-30 Hz share < --beta-threshold,
                    held --persist s. (--persist 5 with no --beta-threshold = the kit's original rule.)
While deep and the head still, one cue every --every s. Head movement pauses cues for 60 s.
Double-tap the headband's forehead pod = SNOOZE: cues held 15 min, a soft low tone confirms (needs --tap-threshold
from snooze_check.py). Every check, cue and snooze is logged in the recording.
Afterwards run stage_nap.py on the same file to check where the cues landed. Test the volume awake first.
(Model, wake guard and snooze added 2026-11-14; designs fixed in RESULTS.md before testing.)
"""
import argparse
import glob
import os
import random
import time

import numpy as np

from common import FS, Recorder, local_clock, stamp
from detector import PERSIST_GUARDED, DeepSleepDetector, PersonalModelDetector, find_double_taps, load_model

SNOOZE_S = 15 * 60


def make_tone(volume, freq=440.0, dur=0.25):
    import pygame
    sr = 44100
    t = np.arange(int(sr * dur)) / sr
    wave = np.sin(2 * np.pi * freq * t) * np.hanning(len(t)) * 0.5
    snd = pygame.sndarray.make_sound(np.repeat((wave * 32767).astype(np.int16)[:, None], 2, axis=1))
    snd.set_volume(volume)
    return snd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="personal model .npz from stage_nap.py")
    ap.add_argument("--threshold", type=float, help="delta-share rule instead of a model")
    ap.add_argument("--beta-threshold", type=float, default=None,
                    help="wake guard for the rule: from stage_nap.py (median beta share in N2)")
    ap.add_argument("--persist", type=float, default=PERSIST_GUARDED,
                    help="seconds 'deep' must hold before the first cue of a bout")
    ap.add_argument("--tap-threshold", type=float, default=None, help="from snooze_check.py; omit = no snooze")
    ap.add_argument("--volume", type=float, default=0.15)
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--cues", help="folder of .wav files")
    ap.add_argument("--every", type=float, default=8.0)
    a = ap.parse_args()
    if (a.model is None) == (a.threshold is None):
        raise SystemExit("Give exactly one of --model or --threshold.")

    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    if a.cues:
        files = sorted(glob.glob(os.path.join(a.cues, "*.wav")))
        if not files:
            raise SystemExit(f"No .wav files in {a.cues}")
        sounds = []
        for f in files:
            s = pygame.mixer.Sound(f)
            s.set_volume(a.volume)
            sounds.append((os.path.splitext(os.path.basename(f))[0], s))
    else:
        sounds = [("beep", make_tone(a.volume))]
    snooze_tone = make_tone(a.volume, freq=330.0, dur=0.15)

    if a.model:
        model = load_model(a.model)
        det = PersonalModelDetector(model, persist=a.persist)
        # Models from train_n2n3_model.py carry their target ("N2/N3"); older ones are N3 models (2026-11-15).
        desc = f"personal model {a.model}: P({model.get('target', 'N3')}) > 0.5 held {a.persist:g} s"
    else:
        det = DeepSleepDetector(a.threshold, a.beta_threshold, a.persist)
        desc = f"rule: delta > {a.threshold}, beta < {a.beta_threshold}, held {a.persist:g} s"
        if a.beta_threshold is None:
            print("Warning: no --beta-threshold, so the beta wake guard is OFF.")
    if a.tap_threshold is None:
        print("Warning: no --tap-threshold, so double-tap snooze is OFF.")

    rec = Recorder(f"results/cuenap-{stamp()}", need_motion=True, autosave_every=60)
    rec.mark("lights_out")
    rec.mark(f"detector|{desc}|tap_threshold={a.tap_threshold}")
    print(f"Detector: {desc}. Snooze: {'double-tap, 15 min' if a.tap_threshold else 'off'}. Ctrl+C to stop.")
    deep, paused_until, last_cue, n_cues, n_snooze, order = False, 0.0, 0.0, 0, 0, []
    tap_scan_from, last_snooze = local_clock(), -1e9
    end = time.time() + a.minutes * 60
    try:
        while time.time() < end:
            time.sleep(5 if not deep else 0.5)
            now = local_clock()
            if a.tap_threshold is not None:
                acc, ts = rec.since("motion", tap_scan_from - 1.0)
                if len(ts):
                    dts = [t for t in find_double_taps(acc[:, :3], ts, a.tap_threshold) if t > last_snooze + 1.0]
                    tap_scan_from = ts[-1]
                    if dts:
                        last_snooze = dts[-1]
                        rec.mark("snooze", dts[-1])
                        n_snooze += 1
                        snooze_tone.play()
                        deep, paused_until = False, now + SNOOZE_S
                        det.reset()
                        print(f"{time.strftime('%H:%M:%S')} SNOOZE {n_snooze}: cues held 15 min")
                        continue
            eeg = rec.latest("eeg", 30)
            motion = rec.latest("motion", 30, fs=52)
            if eeg is None or motion is None:
                continue
            moving = motion[-52 * 5:, 3:6].std(axis=0).max() > 3.0
            if moving:
                if deep:
                    rec.mark("cues_held_movement")
                    print(f"{time.strftime('%H:%M:%S')} movement: cues held 60 s")
                deep, paused_until = False, max(paused_until, now + 60)
                det.reset()
                continue
            deep = det.update(eeg[:, 1:3], FS, now)
            v1, v2 = det.last
            rec.mark(f"check|{v1:.3f}|{v2:.3f}|deep|{int(deep)}")
            if deep and now >= paused_until and now - last_cue >= a.every:
                if not order:
                    order = random.sample(range(len(sounds)), len(sounds))
                name, snd = sounds[order.pop()]
                snd.play()
                last_cue = rec.mark(f"cue|{name}")
                n_cues += 1
                print(f"{time.strftime('%H:%M:%S')} cue {n_cues}: {name}  ({v1:.2f})")
    except KeyboardInterrupt:
        pass
    rec.mark("lights_on")
    rec.save()
    print(f"{n_cues} cues played, {n_snooze} snoozes. Now run: python stage_nap.py {rec.path}.npz")


if __name__ == "__main__":
    main()
