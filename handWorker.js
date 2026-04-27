// handWorker.js — MediaPipe HandLandmarker running in a dedicated Web Worker.
//
// Why: detectForVideo takes 15-40 ms on desktop, 30-80 ms on mobile. Previously
// that blocked the main thread every video frame, so rendering had to wait.
// Moving it here lets the render loop run at vsync even while inference is in
// flight — effective FPS ≈ max(render_cost, detect_cost) instead of their sum.
//
// Protocol:
//   ← { type: 'init' }                         → { type: 'ready' | 'error' }
//   ← { type: 'detect', bitmap, timestamp,
//       frameId }                               → { type: 'results', results, frameId, detectMs }
//                                                 (bitmap.close() called in worker)
//   ← { type: 'recover-cpu' }                  → { type: 'recovered' | 'error' }
//
// GPU delegate works in a module-type worker from tasks-vision 0.10+ — the
// library internally creates its own OffscreenCanvas for GL state. If init
// fails (old browser, no module-worker, cross-origin restriction on the CDN,
// etc.), the worker posts 'error' and main.js falls back to the in-page path.

import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let handLandmarker = null;
let vision = null;
let currentDelegate = 'GPU';
let detecting = false;

async function createHandLandmarker(delegate) {
  return await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: delegate
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6
  });
}

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === 'init') {
    try {
      vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
      );
      handLandmarker = await createHandLandmarker('GPU');
      self.postMessage({ type: 'ready', delegate: currentDelegate });
    } catch (err) {
      // GPU failed — try CPU before giving up. Some mobile browsers block
      // WebGL in workers even when the main thread has it.
      try {
        handLandmarker = await createHandLandmarker('CPU');
        currentDelegate = 'CPU';
        self.postMessage({ type: 'ready', delegate: 'CPU' });
      } catch (err2) {
        self.postMessage({ type: 'error', error: err2?.message || String(err2) });
      }
    }
    return;
  }

  if (msg.type === 'detect') {
    // Drop the frame if we're still processing the previous one — it's
    // already stale by the time we'd get to it, and queueing causes visible
    // lag. The next rVFC tick on the main thread will send a fresh bitmap.
    if (!handLandmarker || detecting) {
      try { msg.bitmap && msg.bitmap.close && msg.bitmap.close(); } catch (_) {}
      self.postMessage({ type: 'dropped', frameId: msg.frameId });
      return;
    }
    detecting = true;
    const t0 = performance.now();
    try {
      const results = handLandmarker.detectForVideo(msg.bitmap, msg.timestamp);
      // MediaPipe's result arrays are reused frame-to-frame, so we must
      // structure-copy what we send back or the main thread reads stale data
      // when the next detect overwrites them. postMessage clones by default,
      // but we strip to just the fields main.js uses to keep the copy small.
      const safe = {
        landmarks: results.landmarks ? results.landmarks.map(arr => arr.map(p => ({ x: p.x, y: p.y, z: p.z, visibility: p.visibility }))) : [],
        worldLandmarks: results.worldLandmarks ? results.worldLandmarks.map(arr => arr.map(p => ({ x: p.x, y: p.y, z: p.z, visibility: p.visibility }))) : [],
        handednesses: results.handednesses ? results.handednesses.map(arr => arr.map(h => ({ categoryName: h.categoryName, score: h.score }))) : []
      };
      const detectMs = performance.now() - t0;
      self.postMessage({
        type: 'results',
        results: safe,
        frameId: msg.frameId,
        detectMs,
        delegate: currentDelegate
      });
    } catch (err) {
      self.postMessage({
        type: 'detect-error',
        frameId: msg.frameId,
        error: err?.message || String(err)
      });
    } finally {
      detecting = false;
      try { msg.bitmap && msg.bitmap.close && msg.bitmap.close(); } catch (_) {}
    }
    return;
  }

  if (msg.type === 'recover-cpu') {
    try {
      try { handLandmarker && handLandmarker.close(); } catch (_) {}
      handLandmarker = await createHandLandmarker('CPU');
      currentDelegate = 'CPU';
      self.postMessage({ type: 'recovered', delegate: 'CPU' });
    } catch (err) {
      self.postMessage({ type: 'error', error: err?.message || String(err) });
    }
    return;
  }
};
