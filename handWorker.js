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
    minTrackingConfidence: 0.6,
    output_segmentation_masks: true,
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

      // Extract segmentation mask — measure finger width in worker so we only
      // transfer a small downsampled preview instead of the full-res mask.
      let measuredRadius = 0;
      let segPreview = null, segPreviewW = 0, segPreviewH = 0;

      const maskObj = results.segmentationMasks?.[0];
      if (maskObj) {
        const maskW = maskObj.width;
        const maskH = maskObj.height;
        const f32   = maskObj.getAsFloat32Array();

        // ── Measure ring finger width (LM13 = ring MCP) ─────────────────────
        // ── Method B: Geometric Bounding Boxes ─────────────────────────────
        // Instead of a single scanline at the joint, we find the bounding box
        // of all "on" pixels in the mask around the finger segment. The width
        // of this box (perpendicular to the segment) is the finger thickness.
        const anchorStart = msg.anchorStart ?? 13;
        const anchorEnd   = msg.anchorEnd   ?? 14;
        const lmA = results.landmarks?.[0]?.[anchorStart];
        const lmB = results.landmarks?.[0]?.[anchorEnd];

        if (lmA && lmB) {
          const x1 = Math.round(lmA.x * maskW);
          const y1 = Math.round(lmA.y * maskH);
          const x2 = Math.round(lmB.x * maskW);
          const y2 = Math.round(lmB.y * maskH);

          // Padding ensures we capture the full finger width even if the
          // landmarks are slightly off-center. 25 pixels ≈ 10-15% of frame.
          const padding = 25;
          const minBX = Math.max(0, Math.min(x1, x2) - padding);
          const maxBX = Math.min(maskW - 1, Math.max(x1, x2) + padding);
          const minBY = Math.max(0, Math.min(y1, y2) - padding);
          const maxBY = Math.min(maskH - 1, Math.max(y1, y2) + padding);

          let onMinX = maxBX, onMaxX = minBX;
          let onMinY = maxBY, onMaxY = minBY;
          let found = false;

          for (let y = minBY; y <= maxBY; y++) {
            for (let x = minBX; x <= maxBX; x++) {
              if (f32[y * maskW + x] > 0.5) {
                if (x < onMinX) onMinX = x;
                if (x > onMaxX) onMaxX = x;
                if (y < onMinY) onMinY = y;
                if (y > onMaxY) onMaxY = y;
                found = true;
              }
            }
          }

          if (found) {
            const dx = Math.abs(x2 - x1);
            const dy = Math.abs(y2 - y1);
            // If segment is more vertical, width is horizontal span; else vertical span.
            const widthPixels = (dy > dx) ? (onMaxX - onMinX + 1) : (onMaxY - onMinY + 1);
            // Ortho camera spans 2 world units across full frame width.
            // (widthPixels / maskW) * 2.0 is the full width in world units.
            measuredRadius = (widthPixels / maskW) * 2.0 / 2;

            // Optional: send the bbox back for visualization
            self.measuredBBox = { minX: onMinX, maxX: onMaxX, minY: onMinY, maxY: onMaxY };
          } else {
            self.measuredBBox = null;
          }
        } else {
          self.measuredBBox = null;
        }

        // ── Downsample 4× for visualizer (160×90 ≈ 14 KB vs 300 KB full) ───
        const SCALE  = 4;
        segPreviewW  = Math.floor(maskW / SCALE);
        segPreviewH  = Math.floor(maskH / SCALE);
        segPreview   = new Uint8Array(segPreviewW * segPreviewH);
        for (let y = 0; y < segPreviewH; y++) {
          const srcRow = (y * SCALE) * maskW;
          const dstRow = y * segPreviewW;
          for (let x = 0; x < segPreviewW; x++) {
            segPreview[dstRow + x] = f32[srcRow + x * SCALE] > 0.5 ? 1 : 0;
          }
        }

        try { maskObj.close(); } catch (_) {}
      }

      const detectMs = performance.now() - t0;
      self.postMessage({
        type: 'results',
        results: safe,
        frameId: msg.frameId,
        detectMs,
        delegate: currentDelegate,
        measuredRadius,   // float: ring finger radius in world units (0 = no hand)
        measuredBBox: self.measuredBBox, // Bounding box for visualization
        segPreview,       // Uint8Array 1/4-res mask for visualizer (or null)
        segPreviewW,
        segPreviewH,
      }, segPreview ? [segPreview.buffer] : []);
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
