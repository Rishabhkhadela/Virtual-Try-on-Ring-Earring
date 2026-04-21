import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
import { applyJewelryShading } from './Shader.js';

// =============================================================================
// ONE-EURO FILTER — Jitter Mitigation (replaces rolling-average + adaptiveLerp)
// =============================================================================
class LowPassFilter {
  constructor(alpha, initval = 0) {
    this.s = initval;
    this.initialized = false;
    this.setAlpha(alpha);
  }
  setAlpha(a) { this.alpha = Math.max(0.000001, Math.min(1, a)); }
  filter(value) {
    if (!this.initialized) { this.s = value; this.initialized = true; return value; }
    this.s = this.alpha * value + (1 - this.alpha) * this.s;
    return this.s;
  }
  hatValue() { return this.s; }
  reset() { this.initialized = false; }
}

class OneEuroFilter {
  constructor(freq, minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xFilt = new LowPassFilter(this._alpha(minCutoff));
    this.dxFilt = new LowPassFilter(this._alpha(dCutoff), 0);
    this.lastTime = null;
  }
  _alpha(cutoff) {
    const te = 1.0 / this.freq;
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }
  filter(x, timestamp) {
    if (this.lastTime !== null && timestamp > this.lastTime) {
      this.freq = 1.0 / (timestamp - this.lastTime);
    }
    this.lastTime = timestamp;
    const prevX = this.xFilt.hatValue();
    const dx = this.xFilt.initialized ? (x - prevX) * this.freq : 0;
    this.dxFilt.setAlpha(this._alpha(this.dCutoff));
    const edx = this.dxFilt.filter(dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    this.xFilt.setAlpha(this._alpha(cutoff));
    return this.xFilt.filter(x);
  }
  reset() {
    this.xFilt.reset();
    this.dxFilt.reset();
    this.lastTime = null;
  }
}

class OneEuroFilterVec3 {
  constructor(freq, minCutoff, beta, dCutoff) {
    this.fx = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.fy = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.fz = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
  }
  filter(v, t) {
    return new THREE.Vector3(this.fx.filter(v.x, t), this.fy.filter(v.y, t), this.fz.filter(v.z, t));
  }
  reset() { this.fx.reset(); this.fy.reset(); this.fz.reset(); }
}

class OneEuroFilterQuat {
  constructor(freq, minCutoff, beta, dCutoff) {
    this.fx = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.fy = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.fz = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.fw = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.prev = new THREE.Quaternion();
  }
  filter(q, t) {
    // Ensure shortest-path continuity
    if (this.prev.dot(q) < 0) q = q.clone().set(-q.x, -q.y, -q.z, -q.w);
    this.prev.copy(q);
    const r = new THREE.Quaternion(
      this.fx.filter(q.x, t), this.fy.filter(q.y, t),
      this.fz.filter(q.z, t), this.fw.filter(q.w, t)
    );
    r.normalize();
    return r;
  }
  reset() { this.fx.reset(); this.fy.reset(); this.fz.reset(); this.fw.reset(); }
}

class OneEuroFilterScalar {
  constructor(freq, minCutoff, beta, dCutoff) {
    this.f = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
  }
  filter(v, t) { return this.f.filter(v, t); }
  reset() { this.f.reset(); }
}

// =============================================================================
// STATE
// =============================================================================
let faceLandmarker = null;
let leftEarring = null;
let rightEarring = null;
let leftEarringInner = null;
let rightEarringInner = null;
let lastFaceVideoTime = -1;
let isFacePresent = false;
let leftEarVisible = false;
let rightEarVisible = false;
let faceMeshConnections = null;

// Shared face results for other modules
let _lastLandmarks = null;
let _lastFaceBasis = null;
let _lastTransformMatrix = null;

// Smooth interpolation targets
const targetLeftPos = new THREE.Vector3();
const targetRightPos = new THREE.Vector3();
const targetLeftQuat = new THREE.Quaternion();
const targetRightQuat = new THREE.Quaternion();
const targetEarringScale = new THREE.Vector3(1, 1, 1);

// One-Euro filters — tuned for LOW LAG:
//   high beta  = cutoff rises fast with speed → near-instant tracking on movement
//   minCutoff  = smoothing floor when stationary (kills jitter at rest)
const FILT_FREQ = 30;
const leftPosFilter  = new OneEuroFilterVec3(FILT_FREQ, 1.0, 0.4, 1.0);
const rightPosFilter = new OneEuroFilterVec3(FILT_FREQ, 1.0, 0.4, 1.0);
// NOTE: quat filters removed — pendulum spring-damper already smooths rotation
const scaleFilter = new OneEuroFilterScalar(FILT_FREQ, 1.0, 0.05, 1.0);

// Pendulum physics state
const pendulum = {
  left:  { angle: 0, velocity: 0 },
  right: { angle: 0, velocity: 0 }
};
let lastPendulumTs = 0;

// Face occluder
let faceOccluderMesh = null;
let faceOccluderGeom = null;

// Depth scale for MediaPipe Z → Three.js world Z
const DEPTH_SCALE = 3.0;

// Face oval landmark indices (ordered loop)
const FACE_OVAL_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
];
const OVAL_CENTER_IDX = 4; // nose bridge — center of fan

// Passed-in references
let _scene, _ctx, _canvas, _camera, _mapToOrtho;

const gltfLoader = new THREE.GLTFLoader();

// =============================================================================
// FACE MESH CONNECTIONS (unchanged)
// =============================================================================
const FACE_OVAL = [
  [10,338],[338,297],[297,332],[332,284],[284,251],[251,389],[389,356],[356,454],
  [454,323],[323,361],[361,288],[288,397],[397,365],[365,379],[379,378],[378,400],
  [400,377],[377,152],[152,148],[148,176],[176,149],[149,150],[150,136],[136,172],
  [172,58],[58,132],[132,93],[93,234],[234,127],[127,162],[162,21],[21,54],
  [54,103],[103,67],[67,109],[109,10]
];
const LEFT_EYE = [
  [263,249],[249,390],[390,373],[373,374],[374,380],[380,381],[381,382],[382,362],
  [362,263],[263,466],[466,388],[388,387],[387,386],[386,385],[385,384],[384,398]
];
const RIGHT_EYE = [
  [33,7],[7,163],[163,144],[144,145],[145,153],[153,154],[154,155],[155,133],
  [133,33],[33,246],[246,161],[161,160],[160,159],[159,158],[158,157],[157,173]
];
const LIPS = [
  [61,146],[146,91],[91,181],[181,84],[84,17],[17,314],[314,405],[405,321],[321,375],
  [375,291],[291,409],[409,270],[270,269],[269,267],[267,0],[0,37],[37,39],[39,40],
  [40,185],[185,61]
];
const LEFT_EYEBROW = [[276,283],[283,282],[282,295],[295,285],[300,293],[293,334],[334,296],[296,336]];
const RIGHT_EYEBROW = [[46,53],[53,52],[52,65],[65,55],[70,63],[63,105],[105,66],[66,107]];
const NOSE = [[168,6],[6,197],[197,195],[195,5],[5,4],[4,1],[1,19],[19,94],[94,2],[2,164]];

function buildFallbackConnections() {
  const all = [...FACE_OVAL, ...LEFT_EYE, ...RIGHT_EYE, ...LIPS, ...LEFT_EYEBROW, ...RIGHT_EYEBROW, ...NOSE];
  return all.map(([start, end]) => ({ start, end }));
}

function getFaceMeshConnections() {
  if (faceMeshConnections) return faceMeshConnections;
  try {
    const tess = FaceLandmarker.FACE_LANDMARKS_TESSELATION;
    if (tess && tess.length > 0) {
      faceMeshConnections = tess;
      console.log(`[FaceTracking] Using MediaPipe tessellation (${tess.length} connections)`);
      return faceMeshConnections;
    }
  } catch (e) {
    console.warn("[FaceTracking] FACE_LANDMARKS_TESSELATION not available:", e.message);
  }
  faceMeshConnections = buildFallbackConnections();
  console.log(`[FaceTracking] Using fallback connections (${faceMeshConnections.length} connections)`);
  return faceMeshConnections;
}

// =============================================================================
// PERSPECTIVE MAPPING — uses camera projection + MediaPipe Z depth
// =============================================================================
function mapToPerspectiveSpace(landmark) {
  // NDC (mirrored for selfie)
  const ndcX = (1.0 - landmark.x) * 2 - 1;
  const ndcY = -(landmark.y) * 2 + 1;

  // Camera frustum → world XY (equivalent to Vector3.unproject for ortho camera)
  const worldX = ndcX * (_camera.right - _camera.left) / 2;
  const worldY = ndcY * (_camera.top - _camera.bottom) / 2;

  // MediaPipe Z: negative = closer to camera. Map to Three.js world Z.
  const mpZ = landmark.z || 0;
  const worldZ = -mpZ * DEPTH_SCALE;

  return new THREE.Vector3(worldX, worldY, worldZ);
}

// =============================================================================
// FACE OCCLUDER — invisible depth mask from face oval (Z-buffer occlusion)
// =============================================================================
function setupOccluder() {
  // Geometry: triangle fan — center vertex + 36 oval vertices = 37 verts, 36 tris
  const vertCount = FACE_OVAL_INDICES.length + 1; // +1 for center
  const positions = new Float32Array(vertCount * 3);
  const indices = [];
  for (let i = 0; i < FACE_OVAL_INDICES.length; i++) {
    const next = (i + 1) % FACE_OVAL_INDICES.length;
    indices.push(0, i + 1, next + 1);
  }

  faceOccluderGeom = new THREE.BufferGeometry();
  faceOccluderGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  faceOccluderGeom.setIndex(indices);

  const mat = new THREE.MeshBasicMaterial({
    colorWrite: false,
    depthWrite: true,
    side: THREE.DoubleSide
  });

  faceOccluderMesh = new THREE.Mesh(faceOccluderGeom, mat);
  faceOccluderMesh.renderOrder = -3; // render before everything else
  faceOccluderMesh.visible = false;
  faceOccluderMesh.frustumCulled = false;
  _scene.add(faceOccluderMesh);

  console.log("[FaceTracking] Face occluder mesh created (37 verts, 36 tris).");
}

function updateOccluder(landmarks) {
  if (!faceOccluderMesh || !faceOccluderGeom) return;

  const posArr = faceOccluderGeom.attributes.position.array;

  // Vertex 0: center (nose bridge)
  const center = mapToPerspectiveSpace(landmarks[OVAL_CENTER_IDX]);
  posArr[0] = center.x; posArr[1] = center.y; posArr[2] = center.z;

  // Vertices 1–36: face oval
  for (let i = 0; i < FACE_OVAL_INDICES.length; i++) {
    const p = mapToPerspectiveSpace(landmarks[FACE_OVAL_INDICES[i]]);
    const off = (i + 1) * 3;
    posArr[off]     = p.x;
    posArr[off + 1] = p.y;
    posArr[off + 2] = p.z;
  }

  faceOccluderGeom.attributes.position.needsUpdate = true;
  faceOccluderGeom.computeBoundingSphere();
  faceOccluderMesh.visible = true;
}

// =============================================================================
// PENDULUM PHYSICS — head-relative swing with gravity blend
// =============================================================================
function computePendulumOrientation(headQuat, side, nowSec) {
  const state = pendulum[side];

  // Delta-time (clamped to avoid instability after pauses)
  const dt = lastPendulumTs > 0 ? Math.min(nowSec - lastPendulumTs, 0.05) : 0.016;

  // Decompose head rotation
  const headEuler = new THREE.Euler().setFromQuaternion(headQuat, 'YXZ');
  const targetSwing = headEuler.z; // head roll → earring swing target

  // Spring-damper: stiff follow with light swing overshoot
  const stiffness = 45.0;
  const damping = 8.0;
  const springForce = stiffness * (targetSwing - state.angle);
  const dampForce = -damping * state.velocity;
  state.velocity += (springForce + dampForce) * dt;
  state.angle += state.velocity * dt;

  // Build final quaternion: head yaw + partial pitch + pendulum roll
  const yawQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), headEuler.y);
  const pitchQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), headEuler.x * 0.5);
  const swingQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), state.angle);

  const result = new THREE.Quaternion();
  result.multiply(yawQ).multiply(pitchQ).multiply(swingQ);

  // Light gravity blend — cosmetic only, not a competing force
  const gravity = new THREE.Quaternion(); // identity
  result.slerp(gravity, 0.08);

  return result;
}

// =============================================================================
// PUBLIC API
// =============================================================================

export async function initializeFaceTracking(video, scene, camera, ctx, canvas, mapToOrthographicSpace) {
  _scene = scene;
  _ctx = ctx;
  _canvas = canvas;
  _camera = camera;
  _mapToOrtho = mapToOrthographicSpace; // keep for backward compat / pendant

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: true,
    numFaces: 1,
    runningMode: "VIDEO"
  });
  console.log("[FaceTracking] FaceLandmarker loaded.");

  // Setup face occluder mesh
  setupOccluder();

  await loadEarringModel('assets/Earring2.glb');
  console.log("[FaceTracking] Initialization complete.");
}

export function updateFaceTracking(videoElement, timestamp) {
  const tSec = timestamp / 1000; // seconds for filters

  // --- Detection: only on new video frames ---
  if (faceLandmarker && videoElement.readyState >= 2 && videoElement.currentTime !== lastFaceVideoTime) {
    lastFaceVideoTime = videoElement.currentTime;
    const results = faceLandmarker.detectForVideo(videoElement, timestamp);

    if (!results.faceLandmarks || results.faceLandmarks.length === 0) {
      isFacePresent = false;
      _lastLandmarks = null;
      _lastFaceBasis = null;
      _lastTransformMatrix = null;
      // Reset filters
      leftPosFilter.reset(); rightPosFilter.reset();
      scaleFilter.reset();
      pendulum.left.angle = 0; pendulum.left.velocity = 0;
      pendulum.right.angle = 0; pendulum.right.velocity = 0;
      lastPendulumTs = 0;
      if (leftEarring) leftEarring.visible = false;
      if (rightEarring) rightEarring.visible = false;
      if (faceOccluderMesh) faceOccluderMesh.visible = false;
      return;
    }

    isFacePresent = true;
    const landmarks = results.faceLandmarks[0];

    // --- FACE BASIS VECTORS ---
    const faceBasis = computeFaceBasis(landmarks);
    _lastLandmarks = landmarks;
    _lastFaceBasis = faceBasis;

    // --- EAR VISIBILITY (hide when ear is occluded by head turn) ---
    const noseTip = landmarks[1];
    const leftTragion = landmarks[234];
    const rightTragion = landmarks[454];
    const leftEarSpread = Math.abs(leftTragion.x - noseTip.x);
    const rightEarSpread = Math.abs(rightTragion.x - noseTip.x);
    const interEarX = Math.abs(leftTragion.x - rightTragion.x);
    const occlusionThreshold = interEarX * 0.15;
    leftEarVisible = leftEarSpread > occlusionThreshold;
    rightEarVisible = rightEarSpread > occlusionThreshold;

    // --- POSITION (One-Euro filtered) ---
    const { left, right } = computeEarlobePositions(landmarks, faceBasis);
    const filteredLeft = leftPosFilter.filter(left, tSec);
    const filteredRight = rightPosFilter.filter(right, tSec);
    targetLeftPos.copy(filteredLeft);
    targetRightPos.copy(filteredRight);

    // --- HEAD ORIENTATION → PENDULUM ---
    if (results.facialTransformationMatrixes?.length > 0) {
      _lastTransformMatrix = results.facialTransformationMatrixes[0];
      const headQuat = computeHeadOrientation(results.facialTransformationMatrixes[0]);

      const leftPendQuat = computePendulumOrientation(headQuat, 'left', tSec);
      const rightPendQuat = computePendulumOrientation(headQuat, 'right', tSec);
      lastPendulumTs = tSec;

      // Apply directly — pendulum spring-damper is itself a smoother
      targetLeftQuat.copy(leftPendQuat);
      targetRightQuat.copy(rightPendQuat);
    }

    // --- SCALE (One-Euro filtered) ---
    const leftEarWorld = mapToPerspectiveSpace(landmarks[234]);
    const rightEarWorld = mapToPerspectiveSpace(landmarks[454]);
    const interEarDist = leftEarWorld.distanceTo(rightEarWorld);
    const userScale = parseFloat(document.getElementById('earringScale')?.value || '1.0');
    const earZOff = parseFloat(document.getElementById('earringZOffset')?.value || '0');
    const earDepthScale = 1.0 + earZOff * 5.0;
    const rawScale = interEarDist * userScale * 0.15 * earDepthScale;
    const s = scaleFilter.filter(rawScale, tSec);
    targetEarringScale.set(s, s, s);

    // Apply user rotation to inner earring groups
    const earRotX = parseFloat(document.getElementById('earringRotX')?.value || '0') * Math.PI / 180;
    const earRotY = parseFloat(document.getElementById('earringRotY')?.value || '0') * Math.PI / 180;
    const earRotZ = parseFloat(document.getElementById('earringRotZ')?.value || '0') * Math.PI / 180;
    if (leftEarringInner) leftEarringInner.rotation.set(earRotX, earRotY, earRotZ);
    if (rightEarringInner) rightEarringInner.rotation.set(earRotX, earRotY, earRotZ);

    // Toggle axis helpers
    const showAxes = document.getElementById('showAxisHelpers')?.checked;
    [leftEarring, rightEarring].forEach(group => {
      if (!group) return;
      const axes = group.getObjectByName('earringAxes');
      if (axes) axes.visible = !!showAxes;
    });

    // --- UPDATE FACE OCCLUDER ---
    updateOccluder(landmarks);

    // --- FACE MESH DRAWING ---
    try {
      if (document.getElementById('showFaceMesh')?.checked) {
        drawFaceMesh(landmarks, faceBasis);
      }
    } catch (e) {
      console.warn("[FaceTracking] Face mesh draw error:", e.message);
    }
  }

  // --- Apply: every frame — DIRECT copy from filtered values (no extra lerp/slerp) ---
  if (leftEarring && rightEarring && isFacePresent) {
    if (leftEarVisible) {
      leftEarring.visible = true;
      leftEarring.position.copy(targetLeftPos);
      leftEarring.quaternion.copy(targetLeftQuat);
      leftEarring.scale.copy(targetEarringScale);
    } else {
      leftEarring.visible = false;
    }

    if (rightEarVisible) {
      rightEarring.visible = true;
      rightEarring.position.copy(targetRightPos);
      rightEarring.quaternion.copy(targetRightQuat);
      rightEarring.scale.copy(targetEarringScale);
    } else {
      rightEarring.visible = false;
    }
  }
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

function computeFaceBasis(landmarks) {
  // Build face-local coordinate frame in 3D world space (with depth)
  const leftEarW = mapToPerspectiveSpace(landmarks[234]);
  const rightEarW = mapToPerspectiveSpace(landmarks[454]);
  const faceRight = new THREE.Vector3().subVectors(rightEarW, leftEarW).normalize();

  const chinW = mapToPerspectiveSpace(landmarks[152]);
  const foreheadW = mapToPerspectiveSpace(landmarks[10]);
  const faceUp = new THREE.Vector3().subVectors(foreheadW, chinW).normalize();

  const faceForward = new THREE.Vector3().crossVectors(faceRight, faceUp).normalize();
  const faceUpOrtho = new THREE.Vector3().crossVectors(faceForward, faceRight).normalize();

  return { right: faceRight, up: faceUpOrtho, forward: faceForward };
}

function computeEarlobePositions(landmarks, faceBasis) {
  const leftTragion = landmarks[234];
  const rightTragion = landmarks[454];
  const leftJawAngle = landmarks[132];
  const rightJawAngle = landmarks[361];
  const noseTip = landmarks[1];
  const forehead = landmarks[10];
  const chin = landmarks[152];
  const leftCheekEdge = landmarks[177];
  const rightCheekEdge = landmarks[401];
  const leftPreauricular = landmarks[127];
  const rightPreauricular = landmarks[356];

  const faceHeight = Math.abs(forehead.y - chin.y);

  // Earlobe Y — target the bottom of the lobe, not mid-ear.
  // Tragion (234/454) sits at ear-canal height. Real piercing hole is ~13% of
  // face-height below that. Use tragion Y as base (more stable than jaw angle).
  const leftLobeY = leftTragion.y + faceHeight * 0.13;
  const rightLobeY = rightTragion.y + faceHeight * 0.13;

  // Earlobe X — lobe is inward of tragion, closer to jaw line.
  // Blend tragion (outer ear) with jaw angle (inner) — lobe sits between them.
  const leftBaseX = leftTragion.x * 0.4 + leftJawAngle.x * 0.6;
  const rightBaseX = rightTragion.x * 0.4 + rightJawAngle.x * 0.6;
  // Minimal outward push (lobe barely protrudes past jaw)
  const leftOutwardPush = (leftBaseX - noseTip.x) * 0.04;
  const rightOutwardPush = (rightBaseX - noseTip.x) * 0.04;
  const leftLobeX = leftBaseX + leftOutwardPush;
  const rightLobeX = rightBaseX + rightOutwardPush;

  // Earlobe Z — use tragion Z (3D depth from MediaPipe)
  const leftLobeNorm  = { x: leftLobeX,  y: leftLobeY,  z: leftTragion.z };
  const rightLobeNorm = { x: rightLobeX, y: rightLobeY, z: rightTragion.z };

  // Convert to 3D world space with depth
  const leftWorld  = mapToPerspectiveSpace(leftLobeNorm);
  const rightWorld = mapToPerspectiveSpace(rightLobeNorm);

  // --- PROPORTIONAL OUTWARD (FORWARD) PUSH ---
  // Push earring forward along face-forward vector so it sits in front of the
  // face mesh / occluder and doesn't clip into skin during profile turns.
  const leftEarW  = mapToPerspectiveSpace(landmarks[234]);
  const rightEarW = mapToPerspectiveSpace(landmarks[454]);
  const faceSize  = leftEarW.distanceTo(rightEarW);
  const forwardPush = faceSize * 0.1; // proportional to face size
  leftWorld.addScaledVector(faceBasis.forward, forwardPush);
  rightWorld.addScaledVector(faceBasis.forward, forwardPush);

  // User fine-tune offsets (face-local space)
  const yOff = parseFloat(document.getElementById('earringYOffset')?.value || '0');
  const xOff = parseFloat(document.getElementById('earringXOffset')?.value || '0');
  const zOff = parseFloat(document.getElementById('earringZOffset')?.value || '0');
  const offsetScale = faceSize * 5;

  const leftOffset = new THREE.Vector3()
    .addScaledVector(faceBasis.right, -xOff * offsetScale)
    .addScaledVector(faceBasis.up, -yOff * offsetScale)
    .addScaledVector(faceBasis.forward, zOff * offsetScale);
  leftWorld.add(leftOffset);

  const rightOffset = new THREE.Vector3()
    .addScaledVector(faceBasis.right, xOff * offsetScale)
    .addScaledVector(faceBasis.up, -yOff * offsetScale)
    .addScaledVector(faceBasis.forward, zOff * offsetScale);
  rightWorld.add(rightOffset);

  return { left: leftWorld, right: rightWorld };
}

export function computeHeadOrientation(matrix) {
  if (!matrix?.data) return new THREE.Quaternion();

  const mat4 = new THREE.Matrix4().fromArray(matrix.data);
  const quat = new THREE.Quaternion();
  const pos = new THREE.Vector3();
  const scale = new THREE.Vector3();
  mat4.decompose(pos, quat, scale);

  // Coordinate conversion: MediaPipe → Three.js mirrored CSS
  quat.y = -quat.y;
  quat.z = -quat.z;

  return quat;
}

// Permanently remove triangles below a Y threshold (world space).
// Hides whole meshes that are entirely below, crops individual triangles for
// meshes that span the boundary. Works at any position/rotation after load.
function cropGeometryBelow(group, clipY) {
  group.updateMatrixWorld(true);
  const v = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];

  group.traverse(node => {
    if (!node.isMesh || !node.geometry) return;
    const geom = node.geometry;
    const pos = geom.attributes.position;
    if (!pos) return;

    // If entire mesh is below clipY → hide it
    const meshBox = new THREE.Box3().setFromObject(node);
    if (meshBox.max.y <= clipY + 0.001) {
      node.visible = false;
      console.log(`[Earring] Hidden mesh "${node.name || 'unnamed'}" (below clip)`);
      return;
    }

    // If mesh spans the boundary → crop individual triangles
    if (meshBox.min.y < clipY - 0.001 && geom.index) {
      const oldIdx = Array.from(geom.index.array);
      const newIdx = [];
      for (let i = 0; i < oldIdx.length; i += 3) {
        let anyAbove = false;
        for (let j = 0; j < 3; j++) {
          v[j].fromBufferAttribute(pos, oldIdx[i + j]);
          node.localToWorld(v[j]);
          if (v[j].y >= clipY) { anyAbove = true; break; }
        }
        if (anyAbove) newIdx.push(oldIdx[i], oldIdx[i + 1], oldIdx[i + 2]);
      }
      const removed = (oldIdx.length - newIdx.length) / 3;
      if (removed > 0) {
        geom.setIndex(newIdx);
        geom.computeBoundingBox();
        geom.computeBoundingSphere();
        console.log(`[Earring] Cropped ${removed} tris from "${node.name || 'unnamed'}"`);
      }
    }
  });
}

function loadEarringModel(modelPath) {
  return new Promise((resolve) => {
    gltfLoader.load(modelPath, (gltf) => {
      const raw = gltf.scene;

      // ---- Step 1: Normalize to unit size ----
      const box = new THREE.Box3().setFromObject(raw);
      const size = new THREE.Vector3();
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      if (maxDim > 0) raw.scale.multiplyScalar(1.0 / maxDim);

      // ---- Step 2: No auto-orient ----
      // GLB native orientation + user Rot sliders handle orientation.
      // Default Rot values are tuned in index.html.
      raw.updateMatrixWorld(true);

      // ---- Step 3: Permanently crop the post/backing ----
      // The decorative part is the UPPER portion of the model (positive Y).
      // The post/backing is the LOWER portion (negative Y from center).
      // Permanently delete geometry below the clip boundary.
      const sBox = new THREE.Box3().setFromObject(raw);
      const center = new THREE.Vector3();
      sBox.getCenter(center);
      const modelHeight = sBox.max.y - sBox.min.y;
      // Clip at 50% (center) — the tulip/decorative part is above, post is below
      const clipY = sBox.min.y + modelHeight * 0.50;

      console.log(`[Earring] height=${modelHeight.toFixed(3)}, clip at Y=${clipY.toFixed(3)}`);
      cropGeometryBelow(raw, clipY);

      // ---- Step 4: Pivot at clip boundary ----
      // The clip boundary = earlobe attachment. Decorative part hangs below.
      raw.position.set(-center.x, -clipY, -center.z);

      // ---- Step 4.5: PBR shading (HDR IBL via scene.environment) ----
      applyJewelryShading(raw, {
        metal:   { metalness: 1.0, roughness: 0.05, envMapIntensity: 1.8, clearcoat: 0.7, clearcoatRoughness: 0.04 },
        diamond: {
          transmission: 1.0, thickness: 0.5, ior: 2.417, roughness: 0.0,
          envMapIntensity: 3.5, attenuationDistance: 2.5,
          clearcoat: 1.0, clearcoatRoughness: 0.0, dispersion: 0.010,
          sparkleStrength: 0.85, fringeStrength: 0.07,
          // CubeCamera is anchored to the ring's position in main.js — sampling
          // it from the earring (on the ear) gives wrong-perspective reflections
          // that read as dark patches. Sample HDR instead for a clean studio
          // look that matches the ring visually.
          useCubeReflection: false
        }
      });

      // ---- Step 5: Build hierarchy ----
      leftEarringInner = new THREE.Group();
      leftEarringInner.add(raw);
      leftEarring = new THREE.Group();
      leftEarring.add(leftEarringInner);
      leftEarring.visible = false;

      // Right earring — mirrored clone
      const rawClone = raw.clone(true);
      rawClone.scale.x *= -1;
      rawClone.traverse(n => {
        if (n.isMesh && n.material) {
          n.material = n.material.clone();
          n.material.side = THREE.DoubleSide;
        }
      });
      rightEarringInner = new THREE.Group();
      rightEarringInner.add(rawClone);
      rightEarring = new THREE.Group();
      rightEarring.add(rightEarringInner);
      rightEarring.visible = false;

      // Debug axes helpers
      [leftEarring, rightEarring].forEach(group => {
        const axH = new THREE.AxesHelper(0.5);
        axH.renderOrder = 999;
        axH.name = 'earringAxes';
        axH.traverse(n => {
          if (n.material) {
            n.material.depthTest = false;
            n.material.depthWrite = false;
          }
        });
        group.add(axH);
      });

      _scene.add(leftEarring);
      _scene.add(rightEarring);
      console.log(`[FaceTracking] Earring loaded (geometry crop, no auto-orient). Size:`, size);
      resolve();
    });
  });
}

function drawFaceMesh(landmarks, faceBasis) {
  const w = _canvas.width;
  const h = _canvas.height;
  const connections = getFaceMeshConnections();

  _ctx.strokeStyle = 'rgba(255, 165, 0, 0.5)';
  _ctx.lineWidth = 1;
  _ctx.beginPath();
  for (const { start, end } of connections) {
    if (start >= landmarks.length || end >= landmarks.length) continue;
    const s = landmarks[start], e = landmarks[end];
    _ctx.moveTo((1 - s.x) * w, s.y * h);
    _ctx.lineTo((1 - e.x) * w, e.y * h);
  }
  _ctx.stroke();

  _ctx.fillStyle = 'rgba(255, 200, 50, 0.7)';
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    _ctx.beginPath();
    _ctx.arc((1 - lm.x) * w, lm.y * h, 2, 0, Math.PI * 2);
    _ctx.fill();
  }

  _ctx.fillStyle = 'red';
  for (const idx of [234, 454, 132, 361, 1, 152, 10]) {
    if (idx < landmarks.length) {
      const lm = landmarks[idx];
      _ctx.beginPath();
      _ctx.arc((1 - lm.x) * w, lm.y * h, 5, 0, Math.PI * 2);
      _ctx.fill();
    }
  }

  _ctx.font = '10px monospace';
  _ctx.fillStyle = 'white';
  const labelMap = { 234: 'L-trag', 454: 'R-trag', 132: 'L-jaw', 361: 'R-jaw' };
  for (const [idx, label] of Object.entries(labelMap)) {
    const lm = landmarks[parseInt(idx)];
    if (lm) _ctx.fillText(label, (1 - lm.x) * w + 7, lm.y * h + 3);
  }

  // Estimated earlobe positions (debug) — MUST match computeEarlobePositions formula
  const noseTip = landmarks[1];
  const forehead = landmarks[10];
  const chin = landmarks[152];
  const faceHeight = Math.abs(forehead.y - chin.y);
  const leftTragion = landmarks[234];
  const rightTragion = landmarks[454];
  const leftJawAngle = landmarks[132];
  const rightJawAngle = landmarks[361];

  const leftLobeY = leftTragion.y + faceHeight * 0.13;
  const rightLobeY = rightTragion.y + faceHeight * 0.13;
  const leftDbgBaseX = leftTragion.x * 0.4 + leftJawAngle.x * 0.6;
  const rightDbgBaseX = rightTragion.x * 0.4 + rightJawAngle.x * 0.6;
  const leftLobeX = leftDbgBaseX + (leftDbgBaseX - noseTip.x) * 0.04;
  const rightLobeX = rightDbgBaseX + (rightDbgBaseX - noseTip.x) * 0.04;

  _ctx.fillStyle = leftEarVisible ? 'lime' : 'rgba(128,128,128,0.5)';
  _ctx.beginPath();
  _ctx.arc((1 - leftLobeX) * w, leftLobeY * h, 7, 0, Math.PI * 2);
  _ctx.fill();

  _ctx.fillStyle = rightEarVisible ? 'lime' : 'rgba(128,128,128,0.5)';
  _ctx.beginPath();
  _ctx.arc((1 - rightLobeX) * w, rightLobeY * h, 7, 0, Math.PI * 2);
  _ctx.fill();
}

// --- Shared face results for other modules ---
export function getLatestFaceResults() {
  return {
    landmarks: _lastLandmarks,
    faceBasis: _lastFaceBasis,
    transformMatrix: _lastTransformMatrix,
    isFacePresent
  };
}
