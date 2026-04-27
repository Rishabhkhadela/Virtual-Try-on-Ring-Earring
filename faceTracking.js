import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
import { applyJewelryShading, setMetalColor } from './Shader.js';

const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
// BVH ray-trace always on — it's what gives earring diamonds the internal
// facet pattern. Mobile uses 3 bounces instead of 5 (subtle quality loss,
// meaningful perf gain). Mirrors main.js PERF_TIER.
const DIAMOND_USE_BVH = true;
const DIAMOND_BOUNCES = 5;
const DIAMOND_FILL_AMOUNT = 0;
const DIAMOND_SPARKLE_STRENGTH = 0.85;
const DIAMOND_FRINGE_STRENGTH = 0.07;

// =============================================================================
// ONE-EURO FILTER
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

export class OneEuroFilterVec3 {
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

export class OneEuroFilterQuat {
  constructor(freq, minCutoff, beta, dCutoff) {
    this.fx = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.fy = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.fz = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.fw = new OneEuroFilter(freq, minCutoff, beta, dCutoff);
    this.prev = new THREE.Quaternion();
  }
  filter(q, t) {
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

export class OneEuroFilterScalar {
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
let _vision = null;
let _currentDelegate = 'GPU';
let _consecutiveMisses = 0;
let _hadDetectionOnce = false;
let _recovering = false;

let leftEarring = null;
let rightEarring = null;
let leftEarringInner = null;
let rightEarringInner = null;
let lastFaceVideoTime = -1;
let isFacePresent = false;
let leftEarVisible = false;
let rightEarVisible = false;
let faceMeshConnections = null;

let _faceTrackingActive = false;
let _lastLandmarks = null;
let _lastFaceBasis = null;

const targetLeftPos = new THREE.Vector3();
const targetRightPos = new THREE.Vector3();
const targetLeftQuat = new THREE.Quaternion();
const targetRightQuat = new THREE.Quaternion();
const targetEarringScale = new THREE.Vector3(1, 1, 1);

const FILT_FREQ = 30;
const leftPosFilter  = new OneEuroFilterVec3(FILT_FREQ, 1.0, 0.55, 1.0);
const rightPosFilter = new OneEuroFilterVec3(FILT_FREQ, 1.0, 0.55, 1.0);
const scaleFilter = new OneEuroFilterScalar(FILT_FREQ, 1.0, 0.1, 1.0);

const pendulum = {
  left:  { angle: 0, velocity: 0 },
  right: { angle: 0, velocity: 0 }
};
let lastPendulumTs = 0;

let faceOccluderMesh = null;
let faceOccluderGeom = null;
const DEPTH_SCALE = 3.0;

const FACE_OVAL_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
];
const OVAL_CENTER_IDX = 4;

let _scene, _ctx, _canvas, _camera, _mapToOrtho;
const gltfLoader = new THREE.GLTFLoader();

// =============================================================================
// FACE MESH CONNECTIONS
// =============================================================================
const FACE_OVAL = [[10,338],[338,297],[297,332],[332,284],[284,251],[251,389],[389,356],[356,454],[454,323],[323,361],[361,288],[288,397],[397,365],[365,379],[379,378],[378,400],[400,377],[377,152],[152,148],[148,176],[176,149],[149,150],[150,136],[136,172],[172,58],[58,132],[132,93],[93,234],[234,127],[127,162],[162,21],[21,54],[54,103],[103,67],[67,109],[109,10]];
const LEFT_EYE = [[263,249],[249,390],[390,373],[373,374],[374,380],[380,381],[381,382],[382,362],[362,263],[263,466],[466,388],[388,387],[387,386],[386,385],[385,384],[384,398]];
const RIGHT_EYE = [[33,7],[7,163],[163,144],[144,145],[145,153],[153,154],[154,155],[155,133],[133,33],[33,246],[246,161],[161,160],[160,159],[159,158],[158,157],[157,173]];
const LIPS = [[61,146],[146,91],[91,181],[181,84],[84,17],[17,314],[314,405],[405,321],[321,375],[375,291],[291,409],[409,270],[270,269],[269,267],[267,0],[0,37],[37,39],[39,40],[40,185],[185,61]];
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
      return faceMeshConnections;
    }
  } catch (e) {}
  faceMeshConnections = buildFallbackConnections();
  return faceMeshConnections;
}

// =============================================================================
// PERSPECTIVE MAPPING
// =============================================================================
function mapToPerspectiveSpace(landmark) {
  if (!landmark || !Number.isFinite(landmark.x) || !Number.isFinite(landmark.y)) {
    return new THREE.Vector3(0, 0, 0);
  }
  const ndcX = (1.0 - landmark.x) * 2 - 1;
  const ndcY = -(landmark.y) * 2 + 1;
  const frustumW = _camera ? (_camera.right - _camera.left) : 2;
  const frustumH = _camera ? (_camera.top - _camera.bottom) : 2;
  const worldX = ndcX * frustumW / 2;
  const worldY = ndcY * frustumH / 2;
  const mpZ = Number.isFinite(landmark.z) ? landmark.z : 0;
  const worldZ = -mpZ * DEPTH_SCALE;
  return new THREE.Vector3(worldX, worldY, worldZ);
}

// =============================================================================
// FACE OCCLUDER
// =============================================================================
function setupOccluder() {
  const vertCount = FACE_OVAL_INDICES.length + 1;
  const positions = new Float32Array(vertCount * 3);
  const indices = [];
  for (let i = 0; i < FACE_OVAL_INDICES.length; i++) {
    const next = (i + 1) % FACE_OVAL_INDICES.length;
    indices.push(0, i + 1, next + 1);
  }
  faceOccluderGeom = new THREE.BufferGeometry();
  faceOccluderGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  faceOccluderGeom.setIndex(indices);
  const mat = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true, side: THREE.DoubleSide });
  faceOccluderMesh = new THREE.Mesh(faceOccluderGeom, mat);
  faceOccluderMesh.renderOrder = -3;
  faceOccluderMesh.visible = false;
  faceOccluderMesh.frustumCulled = false;
  _scene.add(faceOccluderMesh);
}

function updateOccluder(landmarks) {
  if (!faceOccluderMesh || !faceOccluderGeom) return;
  const posArr = faceOccluderGeom.attributes.position.array;
  const center = mapToPerspectiveSpace(landmarks[OVAL_CENTER_IDX]);
  if (!Number.isFinite(center.x)) { faceOccluderMesh.visible = false; return; }
  posArr[0] = center.x; posArr[1] = center.y; posArr[2] = center.z;
  for (let i = 0; i < FACE_OVAL_INDICES.length; i++) {
    const p = mapToPerspectiveSpace(landmarks[FACE_OVAL_INDICES[i]]);
    const off = (i + 1) * 3;
    posArr[off]     = Number.isFinite(p.x) ? p.x : 0;
    posArr[off + 1] = Number.isFinite(p.y) ? p.y : 0;
    posArr[off + 2] = Number.isFinite(p.z) ? p.z : 0;
  }
  faceOccluderGeom.attributes.position.needsUpdate = true;
  faceOccluderGeom.computeBoundingSphere();
  faceOccluderMesh.visible = true;
}

// =============================================================================
// PENDULUM PHYSICS
// =============================================================================
function computePendulumOrientation(headQuat, side, nowSec) {
  const state = pendulum[side];
  const dt = lastPendulumTs > 0 ? Math.min(nowSec - lastPendulumTs, 0.05) : 0.016;
  const headEuler = new THREE.Euler().setFromQuaternion(headQuat, 'YXZ');
  const targetSwing = headEuler.z;
  const stiffness = 45.0, damping = 8.0;
  const springForce = stiffness * (targetSwing - state.angle);
  const dampForce = -damping * state.velocity;
  state.velocity += (springForce + dampForce) * dt;
  state.angle += state.velocity * dt;
  const yawQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), headEuler.y);
  const pitchQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), headEuler.x * 0.5);
  const swingQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), state.angle);
  const result = new THREE.Quaternion();
  result.multiply(yawQ).multiply(pitchQ).multiply(swingQ);
  const gravity = new THREE.Quaternion();
  result.slerp(gravity, 0.08);
  return result;
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================
function computeFaceBasis(landmarks) {
  const leftEarW = mapToPerspectiveSpace(landmarks[234]);
  const rightEarW = mapToPerspectiveSpace(landmarks[454]);
  const rawRight = new THREE.Vector3().subVectors(rightEarW, leftEarW);
  const right = rawRight.lengthSq() > 1e-12 ? rawRight.normalize() : new THREE.Vector3(1, 0, 0);

  const chinW = mapToPerspectiveSpace(landmarks[152]);
  const foreheadW = mapToPerspectiveSpace(landmarks[10]);
  const rawUp = new THREE.Vector3().subVectors(foreheadW, chinW);
  const up = rawUp.lengthSq() > 1e-12 ? rawUp.normalize() : new THREE.Vector3(0, 1, 0);

  const rawFwd = new THREE.Vector3().crossVectors(right, up);
  const forward = rawFwd.lengthSq() > 1e-12 ? rawFwd.normalize() : new THREE.Vector3(0, 0, 1);
  const upOrtho = new THREE.Vector3().crossVectors(forward, right).normalize();

  return { right, up: upOrtho, forward };
}

function quatFromBasis(basis) {
  const m = new THREE.Matrix4().makeBasis(basis.right, basis.up, basis.forward);
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

function computeEarlobePositions(landmarks, faceBasis) {
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
  const leftBaseX = leftTragion.x * 0.4 + leftJawAngle.x * 0.6;
  const rightBaseX = rightTragion.x * 0.4 + rightJawAngle.x * 0.6;
  const leftOutwardPush = (leftBaseX - noseTip.x) * 0.04;
  const rightOutwardPush = (rightBaseX - noseTip.x) * 0.04;
  const leftLobeX = leftBaseX + leftOutwardPush;
  const rightLobeX = rightBaseX + rightOutwardPush;

  const leftLobeNorm  = { x: leftLobeX,  y: leftLobeY,  z: leftTragion.z };
  const rightLobeNorm = { x: rightLobeX, y: rightLobeY, z: rightTragion.z };
  const leftWorld  = mapToPerspectiveSpace(leftLobeNorm);
  const rightWorld = mapToPerspectiveSpace(rightLobeNorm);

  const leftEarW  = mapToPerspectiveSpace(landmarks[234]);
  const rightEarW = mapToPerspectiveSpace(landmarks[454]);
  const faceSize  = leftEarW.distanceTo(rightEarW);
  const forwardPush = faceSize * 0.1;
  leftWorld.addScaledVector(faceBasis.forward, forwardPush);
  rightWorld.addScaledVector(faceBasis.forward, forwardPush);

  const yOff = parseFloat(document.getElementById('earringYOffset')?.value || '0');
  const xOff = parseFloat(document.getElementById('earringXOffset')?.value || '0');
  const zOff = parseFloat(document.getElementById('earringZOffset')?.value || '0');
  const offsetScale = faceSize * 5;
  leftWorld.addScaledVector(faceBasis.right, -xOff * offsetScale).addScaledVector(faceBasis.up, -yOff * offsetScale).addScaledVector(faceBasis.forward, zOff * offsetScale);
  rightWorld.addScaledVector(faceBasis.right, xOff * offsetScale).addScaledVector(faceBasis.up, -yOff * offsetScale).addScaledVector(faceBasis.forward, zOff * offsetScale);
  return { left: leftWorld, right: rightWorld };
}

// =============================================================================
// PUBLIC API
// =============================================================================
async function createFaceLandmarker(delegate) {
  return await FaceLandmarker.createFromOptions(_vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: delegate
    },
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
    numFaces: 1,
    runningMode: "VIDEO"
  });
}

async function recoverFaceToCpu() {
  if (_recovering || _currentDelegate === 'CPU') return;
  _recovering = true;
  window.__forceAllCpu = true;
  console.warn("[FaceTracking] GPU stuck — recreating on CPU.");
  try { faceLandmarker.close(); } catch (e) {}
  faceLandmarker = null;
  try {
    faceLandmarker = await createFaceLandmarker('CPU');
    _currentDelegate = 'CPU';
    _consecutiveMisses = 0;
    _hadDetectionOnce = false;
    lastFaceVideoTime = -1;
    console.log("[FaceTracking] Recovered on CPU.");
  } catch (e) {
    console.error("[FaceTracking] CPU recreate failed:", e);
  } finally {
    _recovering = false;
  }
}

export async function initializeFaceTracking(video, scene, camera, ctx, canvas, mapToOrtho) {
  _scene = scene; _ctx = ctx; _canvas = canvas; _camera = camera; _mapToOrtho = mapToOrtho;
  _vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
  faceLandmarker = await createFaceLandmarker('GPU');
  setupOccluder();
  await loadEarringModel('assets/Earring2.glb');
}

export async function setEarringModel(path) {
  if (!_scene) return;
  if (leftEarring) { _scene.remove(leftEarring); leftEarring = null; }
  if (rightEarring) { _scene.remove(rightEarring); rightEarring = null; }
  leftEarringInner = null; rightEarringInner = null;
  leftPosFilter.reset(); rightPosFilter.reset(); scaleFilter.reset();
  pendulum.left.angle = 0; pendulum.left.velocity = 0; pendulum.right.angle = 0; pendulum.right.velocity = 0;
  lastPendulumTs = 0;
  await loadEarringModel(path);
}

export function updateFaceTracking(videoElement, timestamp) {
  if (!_faceTrackingActive || !faceLandmarker) return;
  const tSec = timestamp / 1000;
  if (videoElement.readyState >= 2 && videoElement.currentTime !== lastFaceVideoTime) {
    lastFaceVideoTime = videoElement.currentTime;
    let results = null; let threw = false;
    try { results = faceLandmarker.detectForVideo(videoElement, timestamp); }
    catch (e) { threw = true; console.warn("[FaceTracking] detectForVideo threw:", e?.message || e); }

    const nDetections = results?.faceLandmarks?.length || 0;
    if (nDetections > 0) { _hadDetectionOnce = true; _consecutiveMisses = 0; }
    else { _consecutiveMisses++; }
    if (threw) _consecutiveMisses++;

    if (_currentDelegate === 'GPU' && !_recovering && (window.__forceAllCpu || (_hadDetectionOnce && _consecutiveMisses > 20) || (!_hadDetectionOnce && _consecutiveMisses > 60))) {
      recoverFaceToCpu();
    }
    if (threw || !results || nDetections === 0) {
      isFacePresent = false; _lastLandmarks = null; _lastFaceBasis = null;
      if (leftEarring) leftEarring.visible = false;
      if (rightEarring) rightEarring.visible = false;
      if (faceOccluderMesh) faceOccluderMesh.visible = false;
      return;
    }
    isFacePresent = true;
    const landmarks = results.faceLandmarks[0];
    const faceBasis = computeFaceBasis(landmarks);
    _lastLandmarks = landmarks; _lastFaceBasis = faceBasis;

    const noseTip = landmarks[1], leftTragion = landmarks[234], rightTragion = landmarks[454];
    const leftEarSpread = Math.abs(leftTragion.x - noseTip.x), rightEarSpread = Math.abs(rightTragion.x - noseTip.x);
    const interEarX = Math.abs(leftTragion.x - rightTragion.x), occlusionThreshold = interEarX * 0.15;
    leftEarVisible = leftEarSpread > occlusionThreshold; rightEarVisible = rightEarSpread > occlusionThreshold;

    const { left, right } = computeEarlobePositions(landmarks, faceBasis);
    targetLeftPos.copy(leftPosFilter.filter(left, tSec));
    targetRightPos.copy(rightPosFilter.filter(right, tSec));

    const headQuat = quatFromBasis(faceBasis);
    targetLeftQuat.copy(computePendulumOrientation(headQuat, 'left', tSec));
    targetRightQuat.copy(computePendulumOrientation(headQuat, 'right', tSec));
    lastPendulumTs = tSec;

    const leftEarWorld = mapToPerspectiveSpace(landmarks[234]), rightEarWorld = mapToPerspectiveSpace(landmarks[454]);
    const interEarDist = leftEarWorld.distanceTo(rightEarWorld);
    const userScale = parseFloat(document.getElementById('earringScale')?.value || '1.0');
    const earZOff = parseFloat(document.getElementById('earringZOffset')?.value || '0');
    const rawScale = interEarDist * userScale * 0.15 * (1.0 + earZOff * 5.0);
    const s = scaleFilter.filter(rawScale, tSec);
    targetEarringScale.set(s, s, s);

    const earRotX = parseFloat(document.getElementById('earringRotX')?.value || '0') * Math.PI / 180;
    const earRotY = parseFloat(document.getElementById('earringRotY')?.value || '0') * Math.PI / 180;
    const earRotZ = parseFloat(document.getElementById('earringRotZ')?.value || '0') * Math.PI / 180;
    if (leftEarringInner) leftEarringInner.rotation.set(earRotX, earRotY, earRotZ);
    if (rightEarringInner) rightEarringInner.rotation.set(earRotX, earRotY, earRotZ);

    updateOccluder(landmarks);
    if (document.getElementById('showFaceMesh')?.checked) drawFaceMesh(landmarks, faceBasis);
  }

  if (leftEarring && rightEarring && isFacePresent) {
    leftEarring.visible = leftEarVisible;
    if (leftEarVisible) { leftEarring.position.copy(targetLeftPos); leftEarring.quaternion.copy(targetLeftQuat); leftEarring.scale.copy(targetEarringScale); }
    rightEarring.visible = rightEarVisible;
    if (rightEarVisible) { rightEarring.position.copy(targetRightPos); rightEarring.quaternion.copy(targetRightQuat); rightEarring.scale.copy(targetEarringScale); }
  }
}

// =============================================================================
// MODEL LOADING
// =============================================================================
function cropGeometryBelow(group, clipY) {
  group.updateMatrixWorld(true);
  const v = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
  group.traverse(node => {
    if (!node.isMesh || !node.geometry) return;
    const geom = node.geometry, pos = geom.attributes.position;
    if (!pos) return;
    const meshBox = new THREE.Box3().setFromObject(node);
    if (meshBox.max.y <= clipY + 0.001) { node.visible = false; return; }
    if (meshBox.min.y < clipY - 0.001 && geom.index) {
      const oldIdx = Array.from(geom.index.array), newIdx = [];
      for (let i = 0; i < oldIdx.length; i += 3) {
        let anyAbove = false;
        for (let j = 0; j < 3; j++) {
          v[j].fromBufferAttribute(pos, oldIdx[i + j]); node.localToWorld(v[j]);
          if (v[j].y >= clipY) { anyAbove = true; break; }
        }
        if (anyAbove) newIdx.push(oldIdx[i], oldIdx[i+1], oldIdx[i+2]);
      }
      if (oldIdx.length !== newIdx.length) geom.setIndex(newIdx);
    }
  });
}

function loadEarringModel(modelPath) {
  return new Promise(resolve => {
    gltfLoader.load(modelPath, gltf => {
      const raw = gltf.scene;
      const box = new THREE.Box3().setFromObject(raw), size = new THREE.Vector3(); box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z); if (maxDim > 0) raw.scale.multiplyScalar(1.0 / maxDim);
      raw.updateMatrixWorld(true);
      const sBox = new THREE.Box3().setFromObject(raw), center = new THREE.Vector3(); sBox.getCenter(center);
      const clipY = sBox.min.y + (sBox.max.y - sBox.min.y) * 0.50;
      cropGeometryBelow(raw, clipY);
      raw.position.set(-center.x, -clipY, -center.z);
      applyJewelryShading(raw, {
        metal: {
          // Mobile parity boost — see main.js ring block. Emissive lift
          // keeps thin earring geometry visible at mobile's lower pixel
          // ratio. Tinted to the metal color so gold stays gold.
          metalness: 1.0,
          roughness: 0.05,
          envMapIntensity: 1.8,
          clearcoat: 0.7,
          clearcoatRoughness: 0.04,
          emissiveIntensity: 0,
          color: _currentEarringMetalPreset
        },
        diamond: { transmission: 1.0, thickness: 0.5, ior: 2.417, metalness: 0.0, roughness: 0.0, clearcoat: 0.0, clearcoatRoughness: 0.0, envMapIntensity: 3.5, attenuationDistance: 2.5, dispersion: 0.010, sparkleStrength: DIAMOND_SPARKLE_STRENGTH, fringeStrength: DIAMOND_FRINGE_STRENGTH, useCubeReflection: false, useBVH: DIAMOND_USE_BVH, bounces: DIAMOND_BOUNCES, fillAmount: DIAMOND_FILL_AMOUNT }
      });
      leftEarringInner = new THREE.Group(); leftEarringInner.add(raw);
      leftEarring = new THREE.Group(); leftEarring.add(leftEarringInner); leftEarring.visible = false;
      const rawClone = raw.clone(true);
      rightEarringInner = new THREE.Group(); rightEarringInner.add(rawClone);
      rightEarring = new THREE.Group(); rightEarring.add(rightEarringInner); rightEarring.visible = false;
      [leftEarring, rightEarring].forEach(group => {
        const axH = new THREE.AxesHelper(0.5); axH.renderOrder = 999; axH.name = 'earringAxes';
        axH.traverse(n => { if (n.material) { n.material.depthTest = false; n.material.depthWrite = false; } });
        group.add(axH);
      });
      _scene.add(leftEarring); _scene.add(rightEarring); resolve();
    });
  });
}

function drawFaceMesh(landmarks, faceBasis) {
  const w = _canvas.width, h = _canvas.height, connections = getFaceMeshConnections();
  _ctx.strokeStyle = 'rgba(255, 165, 0, 0.5)'; _ctx.lineWidth = 1; _ctx.beginPath();
  for (const { start, end } of connections) {
    if (start >= landmarks.length || end >= landmarks.length) continue;
    _ctx.moveTo((1 - landmarks[start].x) * w, landmarks[start].y * h);
    _ctx.lineTo((1 - landmarks[end].x) * w, landmarks[end].y * h);
  }
  _ctx.stroke();
  _ctx.fillStyle = 'red';
  for (const idx of [234, 454, 132, 361, 1, 152, 10]) {
    const lm = landmarks[idx];
    if (lm) { _ctx.beginPath(); _ctx.arc((1 - lm.x) * w, lm.y * h, 5, 0, Math.PI * 2); _ctx.fill(); }
  }
}

export function setEarringMetalColor(preset) {
  if (leftEarringInner) setMetalColor(leftEarringInner, preset);
  if (rightEarringInner) setMetalColor(rightEarringInner, preset);
}

export function isFaceTrackingActive() { return _faceTrackingActive; }
export function activateFaceTracking() { _faceTrackingActive = true; lastFaceVideoTime = -1; }
export function deactivateFaceTracking() {
  _faceTrackingActive = false; isFacePresent = false; _lastLandmarks = null; _lastFaceBasis = null;
  if (leftEarring) leftEarring.visible = false; if (rightEarring) rightEarring.visible = false; if (faceOccluderMesh) faceOccluderMesh.visible = false;
  leftPosFilter.reset(); rightPosFilter.reset(); scaleFilter.reset();
  pendulum.left.angle = 0; pendulum.left.velocity = 0; pendulum.right.angle = 0; pendulum.right.velocity = 0; lastPendulumTs = 0;
}
let _currentEarringMetalPreset = null;
export function setEarringMetalColorPreset(preset) { _currentEarringMetalPreset = preset; }
export function getEarringMetalColorPreset() { return _currentEarringMetalPreset; }
export function getLatestFaceResults() { return { landmarks: _lastLandmarks, faceBasis: _lastFaceBasis, isFacePresent }; }
