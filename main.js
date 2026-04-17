import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
import { initializeFaceTracking, updateFaceTracking } from './faceTracking.js';
import { initializePendantTracking, updatePendantTracking } from './pendantTracking.js';
import { initializeWristTracking, updateWristTracking } from './wristTracking.js';

// --- DOM Elements ---
const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const landmarkCanvas = document.getElementById('landmark_canvas');
const landmarkCtx = landmarkCanvas.getContext('2d');
const loadingElement = document.getElementById('loading');
const containerElement = document.querySelector('.mirror-container');
const activeFingerLabel = document.getElementById('activeFingerLabel');
const cycleFingerButton = document.getElementById('cycleFingerBtn');

// --- Three.js Setup (Orthographic for Zero Drift) ---
const scene = new THREE.Scene();
// Frustum will be resized to match pixels perfectly in checkCanvasSize
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({
  canvas: canvasElement,
  alpha: true,
  antialias: true,
  depth: true       // explicit depth buffer for face occluder Z-clipping
});
renderer.sortObjects = true; // respect renderOrder for occluder-before-jewelry
renderer.localClippingEnabled = true; // allow per-material clipping planes (earring post clip)

scene.add(new THREE.AmbientLight(0xffffff, 1.5));
const dl1 = new THREE.DirectionalLight(0xffffff, 2.5);
dl1.position.set(1, 1, 5);
scene.add(dl1);

// --- Occluder (Finger Mask) ---
const occluderGeometry = new THREE.CylinderGeometry(0.5, 0.5, 2, 32);
const occluderMaterial = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true });
const occluderMesh = new THREE.Mesh(occluderGeometry, occluderMaterial);
occluderMesh.renderOrder = -1;
scene.add(occluderMesh);
const blockerOccluderMeshes = [];

// --- Load GLTF Model ---
let ringModel = null;
const gltfLoader = new THREE.GLTFLoader();

// Available ring models
const RING_MODELS = [
  { id: 'ring', label: 'Default Ring', path: 'assets/ring.glb' },
  { id: 'Rotation Test 01', label: 'Rotation Test 01', path: 'assets/Rotation Test 01.glb' },
  { id: 'Rotation Test 02', label: 'Rotation Test 02', path: 'assets/Rotation Test 02.glb' },
  { id: 'Rotation Test 03', label: 'Rotation Test 03', path: 'assets/Rotation Test 03.glb' },
  { id: 'Rotation Test 04', label: 'Rotation Test 04', path: 'assets/Rotation Test 04.glb' },
  { id: 'Rotation Test 05', label: 'Rotation Test 05', path: 'assets/Rotation Test 05.glb' }
];

let currentRingIndex = 4; // Start with Rotation Test 02
let previousRingIndex = 4; // For revert

function loadRingModel(modelPath, onComplete) {
  // Remove old ring model from scene
  if (ringModel) {
    scene.remove(ringModel);
    ringModel = null;
  }

  gltfLoader.load(modelPath, (gltf) => {
    const rawModel = gltf.scene;

    // Step 1: Get original bounding box for normalization
    const origBox = new THREE.Box3().setFromObject(rawModel);
    const size = new THREE.Vector3();
    origBox.getSize(size);
    const maxDimRaw = Math.max(size.x, size.y, size.z);
    console.log(`[${modelPath}] Raw bounding box:`, size, "maxDim:", maxDimRaw);

    // Step 2: Normalize to unit size so ALL ring GLBs behave identically
    if (maxDimRaw > 0) {
      rawModel.scale.multiplyScalar(1.0 / maxDimRaw);
    }

    // Step 3: Center AFTER scaling (avoids offset drift)
    const scaledBox = new THREE.Box3().setFromObject(rawModel);
    const scaledCenter = new THREE.Vector3();
    scaledBox.getCenter(scaledCenter);
    rawModel.position.sub(scaledCenter);

    // Step 4: Auto-align ring hole to Y-axis using ring-shape detection
    rawModel.updateMatrixWorld(true);
    const vertices = [];
    rawModel.traverse(node => {
      if (node.isMesh && node.geometry && node.geometry.attributes.position) {
        const pos = node.geometry.attributes.position;
        for (let i = 0; i < pos.count; i++) {
          const v = new THREE.Vector3().fromBufferAttribute(pos, i);
          node.localToWorld(v);
          vertices.push(v);
        }
      }
    });

    if (vertices.length > 0) {
      const centroid = new THREE.Vector3();
      vertices.forEach(v => centroid.add(v));
      centroid.divideScalar(vertices.length);

      function ringScore(axis) {
        const distances = vertices.map(v => {
          const dx = v.x - centroid.x, dy = v.y - centroid.y, dz = v.z - centroid.z;
          if (axis === 'x') return Math.sqrt(dy * dy + dz * dz);
          if (axis === 'y') return Math.sqrt(dx * dx + dz * dz);
          return Math.sqrt(dx * dx + dy * dy);
        });
        const mean = distances.reduce((a, b) => a + b, 0) / distances.length;
        if (mean === 0) return Infinity;
        const variance = distances.reduce((a, d) => a + (d - mean) ** 2, 0) / distances.length;
        return Math.sqrt(variance) / mean;
      }

      const cvX = ringScore('x');
      const cvY = ringScore('y');
      const cvZ = ringScore('z');
      console.log(`[${modelPath}] CV scores — holeX:`, cvX.toFixed(4), "holeY:", cvY.toFixed(4), "holeZ:", cvZ.toFixed(4));

      let holeAxis = 'y';
      let bestCV = cvY;
      if (cvX < bestCV) { bestCV = cvX; holeAxis = 'x'; }
      if (cvZ < bestCV) { bestCV = cvZ; holeAxis = 'z'; }
      console.log(`[${modelPath}] Detected hole axis:`, holeAxis);

      // Rotate so hole axis → Y
      if (holeAxis === 'x') {
        rawModel.rotation.z = Math.PI / 2;
      } else if (holeAxis === 'z') {
        rawModel.rotation.x = Math.PI / 2;
      }
    }

    // Step 5: Re-center after alignment rotation
    rawModel.updateMatrixWorld(true);
    const finalBox = new THREE.Box3().setFromObject(rawModel);
    const finalCenter = new THREE.Vector3();
    finalBox.getCenter(finalCenter);
    rawModel.position.sub(finalCenter);

    // Hierarchy: ringModel (tracking) > rawModel (normalized + aligned)
    ringModel = new THREE.Group();
    ringModel.add(rawModel);

    ringModel.traverse(n => {
      if (n.isMesh && n.material) {
        n.material.metalness = 1.0;
        n.material.roughness = 0.1;
        n.material.envMapIntensity = 2.0;
      }
    });

    // Axes helper for debugging
    const finalSize = new THREE.Vector3();
    new THREE.Box3().setFromObject(ringModel).getSize(finalSize);
    const maxDim = Math.max(finalSize.x, finalSize.y, finalSize.z);
    const axesHelper = new THREE.AxesHelper(maxDim * 2);
    axesHelper.renderOrder = 999;
    axesHelper.traverse(n => {
      if (n.material) {
        n.material.depthTest = false;
        n.material.depthWrite = false;
      }
    });
    ringModel.add(axesHelper);
    console.log(`[${modelPath}] Normalized size:`, finalSize, "Axes size:", maxDim * 2);

    ringModel.visible = false;
    scene.add(ringModel);

    if (onComplete) onComplete();
  });
}

// Ring selector UI
const ringSelect = document.getElementById('ringSelect');
const revertRingBtn = document.getElementById('revertRingBtn');

if (ringSelect) {
  RING_MODELS.forEach((model, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = model.label;
    ringSelect.appendChild(option);
  });
  ringSelect.value = currentRingIndex;

  ringSelect.addEventListener('change', () => {
    const newIndex = parseInt(ringSelect.value);
    if (newIndex !== currentRingIndex) {
      previousRingIndex = currentRingIndex;
      currentRingIndex = newIndex;
      loadRingModel(RING_MODELS[currentRingIndex].path);
      if (revertRingBtn) revertRingBtn.disabled = false;
    }
  });
}

if (revertRingBtn) {
  revertRingBtn.disabled = true;
  revertRingBtn.addEventListener('click', () => {
    if (previousRingIndex !== currentRingIndex) {
      const temp = currentRingIndex;
      currentRingIndex = previousRingIndex;
      previousRingIndex = temp;
      if (ringSelect) ringSelect.value = currentRingIndex;
      loadRingModel(RING_MODELS[currentRingIndex].path);
    }
  });
}

// Load initial model
loadRingModel(RING_MODELS[currentRingIndex].path);

// --- MediaPipe Initialization ---
let handLandmarker;
let lastVideoTime = -1;

async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6
  });
  console.log("MediaPipe Loaded.");
}
initMediaPipe();
initializeFaceTracking(videoElement, scene, camera, landmarkCtx, landmarkCanvas, mapToOrthographicSpace);
initializePendantTracking(scene, mapToOrthographicSpace);
initializeWristTracking(scene, camera, mapToOrthographicSpace);

// AR Variables
let targetPos = new THREE.Vector3(), targetQuat = new THREE.Quaternion(), targetScale = new THREE.Vector3(1, 1, 1);
let targetOccQuat = new THREE.Quaternion(), targetOccScale = new THREE.Vector3(1, 1, 1);
let isHandPresent = false;
let lastHandResults = null; // cached for wrist tracking
const ringFrontFlipQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);

// ORTHOGRAPHIC POSITIONING: Zero Drift Mapping
function mapToOrthographicSpace(point2D) {
  // Simple NDC mapping: No perspective curvature = No Drift.
  let ndcX = (1.0 - point2D.x) * 2 - 1; 
  let ndcY = -(point2D.y) * 2 + 1;
  
  // Convert NDC to world coords based on the camera's current frustum
  return new THREE.Vector3(
    ndcX * (camera.right - camera.left) / 2,
    ndcY * (camera.top - camera.bottom) / 2,
    0
  );
}

// MediaPipe hand connections
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],       // thumb
  [0,5],[5,6],[6,7],[7,8],       // index
  [5,9],[9,10],[10,11],[11,12],  // middle
  [9,13],[13,14],[14,15],[15,16],// ring
  [13,17],[17,18],[18,19],[19,20],// pinky
  [0,17]                          // palm base
];

const SELECTABLE_FINGERS = [
  { id: 'thumb', label: 'Thumb', anchor: [2, 3], widthMul: 1.1 },
  { id: 'index', label: 'Index', anchor: [5, 6], widthMul: 0.95 },
  { id: 'middle', label: 'Middle', anchor: [9, 10], widthMul: 1.0 },
  { id: 'ring', label: 'Ring', anchor: [13, 14], widthMul: 0.95 },
  { id: 'pinky', label: 'Pinky', anchor: [17, 18], widthMul: 0.82 }
];

const BLOCKER_SEGMENTS = [
  { fingerId: 'thumb', pair: [1, 2], widthMul: 1.15 },
  { fingerId: 'thumb', pair: [2, 3], widthMul: 1.05 },
  { fingerId: 'index', pair: [5, 6], widthMul: 0.95 },
  { fingerId: 'index', pair: [6, 7], widthMul: 0.85 },
  { fingerId: 'middle', pair: [9, 10], widthMul: 1.0 },
  { fingerId: 'middle', pair: [10, 11], widthMul: 0.9 },
  { fingerId: 'ring', pair: [13, 14], widthMul: 0.95 },
  { fingerId: 'ring', pair: [14, 15], widthMul: 0.85 },
  { fingerId: 'pinky', pair: [17, 18], widthMul: 0.8 },
  { fingerId: 'pinky', pair: [18, 19], widthMul: 0.72 }
];

let activeFingerIndex = SELECTABLE_FINGERS.findIndex((finger) => finger.id === 'ring');
const Y_AXIS = new THREE.Vector3(0, 1, 0);

function getActiveFinger() {
  return SELECTABLE_FINGERS[activeFingerIndex];
}

function updateFingerPickerUi() {
  if (activeFingerLabel) {
    activeFingerLabel.textContent = `Finger: ${getActiveFinger().label}`;
  }
}

if (cycleFingerButton) {
  cycleFingerButton.addEventListener('click', () => {
    activeFingerIndex = (activeFingerIndex + 1) % SELECTABLE_FINGERS.length;
    updateFingerPickerUi();
  });
}
updateFingerPickerUi();

for (let i = 0; i < BLOCKER_SEGMENTS.length; i++) {
  const blockerMesh = new THREE.Mesh(occluderGeometry, occluderMaterial);
  blockerMesh.renderOrder = -2;
  blockerMesh.visible = false;
  blockerOccluderMeshes.push(blockerMesh);
  scene.add(blockerMesh);
}

function drawLandmarks(landmarks) {
  const w = landmarkCanvas.width;
  const h = landmarkCanvas.height;
  const activeAnchor = new Set(getActiveFinger().anchor);
  landmarkCtx.clearRect(0, 0, w, h);

  // Draw connections
  landmarkCtx.strokeStyle = 'rgba(0, 255, 0, 0.7)';
  landmarkCtx.lineWidth = 2;
  for (const [i, j] of HAND_CONNECTIONS) {
    const x1 = (1 - landmarks[i].x) * w; // mirrored
    const y1 = landmarks[i].y * h;
    const x2 = (1 - landmarks[j].x) * w;
    const y2 = landmarks[j].y * h;
    landmarkCtx.beginPath();
    landmarkCtx.moveTo(x1, y1);
    landmarkCtx.lineTo(x2, y2);
    landmarkCtx.stroke();
  }

  // Draw points
  for (let i = 0; i < landmarks.length; i++) {
    const x = (1 - landmarks[i].x) * w;
    const y = landmarks[i].y * h;
    landmarkCtx.fillStyle = activeAnchor.has(i) ? 'red' : 'cyan';
    landmarkCtx.beginPath();
    landmarkCtx.arc(x, y, 4, 0, Math.PI * 2);
    landmarkCtx.fill();
  }
}

function getMidpoint2D(landmarks, startIdx, endIdx) {
  return {
    x: (landmarks[startIdx].x + landmarks[endIdx].x) / 2,
    y: (landmarks[startIdx].y + landmarks[endIdx].y) / 2
  };
}

function getSegmentLength2D(landmarks, startIdx, endIdx) {
  return Math.sqrt(
    Math.pow(landmarks[endIdx].x - landmarks[startIdx].x, 2) +
    Math.pow(landmarks[endIdx].y - landmarks[startIdx].y, 2)
  );
}

function getSegmentDirection(landmarks, worldLandmarks, startIdx, endIdx) {
  if (worldLandmarks) {
    const start = worldLandmarks[startIdx];
    const end = worldLandmarks[endIdx];
    return new THREE.Vector3(
      -(end.x - start.x),
      -(end.y - start.y),
      -(end.z - start.z)
    ).normalize();
  }

  return new THREE.Vector3().subVectors(
    mapToOrthographicSpace(landmarks[endIdx]),
    mapToOrthographicSpace(landmarks[startIdx])
  ).normalize();
}

function getPalmSpanVector(landmarks, worldLandmarks) {
  if (worldLandmarks) {
    const w5 = worldLandmarks[5];
    const w17 = worldLandmarks[17];
    return new THREE.Vector3(
      -(w17.x - w5.x),
      -(w17.y - w5.y),
      -(w17.z - w5.z)
    ).normalize();
  }

  return new THREE.Vector3().subVectors(
    mapToOrthographicSpace(landmarks[17]),
    mapToOrthographicSpace(landmarks[5])
  ).normalize();
}

function getSegmentDepth(worldLandmarks, startIdx, endIdx) {
  if (!worldLandmarks) return null;
  return ((-worldLandmarks[startIdx].z) + (-worldLandmarks[endIdx].z)) / 2;
}

function hideBlockerOccluders() {
  for (const blockerMesh of blockerOccluderMeshes) {
    blockerMesh.visible = false;
  }
}

function updateBlockerOccluders(landmarks, worldLandmarks, activeFingerId, activeDepth) {
  if (!worldLandmarks || activeDepth === null) {
    hideBlockerOccluders();
    return;
  }

  const occBase = parseFloat(document.getElementById('occluderScale').value);

  for (let i = 0; i < BLOCKER_SEGMENTS.length; i++) {
    const blocker = BLOCKER_SEGMENTS[i];
    const blockerMesh = blockerOccluderMeshes[i];

    if (blocker.fingerId === activeFingerId) {
      blockerMesh.visible = false;
      continue;
    }

    const [startIdx, endIdx] = blocker.pair;
    const depthDelta = getSegmentDepth(worldLandmarks, startIdx, endIdx) - activeDepth;

    if (depthDelta <= 0.004) {
      blockerMesh.visible = false;
      continue;
    }

    const midpoint = getMidpoint2D(landmarks, startIdx, endIdx);
    const segmentDirection = getSegmentDirection(landmarks, worldLandmarks, startIdx, endIdx);
    const segmentLength2D = getSegmentLength2D(landmarks, startIdx, endIdx);
    const blockerQuat = new THREE.Quaternion().setFromUnitVectors(Y_AXIS, segmentDirection);
    const blockerPos = mapToOrthographicSpace(midpoint);
    blockerPos.z = Math.min(0.35, 0.05 + depthDelta * 4.0);

    blockerMesh.position.copy(blockerPos);
    blockerMesh.quaternion.copy(blockerQuat);
    blockerMesh.scale.set(
      segmentLength2D * occBase * blocker.widthMul,
      segmentLength2D * 3.2,
      segmentLength2D * occBase * blocker.widthMul
    );
    blockerMesh.visible = true;
  }
}

function processResults(results) {
  if (!ringModel || !results.landmarks || results.landmarks.length === 0) {
    isHandPresent = false;
    if (ringModel) ringModel.visible = false;
    occluderMesh.visible = false;
    hideBlockerOccluders();
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
    return;
  }

  isHandPresent = true;
  ringModel.visible = true;
  const landmarks = results.landmarks[0];
  drawLandmarks(landmarks);
  const worldLandmarks = results.worldLandmarks ? results.worldLandmarks[0] : null;
  const isReportedRightHand = results.handednesses &&
    results.handednesses[0] &&
    results.handednesses[0][0] &&
    results.handednesses[0][0].categoryName === 'Right';
  const activeFinger = getActiveFinger();
  const [anchorStart, anchorEnd] = activeFinger.anchor;

  // Position (Anchored to the selected finger segment)
  const mid2D = getMidpoint2D(landmarks, anchorStart, anchorEnd);
  targetPos.copy(mapToOrthographicSpace(mid2D));

  // Rotation Basis (Camera independent from World Landmarks)
  // MediaPipe world: X-right, Y-down, Z-away-from-camera
  // Three.js:        X-right, Y-up,   Z-toward-camera
  // Video is CSS mirrored (scaleX(-1)), so X also negates.
  // Conversion: x -> -x, y -> -y, z -> -z
  const D = getSegmentDirection(landmarks, worldLandmarks, anchorStart, anchorEnd);
  const K = getPalmSpanVector(landmarks, worldLandmarks);

  // Build orthonormal basis: D = finger (green/Y), N = palm normal (blue/Z), X = side (red/X)
  const N = new THREE.Vector3().crossVectors(K, D).normalize();
  const X = new THREE.Vector3().crossVectors(D, N).normalize();
  targetOccQuat.setFromRotationMatrix(new THREE.Matrix4().makeBasis(X, D, N));
  
  const rotX = document.getElementById('rotX').value * Math.PI / 180;
  const rotY = document.getElementById('rotY').value * Math.PI / 180;
  const rotZ = document.getElementById('rotZ').value * Math.PI / 180;
  targetQuat.copy(targetOccQuat);
  // MediaPipe handedness is interpreted in the camera/selfie frame, so the
  // reported label is effectively mirrored relative to the on-screen hand here.
  if (!isReportedRightHand) targetQuat.multiply(ringFrontFlipQuat);
  targetQuat.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(rotX, rotY, rotZ, 'XYZ')));

  // Scale (Auto-computed from finger segment — works for any GLB)
  const segStartWorld = mapToOrthographicSpace(landmarks[anchorStart]);
  const segEndWorld = mapToOrthographicSpace(landmarks[anchorEnd]);
  const segLengthWorld = segStartWorld.distanceTo(segEndWorld);
  const sBase = parseFloat(document.getElementById('scaleBase').value);
  const scaleVal = segLengthWorld * sBase;
  targetScale.set(scaleVal, scaleVal, scaleVal);
  const dist2D = getSegmentLength2D(landmarks, anchorStart, anchorEnd);
  
  const occBase = parseFloat(document.getElementById('occluderScale').value);
  targetOccScale.set(
    dist2D * occBase * activeFinger.widthMul,
    dist2D * 4,
    dist2D * occBase * activeFinger.widthMul
  );

  updateBlockerOccluders(
    landmarks,
    worldLandmarks,
    activeFinger.id,
    getSegmentDepth(worldLandmarks, anchorStart, anchorEnd)
  );
}

function checkCanvasSize() {
  const vw = videoElement.videoWidth, vh = videoElement.videoHeight;
  if (vw && vh) {
    if (containerElement.style.aspectRatio !== `${vw} / ${vh}`) {
        containerElement.style.aspectRatio = `${vw} / ${vh}`;
    }
    const rect = containerElement.getBoundingClientRect();
    if (canvasElement.width !== Math.floor(rect.width) || canvasElement.height !== Math.floor(rect.height)) {
      renderer.setSize(rect.width, rect.height, false);
      landmarkCanvas.width = Math.floor(rect.width);
      landmarkCanvas.height = Math.floor(rect.height);
      // Sync Orthographic Frustum to perfectly match the Aspect Ratio
      const aspect = rect.width / rect.height;
      camera.left = -aspect;
      camera.right = aspect;
      camera.top = 1;
      camera.bottom = -1;
      camera.updateProjectionMatrix();
      loadingElement.style.display = 'none';
    }
  }
}

function animate() {
  requestAnimationFrame(animate);
  checkCanvasSize();

  if (handLandmarker && videoElement.readyState >= 2) {
    const startTimeMs = performance.now();
    if (videoElement.currentTime !== lastVideoTime) {
      lastVideoTime = videoElement.currentTime;
      const results = handLandmarker.detectForVideo(videoElement, startTimeMs);
      lastHandResults = results;
      processResults(results);
    }
  }

  // Face tracking (earrings)
  updateFaceTracking(videoElement, performance.now());
  // Pendant tracking (must come after face tracking)
  updatePendantTracking();

  // Wrist tracking (bracelet/watch) — reuses existing hand detection results
  if (lastHandResults && lastHandResults.landmarks && lastHandResults.landmarks.length > 0) {
    const wristLandmarks = lastHandResults.landmarks[0];
    const wristWorldLandmarks = lastHandResults.worldLandmarks ? lastHandResults.worldLandmarks[0] : null;
    const isRightHand = lastHandResults.handednesses &&
      lastHandResults.handednesses[0] &&
      lastHandResults.handednesses[0][0] &&
      lastHandResults.handednesses[0][0].categoryName === 'Right';
    updateWristTracking(wristLandmarks, wristWorldLandmarks, isRightHand);
  } else {
    updateWristTracking(null, null, true);
  }

  if (ringModel && isHandPresent) {
    ringModel.position.lerp(targetPos, 0.9); // Immediate lock-on
    occluderMesh.position.copy(ringModel.position);
    // Keep quaternion interpolation on the shortest arc to avoid apparent spin reversals.
    if (ringModel.quaternion.dot(targetQuat) < 0) targetQuat.set(-targetQuat.x, -targetQuat.y, -targetQuat.z, -targetQuat.w);
    if (occluderMesh.quaternion.dot(targetOccQuat) < 0) targetOccQuat.set(-targetOccQuat.x, -targetOccQuat.y, -targetOccQuat.z, -targetOccQuat.w);
    ringModel.quaternion.slerp(targetQuat, 0.3);
    occluderMesh.quaternion.slerp(targetOccQuat, 0.3);
    ringModel.scale.lerp(targetScale, 0.5);
    occluderMesh.scale.lerp(targetOccScale, 0.5);
    occluderMesh.visible = true;
  } else if (occluderMesh) {
    occluderMesh.visible = false;
    hideBlockerOccluders();
  }
  renderer.render(scene, camera);
}
animate();

navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 960 } })
  .then(stream => { videoElement.srcObject = stream; })
  .catch(() => { loadingElement.innerText = "Camera Error"; });
