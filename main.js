import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
import {
  initializeFaceTracking, updateFaceTracking,
  setEarringMetalColor, setEarringMetalColorPreset,
  activateFaceTracking, deactivateFaceTracking,
  setEarringModel,
  OneEuroFilterVec3, OneEuroFilterQuat, OneEuroFilterScalar
} from './faceTracking.js';
import { configureRenderer, loadHDREnvironment, applyJewelryShading, updateDiamondEnvTwist, setDiamondEnvCube, setDiamondEnvHDR, setMetalColor } from './Shader.js';

// --- DOM Elements ---
const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const landmarkCanvas = document.getElementById('landmark_canvas');
const landmarkCtx = landmarkCanvas.getContext('2d');
const loadingElement = document.getElementById('loading');
const containerElement = document.querySelector('.mirror-container');
const activeFingerLabel = document.getElementById('activeFingerLabel');
const cycleFingerButton = document.getElementById('cycleFingerBtn');

// --- Diagnostic Logging ---
const diagLog = document.getElementById('diag-log');
const DEBUG = false;
function log(msg, isError = false) {
  if (DEBUG) console.log(msg);
  if (!diagLog) return;
  const line = document.createElement('div');
  line.textContent = `[${new Date().toLocaleTimeString().split(' ')[0]}] ${msg}`;
  if (isError) {
    line.style.color = '#f00';
    diagLog.style.display = 'block';
  }
  diagLog.appendChild(line);
  diagLog.scrollTop = diagLog.scrollHeight;
}

window.addEventListener('error', (e) => {
  log(`Global Error: ${e.message} at ${e.filename}:${e.lineno}`, true);
});
window.addEventListener('unhandledrejection', (e) => {
  log(`Promise Rejection: ${e.reason}`, true);
});

// Secret tap to show log
let loadingTaps = 0;
loadingElement?.addEventListener('click', () => {
  if (++loadingTaps >= 5) {
    diagLog.style.display = diagLog.style.display === 'none' ? 'block' : 'none';
    loadingTaps = 0;
  }
});

log("App Starting...");
log(`UserAgent: ${navigator.userAgent}`);
log(`SecureContext: ${window.isSecureContext}`);

const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
log(`IsMobile: ${isMobile}`);

const PERF_TIER = {
  pixelRatioCap: 2.0,
  cubeUpdateStride: 6,
  cubeSize: 64,
  useBVH: true,
  diamondBounces: 2,
  diamondFillAmount: 0.05,
  diamondSparkleStrength: 0.85,
  diamondFringeStrength: 0.07,
  bloomStrength: 0.22,
  bloomRadius: 0.2,
  bloomThreshold: 0.90,
  usePostFX: false,
  useDynamicDiamondCube: true,
};
log(`PerfTier: ${isMobile ? 'mobile' : 'desktop'} / pixelRatio=${PERF_TIER.pixelRatioCap}, useBVH=${PERF_TIER.useBVH}, bounces=${PERF_TIER.diamondBounces}, postFX=${PERF_TIER.usePostFX}, cube=${PERF_TIER.cubeSize}³ stride ${PERF_TIER.cubeUpdateStride}`);

// The camera feed is requested at 30fps, so rendering faster than that mostly
// repeats the same camera frame while burning GPU time on diamonds/reflections.
// Pacing the final render to the camera cadence gives the tracker and browser
// compositor breathing room and produces a steadier 25-30fps.
const TARGET_RENDER_FPS = 30;
const RENDER_FRAME_MS = 1000 / TARGET_RENDER_FPS;

// Lightweight in-app FPS meter for quick perf checks.
const fpsMeterElement = document.createElement('div');
fpsMeterElement.id = 'fps-meter';
fpsMeterElement.style.cssText = [
  'position:absolute',
  'left:12px',
  'bottom:12px',
  'z-index:120',
  'padding:4px 8px',
  'border-radius:8px',
  'background:rgba(0,0,0,0.55)',
  'border:1px solid rgba(255,255,255,0.2)',
  'color:#fff',
  'font:600 12px/1.2 monospace',
  'pointer-events:none',
].join(';');
fpsMeterElement.textContent = 'FPS: --';
if (containerElement) containerElement.appendChild(fpsMeterElement);

// --- Three.js Setup ---
const scene = new THREE.Scene();
// Add a fallback reflection cube so models aren't black if HDR fails
const gen = new THREE.PMREMGenerator(new THREE.WebGLRenderer({ alpha: true }));
scene.environment = gen.fromScene(new THREE.Scene()).texture;
gen.dispose();

const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({
  canvas: canvasElement,
  alpha: true,
  antialias: false, // Turn OFF for major GPU speedup on Intel UHD
  fps: 60,
  depth: true
});
renderer.sortObjects = true; // respect renderOrder for occluder-before-jewelry
renderer.localClippingEnabled = true; // allow per-material clipping planes (earring post clip)
renderer.setClearAlpha(0);
configureRenderer(renderer);
log("Renderer initialized");

// Pixel-ratio cap. Increased to 2x on mobile to match desktop clarity.
// Caps at 2x even on 3x screens to maintain a balance between sharpness and performance.
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, PERF_TIER.pixelRatioCap));

// --- Video backdrop plane ---------------------------------------------------
// The video feed was previously ONLY behind the canvas via CSS, so the 3D
// scene had no opaque backdrop — which meant MeshPhysicalMaterial.transmission
// had nothing to refract through, collapsing gems into chrome/glass.
// Putting a full-screen textured quad at the far plane makes the video part of
// the opaque pass → transmission's internal render target captures it → the
// gem's refraction shader actually bends the hand pixels. This is the physics
// that produces real diamond look.
const videoTexture = new THREE.VideoTexture(videoElement);
videoTexture.minFilter = THREE.LinearFilter;
videoTexture.magFilter = THREE.LinearFilter;
videoTexture.encoding = THREE.sRGBEncoding;
const videoBackdrop = new THREE.Mesh(
  new THREE.PlaneGeometry(2, 2),
  new THREE.ShaderMaterial({
    uniforms: { tVideo: { value: videoTexture } },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        // clip-space fullscreen, pushed to the far plane so it never clips geometry
        gl_Position = vec4(position.xy, 0.99999, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D tVideo;
      varying vec2 vUv;
      void main() {
        // CSS on the <video> uses scaleX(-1) for selfie mode; match by flipping u
        gl_FragColor = texture2D(tVideo, vec2(1.0 - vUv.x, vUv.y));
      }
    `,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,   // passthrough — the raw camera feed, no ACES pass
  })
);
videoBackdrop.frustumCulled = false;
videoBackdrop.renderOrder = -1000;       // render first → lands in transmission RT
scene.add(videoBackdrop);
log("Video backdrop added");

// --- Post-processing: UnrealBloomPass for diamond glint ---
// threshold 0.95 → only the brightest dispersion pings bloom, so the bloom
// feels like sparkle (not a milky haze over everything).
// RenderTargets use RGBA so the transparent canvas (video background) is preserved.
const bloomRT = new THREE.WebGLRenderTarget(1, 1, {
  minFilter: THREE.LinearFilter,
  magFilter: THREE.LinearFilter,
  format: THREE.RGBAFormat,
  type: THREE.HalfFloatType,
  samples: 0 // Disable MSAA to save massive GPU bandwidth (crucial for 30FPS)
});
const composer = new THREE.EffectComposer(renderer, bloomRT);
const renderPass = new THREE.RenderPass(scene, camera);
renderPass.clearAlpha = 0; // keep background transparent through the pass chain
composer.addPass(renderPass);

const bloomPass = new THREE.UnrealBloomPass(
  new THREE.Vector2(1, 1),
  PERF_TIER.bloomStrength,  // strength — low so bloom feels like a pinpoint sparkle, not a blown-out halo
  PERF_TIER.bloomRadius,   // radius — tightened from 0.3 → crisp star-sparkle points instead of
  //   a soft halo. The glow now reads as a pinpoint flare.
  PERF_TIER.bloomThreshold   // threshold — lowered to 0.90 so hot sparkles bleed light and halo
  //   into glowing stars (was 0.98 — too strict, clipped most pinpoints).
);
composer.addPass(bloomPass);
bloomPass.enabled = PERF_TIER.usePostFX;

// UnrealBloomPass in r128 does not preserve the framebuffer alpha: its composite
// path writes opaque pixels, which hides the <video> element behind the canvas.
// We render the scene a second time into sceneRT purely to capture a clean alpha
// channel, then a final ShaderPass emits vec4(bloom.rgb, sceneRT.a) so empty
// pixels stay transparent and the live camera feed shows through.
const AlphaRestoreShader = {
  uniforms: {
    tDiffuse: { value: null },
    tScene: { value: null }, // Assigned in animate() to composer.renderTarget2.texture
    time: { value: 0.0 }
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform sampler2D tScene;
    uniform float time;
    varying vec2 vUv;
    
    // Pseudo-random noise function
    float rand(vec2 co){
        return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
    }
    
    void main() {
      vec4 bloom = texture2D(tDiffuse, vUv);
      vec4 src   = texture2D(tScene,   vUv);
      
      vec3 finalColor = bloom.rgb;
      
      // Apply subtle film grain/noise ONLY to the 3D elements (where alpha > 0)
      // to match the noise profile of the live camera feed.
      if (src.a > 0.01) {
          float noise = (rand(vUv + time) - 0.5) * 0.045;
          finalColor += noise;
      }
      
      gl_FragColor = vec4(finalColor, src.a);
    }
  `
};
const alphaRestorePass = new THREE.ShaderPass(AlphaRestoreShader);
composer.addPass(alphaRestorePass);
log("Composer & Passes ready");

// Low-intensity fill lights — HDR IBL does the heavy lifting, these just cover
// the window between page load and HDR finishing its async load.
scene.add(new THREE.AmbientLight(0xffffff, 0.3));
const dl1 = new THREE.DirectionalLight(0xffffff, 0.3); // Softened to match background
dl1.position.set(1, 1, 5);
scene.add(dl1);

// Sparkle catchlights — angled pair of bright directional lights. With
// flatShading gems, these produce the hard crescent highlights that read as
// diamond fire when the hand rotates. Kept tight so they don't wash out metal.
const sparkleKey = new THREE.DirectionalLight(0xffffff, 1.2); // Softened from 1.6
sparkleKey.position.set(2.5, 3, 4);
scene.add(sparkleKey);
const sparkleRim = new THREE.DirectionalLight(0xffe8d6, 0.8); // Softened from 1.2
sparkleRim.position.set(-2.5, -1.5, 3);
scene.add(sparkleRim);
log("Lights added");

// --- Dynamic diamond environment cube ---------------------------------------
// The BVH diamond shader used to sample the HDR equirect directly, so the gem
// only reflected the studio sky — never the metal prongs next to it. Real
// diamonds reflect their mounting as dark structural lines inside the stone;
// without that, the gem looks like it's floating in a void.
//
// This CubeCamera re-renders the scene into a 128×128 cube every frame, with
// the diamonds + video backdrop temporarily hidden (to prevent self-feedback
// and clip-space artifacts). The resulting cube contains: HDR sky (placed as
// scene.background only during capture) + ring metal. We hand that cube to
// Shader.setDiamondEnvCube — the shader's samplerCube envMap now shows the
// metal right where the prongs are, giving the stone real dark mounting lines.
const diamondCubeRT = new THREE.WebGLCubeRenderTarget(PERF_TIER.cubeSize, {
  format: THREE.RGBAFormat,
  type: THREE.HalfFloatType,              // HDR range preserved for hot highlights
  generateMipmaps: false,
  minFilter: THREE.LinearFilter,
  magFilter: THREE.LinearFilter
});
const diamondCubeCamera = new THREE.CubeCamera(0.01, 100, diamondCubeRT);
scene.add(diamondCubeCamera);
setDiamondEnvCube(diamondCubeRT.texture);

// Held so we can (a) seed the cube with HDR sky on load and (b) swap it in as
// scene.background only while CubeCamera is capturing.
let _hdrEquirectForCube = null;

// Load HDR studio environment → PMREM → scene.environment (IBL for all PBR materials).
// brown_photostudio_02 rotated 90° around Y. The rotation repositions the
// studio softboxes so their specular hits land on flattering angles of the
// ring's band instead of a glancing edge. Rotation is baked ONCE into the
// equirect pixels inside loadHDREnvironment, so every downstream consumer —
// PMREM for metals, CubeCamera seed, scene.background during cube capture, and
// the BVH shader's direct envHDR sampling — all inherit the rotated HDR
// automatically without any per-path rotation logic.
loadHDREnvironment(
  renderer,
  scene,
  'assets/brown_photostudio_02_1k.hdr',
  { rotationY: Math.PI / 2 }
)
  .then(({ equirect }) => {
    log("HDR loaded");
    _hdrEquirectForCube = equirect;
    // One-time prefill so the first few frames (before the CubeCamera has run)
    // already show the studio sky instead of pure black in the gem.
    diamondCubeRT.fromEquirectangularTexture(renderer, equirect);
    // Hand the clean HDR to the diamond shader — BVH refractive-exit rays
    // sample it so cluster gems don't darken each other via the cube.
    setDiamondEnvHDR(equirect);
  })
  .catch((err) => {
    log(`HDR error: ${err.message}`, true);
    /* fill lights already cover the fallback case */
  });

// Per-frame cube-reflection capture. Called from animate() after the ring has
// been positioned for this frame and just before the final render. Everything
// that would pollute the cube (diamonds — feedback; video backdrop — clip-space
// quad; axes helpers — colored debug lines) is hidden for the duration of the
// capture.
//
// Perf — stride: cube re-capture renders the scene 6 times (one per face) at
// 128×128. At 60fps that's 360 scene renders per second purely for
// reflections. Updating every Nth frame drops that to 120/sec (stride=3)
// without any visible difference — hand motion is slow enough that stale-by-
// 2-frames reflections are imperceptible. Mobile gets the biggest win here.
const CUBE_UPDATE_STRIDE = PERF_TIER.cubeUpdateStride;
// Stationary skip: if the ring has barely moved since the last cube capture,
// six new scene renders would produce a near-identical texture — skip them.
// Thresholds picked so a user slowly translating the hand still gets fresh
// reflections (pos threshold ≈ 0.5mm in world units, quat dot > 0.9998 ≈ 1°).
// A hard refresh every MAX_STATIC_FRAMES × stride frames guarantees the cube
// still picks up slow-drifting environmental changes (HDR finishing load,
// metal-color swap, ring-model change).
const CUBE_POS_THRESHOLD_SQ = 0.0005 * 0.0005;
const CUBE_QUAT_THRESHOLD = 0.9998;
const CUBE_MAX_STATIC_CYCLES = 20;
let _cubeFrameCounter = 0;
let _cubeStaticCycles = 0;
const _prevCubePos = new THREE.Vector3(Infinity, Infinity, Infinity);
const _prevCubeQuat = new THREE.Quaternion(0, 0, 0, 1);
function updateDiamondReflectionCube() {
  if (!PERF_TIER.useDynamicDiamondCube) return;
  // Skip cube camera entirely when no hand is present — reflections only matter
  // when the ring is actually visible on screen.
  if (!isHandPresent || !ringModel || !ringModel.visible) return;
  if ((_cubeFrameCounter++ % CUBE_UPDATE_STRIDE) !== 0) return;

  const posDeltaSq = _prevCubePos.distanceToSquared(ringModel.position);
  const quatDot = Math.abs(_prevCubeQuat.dot(ringModel.quaternion));
  const stationary = posDeltaSq < CUBE_POS_THRESHOLD_SQ && quatDot > CUBE_QUAT_THRESHOLD;
  if (stationary && _cubeStaticCycles < CUBE_MAX_STATIC_CYCLES) {
    _cubeStaticCycles++;
    return;
  }
  _cubeStaticCycles = 0;
  _prevCubePos.copy(ringModel.position);
  _prevCubeQuat.copy(ringModel.quaternion);

  diamondCubeCamera.position.copy(ringModel.position);

  // traverseVisible skips invisible subtrees — important now that the ring
  // cache keeps all non-active rings resident in the scene with visible=false.
  // Plain traverse would still descend into them and waste cycles.
  const hidden = [];
  scene.traverseVisible((n) => {
    // Hide anything carrying our diamond marker (BVH ShaderMaterial or PBR
    // onBeforeCompile-injected material both set userData.diamondUniforms).
    if (n.isMesh && n.material) {
      const mats = Array.isArray(n.material) ? n.material : [n.material];
      if (mats.some((m) => m?.userData?.diamondUniforms || m?.userData?.isDiamondBVH)) {
        hidden.push(n); n.visible = false; return;
      }
    }
    // AxesHelper / LineSegments — noisy colored lines; don't bake into reflections.
    if (n.isLineSegments || n.type === 'AxesHelper') {
      hidden.push(n); n.visible = false;
    }
  });

  const videoBackdropWasVisible = videoBackdrop.visible;
  videoBackdrop.visible = false;

  const savedBackground = scene.background;
  const savedToneMapping = renderer.toneMapping;
  if (_hdrEquirectForCube) scene.background = _hdrEquirectForCube;
  renderer.toneMapping = THREE.NoToneMapping; // keep linear HDR values in the cube

  diamondCubeCamera.update(renderer, scene);

  scene.background = savedBackground;
  renderer.toneMapping = savedToneMapping;
  videoBackdrop.visible = videoBackdropWasVisible;
  for (const n of hidden) n.visible = true;
}

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

// Available ring models. Each `preset` captures the slider values that make
// the model sit correctly on the finger — applied on load/swap so the user
// doesn't have to re-tune rotX/rotY/rotZ/scale for every ring.
const RING_MODELS = [
  { id: 'ring', label: 'Default Ring', path: 'assets/ring.glb', preset: { rotX: 164, rotY: 90, rotZ: -67, scale: 0.75 } },
  { id: 'Rotation Test 01', label: 'Rotation Test 01', path: 'assets/Rotation Test 01.glb', preset: { rotX: 73, rotY: 90, rotZ: -67, scale: 0.75 } },
  { id: 'Rotation Test 02', label: 'Rotation Test 02', path: 'assets/Rotation Test 02.glb', preset: { rotX: 73, rotY: 90, rotZ: -67, scale: 0.75 } },
  { id: 'Rotation Test 04', label: 'Rotation Test 04', path: 'assets/Rotation Test 04.glb', preset: { rotX: 164, rotY: 90, rotZ: -67, scale: 1.0 } },
  { id: 'Rotation Test 05', label: 'Rotation Test 05', path: 'assets/Rotation Test 05.glb', preset: { rotX: 44, rotY: 86, rotZ: 39, scale: 0.8 } },
  { id: '01 Clean', label: '01 Clean', path: 'assets/01 Clean.glb', preset: { rotX: 101, rotY: 2, rotZ: -2, scale: 0.8 } },
];

// Push a ring's preset into the rotation + scale UI (both range sliders and
// the paired number inputs), so processResults() picks up the new values on
// the next frame without any extra wiring.
function applyRingPreset(preset) {
  if (!preset) return;
  const pairs = [
    ['rotX', 'numX', preset.rotX],
    ['rotY', 'numY', preset.rotY],
    ['rotZ', 'numZ', preset.rotZ],
    ['scaleBase', 'numScale', preset.scale],
  ];
  for (const [rangeId, numId, value] of pairs) {
    if (value === undefined) continue;
    const range = document.getElementById(rangeId);
    const num = document.getElementById(numId);
    if (range) range.value = value;
    if (num) num.value = value;
  }
}

let currentRingIndex = 3; // Start with Rotation Test 04 (after removing Rotation Test 03)
let previousRingIndex = 3; // For revert

// Metal tint applied to every band (ring + earring). Read by applyJewelryShading
// when a model loads, and updated live via setMetalColor() when the user picks
// a new option from the dropdown.
let currentMetalPreset = 'silver';
setEarringMetalColorPreset(currentMetalPreset);

// --- Mode gating for MediaPipe trackers -------------------------------------
// Running both HandLandmarker and FaceLandmarker every frame is the single
// biggest MediaPipe cost on mobile. When the user unchecks "Show Ring" or
// "Show Earring" the corresponding tracker's detectForVideo call is skipped
// and the jewelry it controls is hidden. Tier 1 (pause-only) — landmarkers
// stay loaded; reactivation is instant.
let _handTrackingActive = true;
function isHandTrackingActive() { return _handTrackingActive; }
function activateHandTracking() {
  _handTrackingActive = true;
  lastVideoTime = -1; // force fresh detect on the next animate tick
}
function deactivateHandTracking() {
  _handTrackingActive = false;
  isHandPresent = false;
  if (ringModel) ringModel.visible = false;
  if (typeof occluderMesh !== 'undefined' && occluderMesh) occluderMesh.visible = false;
  if (typeof hideBlockerOccluders === 'function') hideBlockerOccluders();
  // Clear any stale hand-landmark overlay so the canvas doesn't keep the
  // last drawn skeleton while tracking is paused.
  if (landmarkCtx && landmarkCanvas) {
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
  }
  // Hide the pose hint — only relevant during active ring tracking.
  if (typeof _setPoseHintVisible === 'function') _setPoseHintVisible(false);
}

// Ring cache — each loaded GLB gets built once into a fully-shaded THREE.Group
// and kept resident in the scene with visible=false. Switching rings becomes
// a visibility flip instead of a GLB fetch + BVH rebuild + shader compile.
// That's the single biggest per-switch win and it's the whole point of
// Phase B. Key = ring id (from RING_MODELS entry).
const ringCache = new Map();
const ringLoadPromises = new Map();  // id → Promise, dedupes concurrent requests

// Build a ring Group from a GLB path. Does normalization, hole-axis alignment,
// PBR/BVH shading, and attaches the debug axes helper. Returns a Group ready
// to be added to the scene (or, in our case, already added and hidden).
function buildRingGroup(modelPath, rawModel) {
  // Step 1: Get original bounding box for normalization
  const origBox = new THREE.Box3().setFromObject(rawModel);
  const size = new THREE.Vector3();
  origBox.getSize(size);
  const maxDimRaw = Math.max(size.x, size.y, size.z);
  console.log(`[${modelPath}] Raw bounding box:`, size, "maxDim:", maxDimRaw);

  // Step 2: Normalize to unit size so ALL ring GLBs behave identically
  if (maxDimRaw > 0) rawModel.scale.multiplyScalar(1.0 / maxDimRaw);

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

    if (holeAxis === 'x') rawModel.rotation.z = Math.PI / 2;
    else if (holeAxis === 'z') rawModel.rotation.x = Math.PI / 2;
  }

  // Step 5: Re-center after alignment rotation
  rawModel.updateMatrixWorld(true);
  const finalBox = new THREE.Box3().setFromObject(rawModel);
  const finalCenter = new THREE.Vector3();
  finalBox.getCenter(finalCenter);
  rawModel.position.sub(finalCenter);

  // Hierarchy: group (tracking) > rawModel (normalized + aligned)
  const group = new THREE.Group();
  group.add(rawModel);

  const matNames = new Set();
  group.traverse((n) => {
    if (n.isMesh) {
      const mats = Array.isArray(n.material) ? n.material : [n.material];
      mats.forEach((m) => m && matNames.add(`${n.name || '?'} → ${m.name || '(unnamed)'}`));
    }
  });
  console.log(`[${modelPath}] materials:`, [...matNames]);

  applyJewelryShading(group, {
    metal: {
      // Mobile-only boost: at lower pixel ratio (1.65 vs 2.0) the thin
      // ring band can rasterize sub-pixel on certain poses and visually
      // "break up." Desktop hides this because the UnrealBloomPass halo
      // around bright specular highlights visually fills in the gaps.
      // Mobile doesn't run post-FX, so we compensate with three tricks:
      //   1. Sharper highlights (roughness 0.04 vs 0.06)
      //   2. Brighter reflections (envMapIntensity 2.4 vs 1.8)
      //   3. Full clearcoat rim — reflective shell widens edges ~0.5-1px
      //   4. Subtle self-emissive tinted to metal color — lifts thin band
      //      pixels so they stay visible even when specular angle doesn't
      //      catch them. 0.08 intensity is low enough to not read as a
      //      "glow" but high enough to carry thin geometry.
      //
      // Emissive is TINTED TO THE METAL COLOR (default behavior when no
      // emissive is specified in Shader.js), so gold stays gold-lift,
      // silver stays silver-lift. Never reads as wrong color.
      metalness: 1.0,
      roughness: 0.06,
      envMapIntensity: 1.8,
      clearcoat: 0.7,
      clearcoatRoughness: 0.04,
      emissiveIntensity: 0,
      color: currentMetalPreset
    },
    diamond: {
      transmission: 1.0, ior: 2.417,
      metalness: 0.0, roughness: 0.0,
      clearcoat: 0.0, clearcoatRoughness: 0.0,
      envMapIntensity: 3.5,
      attenuationDistance: 2.5,
      dispersion: 0.010,
      sparkleStrength: PERF_TIER.diamondSparkleStrength,
      fringeStrength: PERF_TIER.diamondFringeStrength,
      // Desktop: full BVH internal ray-trace (useBVH=true default in Shader.js).
      // Mobile: useBVH=false falls back to MeshPhysicalMaterial.transmission +
      // sparkle overlay — visually very close, 2-3× fewer fragment ops since
      // there's no per-pixel bounds-hierarchy walk.
      useBVH: PERF_TIER.useBVH,
      bounces: PERF_TIER.diamondBounces,
      fillAmount: PERF_TIER.diamondFillAmount,
      useCubeReflection: PERF_TIER.useDynamicDiamondCube
    }
  });

  const finalSize = new THREE.Vector3();
  new THREE.Box3().setFromObject(group).getSize(finalSize);
  const maxDim = Math.max(finalSize.x, finalSize.y, finalSize.z);
  const axesHelper = new THREE.AxesHelper(maxDim * 2);
  axesHelper.renderOrder = 999;
  axesHelper.traverse(n => {
    if (n.material) { n.material.depthTest = false; n.material.depthWrite = false; }
  });
  const showRingAxesEl = document.getElementById('showRingAxes');
  axesHelper.visible = !!(showRingAxesEl && showRingAxesEl.checked);
  group.add(axesHelper);
  group.userData.axesHelper = axesHelper;
  console.log(`[${modelPath}] Normalized size:`, finalSize, "Axes size:", maxDim * 2);

  group.visible = false;
  return group;
}

// Preload a ring into the cache (idempotent). Returns a Promise<Group>.
// Called eagerly on startup for the default ring, and during idle time for
// the rest so the first swap is effectively instant.
function preloadRing(entry) {
  if (ringCache.has(entry.id)) return Promise.resolve(ringCache.get(entry.id));
  if (ringLoadPromises.has(entry.id)) return ringLoadPromises.get(entry.id);

  const p = new Promise((resolve) => {
    gltfLoader.load(entry.path, (gltf) => {
      const group = buildRingGroup(entry.path, gltf.scene);
      ringCache.set(entry.id, group);
      scene.add(group);
      // Pre-compile: the ring group is in the scene with visible=false, so
      // the renderer hasn't yet linked the BVH shader program for its
      // materials. renderer.compile() traverses ALL materials in the scene
      // (not only visible) and force-compiles their GL programs + uploads
      // textures to the GPU. Cost: one-time 200-500ms at load. Payoff: the
      // user's first ring-swap frame doesn't freeze for 500-2000ms while
      // the driver compiles the shader on demand. Wrapped in try/catch
      // because older WebGL impls occasionally throw on unusual materials.
      try { renderer.compile(scene, camera); } catch (e) {
        console.warn('[PreCompile] renderer.compile failed (non-fatal):', e?.message || e);
      }
      ringLoadPromises.delete(entry.id);
      resolve(group);
    });
  });
  ringLoadPromises.set(entry.id, p);
  return p;
}

// Swap visibility to the named ring. If it's not cached yet, loads it first
// then shows it. Also re-applies the current metal color (the cached ring
// may have been built with a stale preset) and axes checkbox state.
function showRing(entry) {
  const cached = ringCache.get(entry.id);
  if (!cached) {
    preloadRing(entry).then(() => showRing(entry));
    return;
  }
  if (ringModel === cached) return;
  if (ringModel) ringModel.visible = false;
  ringModel = cached;
  // visible=true here means "allowed to show"; animate() still gates on
  // isHandPresent, so off-screen rings don't appear at the origin.
  if (isHandPresent) ringModel.visible = true;

  // Sync current metal tint + axes-helper checkbox onto the swapped-in ring
  // (they may have drifted while this ring was hidden).
  setMetalColor(ringModel, currentMetalPreset);
  const axes = ringModel.userData.axesHelper;
  const showAxes = document.getElementById('showRingAxes')?.checked;
  if (axes) axes.visible = !!showAxes;
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
      applyRingPreset(RING_MODELS[currentRingIndex].preset);
      showRing(RING_MODELS[currentRingIndex]);
      if (revertRingBtn) revertRingBtn.disabled = false;
    }
  });
}

// Metal dropdown — live swap of band tint across ring + earring without
// reloading any models (just mutates each metal material's .color).
const metalSelect = document.getElementById('metalSelect');
if (metalSelect) {
  metalSelect.value = currentMetalPreset;
  metalSelect.addEventListener('change', () => {
    currentMetalPreset = metalSelect.value;
    if (ringModel) setMetalColor(ringModel, currentMetalPreset);
    setEarringMetalColor(currentMetalPreset);
    setEarringMetalColorPreset(currentMetalPreset);
  });
}

// --- Tracker mode toggles ----------------------------------------------------
// Show Ring / Show Earring drive the tracker gates. Unchecking a box pauses
// the corresponding MediaPipe detector (detectForVideo is no longer called)
// and hides all jewelry that tracker controls. Rechecking resumes tracking
// on the next frame.
const showRingToggle = document.getElementById('showRing');
if (showRingToggle) {
  showRingToggle.addEventListener('change', () => {
    if (showRingToggle.checked) activateHandTracking();
    else deactivateHandTracking();
  });
}

// Ring axes helper toggle — hidden by default; flips the cached axesHelper
// stashed on ringModel.userData when a model is loaded.
const showRingAxesToggle = document.getElementById('showRingAxes');
if (showRingAxesToggle) {
  showRingAxesToggle.addEventListener('change', () => {
    if (ringModel && ringModel.userData.axesHelper) {
      ringModel.userData.axesHelper.visible = showRingAxesToggle.checked;
    }
  });
}

const showEarringToggle = document.getElementById('showEarring');
let _faceTrackingInitPromise = null;
let _faceTrackingInitialized = false;

function ensureFaceTrackingInitialized() {
  if (_faceTrackingInitialized) return Promise.resolve();
  if (_faceTrackingInitPromise) return _faceTrackingInitPromise;

  _faceTrackingInitPromise = initializeFaceTracking(
    videoElement,
    scene,
    camera,
    landmarkCtx,
    landmarkCanvas,
    mapToOrthographicSpace
  )
    .then(() => {
      _faceTrackingInitialized = true;
    })
    .catch((err) => {
      _faceTrackingInitPromise = null;
      throw err;
    });

  return _faceTrackingInitPromise;
}

function enableFaceTracking() {
  ensureFaceTrackingInitialized()
    .then(() => activateFaceTracking())
    .catch((err) => {
      console.error('[FaceTracking] Initialization failed:', err);
      deactivateFaceTracking();
      if (showEarringToggle) showEarringToggle.checked = false;
      if (activeCategory === 'earrings') {
        activeCategory = 'rings';
        categoryBtns.forEach((b) => b.classList.toggle('active', b.dataset.category === 'rings'));
        if (showRingToggle) showRingToggle.checked = true;
        renderThumbs();
      }
    });
}

if (showEarringToggle) {
  // Default state: earring off. Sync module state to checkbox value so the
  // face tracker stays quiet until the user opts in.
  if (!showEarringToggle.checked) deactivateFaceTracking();
  showEarringToggle.addEventListener('change', () => {
    if (showEarringToggle.checked) enableFaceTracking();
    else deactivateFaceTracking();
  });
}

// --- Settings panel collapse/expand toggle ----------------------------------
// Gear button floats over the video feed. Clicking it flips a body class
// that CSS uses to slide the panel in/out + rotate the gear icon. Panel
// starts collapsed (class applied in index.html) so the camera is fully
// visible from frame 1 — especially important on mobile where the panel
// would otherwise cover half the screen.
const uiToggleBtn = document.getElementById('ui-toggle');
const uiPanel = document.getElementById('ui-panel');
if (uiToggleBtn && uiPanel) {
  uiToggleBtn.addEventListener('click', () => {
    const isCollapsed = uiPanel.classList.toggle('ui-panel-collapsed');
    document.body.classList.toggle('ui-panel-open', !isCollapsed);
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
      applyRingPreset(RING_MODELS[currentRingIndex].preset);
      showRing(RING_MODELS[currentRingIndex]);
    }
  });
}

// Initial ring — load synchronously and show it. Then kick off background
// preloads for every other ring during idle time so subsequent swaps are
// zero-latency. requestIdleCallback is missing on Safari; fall back to a
// short-delay setTimeout chain so each load still yields to MediaPipe /
// render work between jobs.
applyRingPreset(RING_MODELS[currentRingIndex].preset);
preloadRing(RING_MODELS[currentRingIndex]).then(() => showRing(RING_MODELS[currentRingIndex]));

const schedule = window.requestIdleCallback
  ? (fn) => window.requestIdleCallback(fn, { timeout: 2000 })
  : (fn) => setTimeout(fn, 200);
RING_MODELS.forEach((entry, i) => {
  if (i === currentRingIndex) return;
  schedule(() => preloadRing(entry));
});

// ============================================================================
// Product picker — category tabs (Rings / Earrings) + lightweight model strip.
// Enforces tracker mutex: only one MediaPipe model runs at a time, which is
// the single biggest per-frame cost. Keeps the existing settings panel +
// its checkboxes intact — just syncs their state so they stay coherent.
// ============================================================================
const EARRING_MODELS = [
  { id: 'Earring2', label: 'Drop', path: 'assets/Earring2.glb' },
];

let currentEarringIndex = 0;
let activeCategory = 'rings'; // 'rings' | 'earrings'

const productThumbsEl = document.getElementById('product-thumbs');
const categoryBtns = document.querySelectorAll('.category-btn');

function renderThumbs() {
  if (!productThumbsEl) return;
  productThumbsEl.innerHTML = '';
  const list = activeCategory === 'rings' ? RING_MODELS : EARRING_MODELS;
  const activeIdx = activeCategory === 'rings' ? currentRingIndex : currentEarringIndex;
  list.forEach((item, index) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'product-thumb' + (index === activeIdx ? ' active' : '');
    btn.title = item.label;
    btn.dataset.modelId = item.id;
    btn.textContent = item.label.split(/\s+/).map(w => w[0]).join('').slice(0, 3).toUpperCase();
    btn.addEventListener('click', () => onThumbClick(index));
    productThumbsEl.appendChild(btn);
  });
}

function onThumbClick(index) {
  if (activeCategory === 'rings') {
    if (index === currentRingIndex) return;
    previousRingIndex = currentRingIndex;
    currentRingIndex = index;
    if (ringSelect) ringSelect.value = currentRingIndex;
    if (revertRingBtn) revertRingBtn.disabled = false;
    applyRingPreset(RING_MODELS[currentRingIndex].preset);
    showRing(RING_MODELS[currentRingIndex]);
  } else {
    if (index === currentEarringIndex) return;
    currentEarringIndex = index;
    ensureFaceTrackingInitialized()
      .then(() => setEarringModel(EARRING_MODELS[currentEarringIndex].path))
      .catch((err) => console.error('[FaceTracking] Earring model swap failed:', err));
  }
  renderThumbs();
}

function setCategory(category) {
  if (category === activeCategory) return;
  activeCategory = category;

  categoryBtns.forEach(b => b.classList.toggle('active', b.dataset.category === category));

  // Tracker mutex — only the active category's MediaPipe landmarker runs.
  // Also mirror the state into the existing settings-panel checkboxes so
  // they stay coherent with the picker (user can still flip them manually).
  if (category === 'rings') {
    activateHandTracking();
    deactivateFaceTracking();
    if (showRingToggle) showRingToggle.checked = true;
    if (showEarringToggle) showEarringToggle.checked = false;
  } else {
    deactivateHandTracking();
    enableFaceTracking();
    if (showRingToggle) showRingToggle.checked = false;
    if (showEarringToggle) showEarringToggle.checked = true;
  }

  renderThumbs();
}

categoryBtns.forEach(btn => {
  btn.addEventListener('click', () => setCategory(btn.dataset.category));
});

// Initial render — rings category active by default. No need to call
// setCategory() because that's also the default tracker state (hand on,
// face off), and the 'active' class is already applied in the HTML.
renderThumbs();

// Product model buttons intentionally stay as lightweight text badges. Avoiding
// generated PNG thumbnails keeps a second WebGL renderer and GPU readback off
// the critical path.

// ============================================================================
// Metal picker — three metallic swatches above the product picker. Shares
// state with the existing #metalSelect dropdown in the settings panel, so
// clicks here move the dropdown and vice-versa. Applies to both ring + earring
// instantly via setMetalColor / setEarringMetalColor (no reload).
// ============================================================================
const metalSwatches = document.querySelectorAll('.metal-swatch');
metalSwatches.forEach((sw) => {
  sw.addEventListener('click', () => {
    const preset = sw.dataset.preset;
    if (!preset || preset === currentMetalPreset) return;
    currentMetalPreset = preset;

    // Live-apply to everything currently in the scene.
    if (ringModel) setMetalColor(ringModel, currentMetalPreset);
    setEarringMetalColor(currentMetalPreset);
    setEarringMetalColorPreset(currentMetalPreset);

    // Keep the settings-panel dropdown in sync.
    if (metalSelect) metalSelect.value = currentMetalPreset;

    metalSwatches.forEach((s) => s.classList.toggle('active', s === sw));
  });
});

// If the user changes the settings dropdown instead, mirror the active class
// onto the matching swatch. The dropdown's change handler already applies the
// color — we just handle the visual sync here.
if (metalSelect) {
  metalSelect.addEventListener('change', () => {
    metalSwatches.forEach((s) => s.classList.toggle('active', s.dataset.preset === metalSelect.value));
  });
}

// --- MediaPipe Initialization ---
let handLandmarker;
let _vision = null;
let _handFallbackInitPromise = null;
let _currentDelegate = 'GPU';
let _consecutiveMisses = 0;
let _hadDetectionOnce = false;
let _recovering = false;
window.__forceAllCpu = false;

let lastVideoTime = -1;

async function createHandLandmarker(delegate) {
  return await HandLandmarker.createFromOptions(_vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: delegate
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6
  });
}

async function recoverHandToCpu() {
  if (_recovering || _currentDelegate === 'CPU') return;
  _recovering = true;
  window.__forceAllCpu = true;
  console.warn("[HandTracking] GPU stuck — recreating on CPU.");
  try { handLandmarker.close(); } catch (e) { }
  handLandmarker = null;
  try {
    handLandmarker = await createHandLandmarker('CPU');
    _currentDelegate = 'CPU';
    _consecutiveMisses = 0;
    _hadDetectionOnce = false;
    lastVideoTime = -1;
    console.log("[HandTracking] Recovered on CPU.");
  } catch (e) {
    console.error("[HandTracking] CPU recreate failed:", e);
  } finally {
    _recovering = false;
  }
}

async function ensureMainHandLandmarker(delegate = 'GPU') {
  if (handLandmarker) return handLandmarker;
  if (_handFallbackInitPromise) return _handFallbackInitPromise;

  _handFallbackInitPromise = (async () => {
    console.log("[Init] Loading fallback MediaPipe hand tracker...");
    if (loadingElement) loadingElement.innerText = "Loading AI Models...";

    if (!_vision) {
      _vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
      );
    }
    handLandmarker = await createHandLandmarker(delegate);
    _currentDelegate = delegate;
    console.log(`[Init] Fallback hand tracker loaded (${delegate}).`);
    return handLandmarker;
  })();

  try {
    return await _handFallbackInitPromise;
  } catch (err) {
    _handFallbackInitPromise = null;
    console.error("[Init] Fallback MediaPipe Error:", err);
    if (loadingElement) loadingElement.innerText = "AI Init Error: " + err.message;
    throw err;
  }
}

// --- Hand tracking Web Worker ----------------------------------------------
// Parallelizes MediaPipe detection with rendering. Main thread grabs an
// ImageBitmap from the video each new frame and transfers it to the worker;
// worker runs detectForVideo and posts landmarks back. Main-thread path
// (above) stays initialized as a fallback — if the worker fails to init
// (old browser, no module-worker support, CDN cross-origin block) we just
// keep the synchronous path.
//
// Perf model:
//   Before: main_frame_cost = detect_ms + render_ms         (serial)
//   After:  main_frame_cost = max(detect_ms, render_ms)     (parallel)
// When detect and render are similar (slow GPU + slow MediaPipe) the gain
// is ~50%. When render dominates (integrated GPU + BVH shader), the gain
// is smaller — only the 15-40 ms of detection cost stops stealing from rAF.
let _handWorker = null;
let _workerReady = false;
let _useWorker = false;
let _pendingWorkerDetect = false;
let _workerFrameId = 0;
let _lastWorkerDetectMs = 0;

function _handleWorkerResults(results, detectMs, delegate) {
  _lastWorkerDetectMs = detectMs;
  if (delegate && delegate !== _currentDelegate) _currentDelegate = delegate;
  const nDetections = results?.landmarks?.length || 0;
  if (nDetections > 0) { _hadDetectionOnce = true; _consecutiveMisses = 0; }
  else { _consecutiveMisses++; }

  // Mirror the main-thread recovery heuristic into the worker path: if GPU
  // delegate is stuck (many consecutive empty results), ask the worker to
  // switch to CPU.
  const shouldRecover = _currentDelegate === 'GPU' && !_recovering && (
    window.__forceAllCpu ||
    (_hadDetectionOnce && _consecutiveMisses > 20) ||
    (!_hadDetectionOnce && _consecutiveMisses > 60)
  );
  if (shouldRecover) {
    _recovering = true;
    _handWorker.postMessage({ type: 'recover-cpu' });
  }
  processResults(results);
}

function _initHandWorker() {
  try {
    _handWorker = new Worker(new URL('./handWorker.js', import.meta.url), { type: 'module' });
  } catch (err) {
    console.warn('[HandWorker] Could not spawn worker:', err);
    _useWorker = false;
    return;
  }

  _handWorker.onmessage = (e) => {
    const msg = e.data;
    switch (msg.type) {
      case 'ready':
        _workerReady = true;
        _useWorker = true;
        _currentDelegate = msg.delegate || 'GPU';
        log(`[HandWorker] Ready — delegate: ${_currentDelegate}`);
        break;
      case 'error':
        console.warn('[HandWorker] Error, falling back to main thread:', msg.error);
        _useWorker = false;
        _workerReady = false;
        ensureMainHandLandmarker(window.__forceAllCpu ? 'CPU' : 'GPU').catch(() => { });
        break;
      case 'results':
        _pendingWorkerDetect = false;
        _handleWorkerResults(msg.results, msg.detectMs, msg.delegate);
        break;
      case 'detect-error':
        _pendingWorkerDetect = false;
        _consecutiveMisses++;
        break;
      case 'dropped':
        _pendingWorkerDetect = false;
        break;
      case 'recovered':
        _currentDelegate = 'CPU';
        _recovering = false;
        _consecutiveMisses = 0;
        _hadDetectionOnce = false;
        log(`[HandWorker] Recovered on CPU`);
        break;
    }
  };
  _handWorker.onerror = (e) => {
    console.warn('[HandWorker] Crash — reverting to main thread:', e.message);
    _useWorker = false;
    _workerReady = false;
    ensureMainHandLandmarker(window.__forceAllCpu ? 'CPU' : 'GPU').catch(() => { });
  };

  _handWorker.postMessage({ type: 'init' });
}
_initHandWorker();

// AR Variables
let targetPos = new THREE.Vector3(), targetQuat = new THREE.Quaternion(), targetScale = new THREE.Vector3(1, 1, 1);
let targetOccQuat = new THREE.Quaternion(), targetOccScale = new THREE.Vector3(1, 1, 1);
let isHandPresent = false;
const ringFrontFlipQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
let fpsFrameCount = 0;
let fpsLastSampleTime = performance.now();
let lastRenderTimeMs = 0;

// One-Euro filters for the ring pose — replaces naive lerp-on-noisy-input
// with adaptive smoothing that snaps during fast hand motion and smooths
// hard when stationary. Applied inside processResults to targetPos/Quat/
// Scale before the render-loop lerp/slerp, so the lerp interpolates
// between clean samples instead of chasing MediaPipe's per-frame jitter.
//
// Param tuning (mirrors the face tracker's conventions):
//   freq 30   — video frame rate (filter auto-adapts from timestamp deltas)
//   minCutoff — stationary noise floor; lower = smoother at rest
//   beta      — motion responsiveness; higher = snappier during motion
//
// Slightly reduced smoothing for snappier response.
const HAND_FILT_FREQ = 30;
const ringPosFilter = new OneEuroFilterVec3(HAND_FILT_FREQ, 1.0, 0.8, 1.0);
const ringQuatFilter = new OneEuroFilterQuat(HAND_FILT_FREQ, 1.0, 0.5, 1.0);
const ringScaleFilter = new OneEuroFilterScalar(HAND_FILT_FREQ, 1.0, 0.1, 1.0);

// --- Motion prediction (pose forward-extrapolation) -------------------------
// End-to-end sensor→pixels latency on mobile is ~3-5 frames (50-80ms): camera
// capture → decode → MediaPipe inference → JS math → render → compositor.
// Even with perfect tracking the ring visually trails the finger during fast
// motion. We cancel most of that delay by storing the smoothed per-second
// velocity of targetPos and extrapolating forward in animate() by a tuned
// horizon (~25-40ms). Shorter horizons undershoot the lag, longer ones
// overshoot when the hand stops.
//
// Safety rails:
//   • EWMA alpha 0.30 — velocity adapts within ~3 frames, ignores single-
//     frame detection glitches.
//   • Magnitude clamp (8 units/s) — if MediaPipe jumps landmarks (rare hand
//     re-detection), velocity won't catapult the ring across the screen.
//   • dt check (<200ms) — first frame after long dropout doesn't compute a
//     spurious velocity from stale state.
//   • On hand-lost, _hasPrevTarget resets so re-detection starts clean.
//
// Quaternion/scale prediction intentionally skipped — the existing slerp
// damping already handles rotation well enough, and scale changes are slow.
const PREDICTION_HORIZON_S = 0.025;
const MAX_VELOCITY_UNITS_PER_S = 8.0;
const _posVelocity = new THREE.Vector3();
const _prevTargetPos = new THREE.Vector3();
let _prevTargetTimeMs = 0;
let _hasPrevTarget = false;

// --- Hot-path scratch buffers ------------------------------------------------
// processResults + updateBlockerOccluders run every video frame. Previously
// each call allocated a dozen Vector3 / Quaternion / Matrix4 / Euler objects,
// and on low-end Androids those allocations add up to periodic GC stalls that
// show as micro-stutters in the ring pose. These module-level scratches are
// overwritten in-place inside the hot paths, so steady-state allocations
// drop to zero. Naming convention: _tmp* = "short-lived, don't rely across
// function calls". Safe because updateBlockerOccluders runs AFTER processResults
// has copied its values into targetPos/Quat/Scale/OccQuat/OccScale.
const _tmpPosA = new THREE.Vector3();
const _tmpPosB = new THREE.Vector3();
const _tmpDirD = new THREE.Vector3();
const _tmpDirK = new THREE.Vector3();
const _tmpDirN = new THREE.Vector3();
const _tmpDirX = new THREE.Vector3();
const _tmpMat4 = new THREE.Matrix4();
const _tmpEuler = new THREE.Euler();
const _tmpQuatA = new THREE.Quaternion();
const _tmpMid2D = { x: 0, y: 0 };

// --- Cached DOM refs for sliders + checkboxes read every frame --------------
// `document.getElementById()` is not free — it's a hash-map lookup on the
// document tree plus a protection-check for detached nodes. Doing it 5× per
// frame inside processResults cost ~40-60µs on mobile Chromium, and the same
// 5 elements never change. Resolve once at module init and hold the refs.
const _slRotX = document.getElementById('rotX');
const _slRotY = document.getElementById('rotY');
const _slRotZ = document.getElementById('rotZ');
const _slScaleBase = document.getElementById('scaleBase');
const _slOccluderScale = document.getElementById('occluderScale');
const _cbShowHandMesh = document.getElementById('showHandMesh');
const _poseHintEl = document.getElementById('pose-hint');

// --- Pose-quality gate ------------------------------------------------------
// Two DIFFERENT bad-pose signals — either one trips the hint. Hysteresis on
// each independently so neither flickers.
//
// Signal A — finger-at-camera (|D.z| near 1):
//   The finger direction D aligns with the camera's viewing axis. Ring
//   basis is under-constrained → ring looks tilted/floating in the image.
//   Thresholds lenient: only hides when finger is within ~25° of lens axis.
//
// Signal B — fanned/spread-finger pose (palmSpan / middleFinger > 1.6):
//   When fingers are spread wide (image from THIS session), MediaPipe's
//   hand-model fit gets noisy at the MCP joints — adjacent fingers' landmarks
//   contaminate each other and the ring finger's 3D direction drifts 5-15°,
//   tilting the ring relative to the actual finger. Pure math can't correct
//   this (it's a model-fitting issue, not a geometry one) → show hint
//   instead. Only trips when the palm is visibly fanned wide (ratio 1.6+,
//   measured in 2D so it's camera-distance-invariant).
const POSE_D_Z_HIDE = 0.90;
const POSE_D_Z_SHOW = 0.75;
const POSE_SPREAD_HIDE = 1.60;  // palm-span/middle-finger — fingers fanned out
const POSE_SPREAD_SHOW = 1.40;  // back to relaxed hand — show ring again
let _ringPoseAcceptable = true;

function _setPoseHintVisible(show) {
  if (!_poseHintEl) return;
  if (show) _poseHintEl.classList.add('visible');
  else _poseHintEl.classList.remove('visible');
}

// ORTHOGRAPHIC POSITIONING: Zero Drift Mapping
// *Into variant writes into `out` and returns it (avoids per-call Vector3
// allocation in hot paths). The original wrapper is preserved for any external
// consumer that expects a fresh Vector3 return (currently none in-tree).
function mapToOrthographicSpaceInto(out, point2D) {
  const ndcX = (1.0 - point2D.x) * 2 - 1;
  const ndcY = -(point2D.y) * 2 + 1;
  return out.set(
    ndcX * (camera.right - camera.left) / 2,
    ndcY * (camera.top - camera.bottom) / 2,
    0
  );
}
function mapToOrthographicSpace(point2D) {
  // Simple NDC mapping: No perspective curvature = No Drift.
  return mapToOrthographicSpaceInto(new THREE.Vector3(), point2D);
}

// MediaPipe hand connections
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // index
  [5, 9], [9, 10], [10, 11], [11, 12],  // middle
  [9, 13], [13, 14], [14, 15], [15, 16],// ring
  [13, 17], [17, 18], [18, 19], [19, 20],// pinky
  [0, 17]                          // palm base
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

// Fills the shared _tmpMid2D scratch so callers can consume .x/.y without
// allocating. Returned reference === the scratch; do not cache across frames.
function getMidpoint2D(landmarks, startIdx, endIdx) {
  _tmpMid2D.x = (landmarks[startIdx].x + landmarks[endIdx].x) / 2;
  _tmpMid2D.y = (landmarks[startIdx].y + landmarks[endIdx].y) / 2;
  return _tmpMid2D;
}

function getSegmentLength2D(landmarks, startIdx, endIdx) {
  return Math.sqrt(
    Math.pow(landmarks[endIdx].x - landmarks[startIdx].x, 2) +
    Math.pow(landmarks[endIdx].y - landmarks[startIdx].y, 2)
  );
}

function getSegmentLength3D(worldLandmarks, idxA, idxB) {
  if (!worldLandmarks) return 0;
  const a = worldLandmarks[idxA];
  const b = worldLandmarks[idxB];
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// Aspect-normalized 2D length used ONLY by the pose-quality gate.
// MediaPipe normalizes x by image width and y by image height independently,
// so a ratio of two normalized distances along different axes is biased by
// the frame aspect — desktop landscape and mobile portrait produce different
// numbers for the same physical pose, which makes the popup trip on mobile
// for poses that work fine on desktop. Rescaling dx by (actual_aspect /
// desktop_aspect) cancels that bias so mobile produces the SAME spreadRatio
// desktop is producing today. Desktop is unchanged (multiplied by 1.0), so
// the existing POSE_SPREAD_HIDE / SHOW thresholds keep working unmodified.
const _POSE_REF_ASPECT = 4 / 3;
function getSegmentLengthForPose(landmarks, startIdx, endIdx, vw, vh) {
  const aspect = (vw && vh) ? (vw / vh) : _POSE_REF_ASPECT;
  const dx = (landmarks[endIdx].x - landmarks[startIdx].x) * (aspect / _POSE_REF_ASPECT);
  const dy = (landmarks[endIdx].y - landmarks[startIdx].y);
  return Math.sqrt(dx * dx + dy * dy);
}

// *Into variants write into `out` and return it so the hot path never
// allocates. The no-worldLandmarks branch uses the two _tmpPosA/B scratches
// for the NDC conversion — they're free at this point (the caller consumes
// the returned direction before invoking another function that might reuse
// those scratches).
function getSegmentDirectionInto(out, landmarks, worldLandmarks, startIdx, endIdx) {
  if (worldLandmarks) {
    const start = worldLandmarks[startIdx];
    const end = worldLandmarks[endIdx];
    return out.set(
      -(end.x - start.x),
      -(end.y - start.y),
      -(end.z - start.z)
    ).normalize();
  }
  mapToOrthographicSpaceInto(_tmpPosA, landmarks[endIdx]);
  mapToOrthographicSpaceInto(_tmpPosB, landmarks[startIdx]);
  return out.subVectors(_tmpPosA, _tmpPosB).normalize();
}

function getSegmentDirection(landmarks, worldLandmarks, startIdx, endIdx) {
  return getSegmentDirectionInto(new THREE.Vector3(), landmarks, worldLandmarks, startIdx, endIdx);
}

function getPalmSpanVectorInto(out, landmarks, worldLandmarks) {
  if (worldLandmarks) {
    const w5 = worldLandmarks[5];
    const w17 = worldLandmarks[17];
    return out.set(
      -(w17.x - w5.x),
      -(w17.y - w5.y),
      -(w17.z - w5.z)
    ).normalize();
  }
  return null;
}

function getPalmSpanVector(landmarks, worldLandmarks) {
  return getPalmSpanVectorInto(new THREE.Vector3(), landmarks, worldLandmarks);
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

  const occluderBase = parseFloat(_slOccluderScale.value);

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

    // All these values are copied onto blockerMesh before the next iteration,
    // so sharing scratch buffers across iterations is safe.
    const midpoint = getMidpoint2D(landmarks, startIdx, endIdx);
    getSegmentDirectionInto(_tmpDirD, landmarks, worldLandmarks, startIdx, endIdx);
    const segmentLength2D = getSegmentLength2D(landmarks, startIdx, endIdx);
    _tmpQuatA.setFromUnitVectors(Y_AXIS, _tmpDirD);
    mapToOrthographicSpaceInto(_tmpPosA, midpoint);
    _tmpPosA.z = Math.min(0.35, 0.05 + depthDelta * 4.0);

    blockerMesh.position.copy(_tmpPosA);
    blockerMesh.quaternion.copy(_tmpQuatA);
    blockerMesh.scale.set(
      segmentLength2D * occluderBase * blocker.widthMul,
      segmentLength2D * 3.2,
      segmentLength2D * occluderBase * blocker.widthMul
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
    // Hide the pose hint when no hand is present — the hint is specifically
    // "rotate your hand," which doesn't apply if there's no hand in frame.
    _setPoseHintVisible(false);
    _ringPoseAcceptable = true;   // reset to optimistic for next detection
    // Reset One-Euro state so a re-detection doesn't resume from an old
    // sample (would otherwise "swing" toward the stale position on the
    // first frame after tracking returns).
    ringPosFilter.reset();
    ringQuatFilter.reset();
    ringScaleFilter.reset();
    // Clear motion-prediction state — next detection must establish a fresh
    // velocity baseline instead of extrapolating from whatever the hand was
    // doing before it left frame.
    _hasPrevTarget = false;
    _posVelocity.set(0, 0, 0);
    return;
  }

  isHandPresent = true;
  ringModel.visible = true;
  const landmarks = results.landmarks[0];
  // Gate hand-skeleton overlay on the Show Hand Mesh checkbox. When unchecked,
  // clear the canvas instead of drawing — otherwise the last drawn skeleton
  // would linger on screen.
  if (_cbShowHandMesh && _cbShowHandMesh.checked) {
    drawLandmarks(landmarks);
  } else {
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
  }
  const worldLandmarks = results.worldLandmarks ? results.worldLandmarks[0] : null;
  const isReportedRightHand = results.handednesses &&
    results.handednesses[0] &&
    results.handednesses[0][0] &&
    results.handednesses[0][0].categoryName === 'Right';
  const activeFinger = getActiveFinger();
  const [anchorStart, anchorEnd] = activeFinger.anchor;

  // Position (Anchored to the selected finger segment). Uses _tmpPosA as a
  // scratch for the NDC conversion before copying into targetPos.
  const mid2D = getMidpoint2D(landmarks, anchorStart, anchorEnd);
  mapToOrthographicSpaceInto(_tmpPosA, mid2D);
  targetPos.copy(_tmpPosA);

  // Rotation Basis (Camera independent from World Landmarks)
  // MediaPipe world: X-right, Y-down, Z-away-from-camera
  // Three.js:        X-right, Y-up,   Z-toward-camera
  // Video is CSS mirrored (scaleX(-1)), so X also negates.
  getSegmentDirectionInto(_tmpDirD, landmarks, worldLandmarks, anchorStart, anchorEnd);
  const kResult = getPalmSpanVectorInto(_tmpDirK, landmarks, worldLandmarks);

  // If no worldLandmarks, skip frame — don't render with bad basis
  if (!kResult) return;

  // Build orthonormal basis: D = finger (green/Y), N = palm normal (blue/Z), X = side (red/X)
  _tmpDirN.crossVectors(_tmpDirK, _tmpDirD).normalize();

  // Guard: K and D too parallel → degenerate normal
  if (_tmpDirN.lengthSq() < 0.01) {
    _tmpDirN.crossVectors(Y_AXIS, _tmpDirD).normalize();
    if (_tmpDirN.lengthSq() < 0.01) {
      _tmpDirN.crossVectors(new THREE.Vector3(0, 0, 1), _tmpDirD).normalize();
    }
  }

  _tmpDirX.crossVectors(_tmpDirD, _tmpDirN).normalize();
  _tmpMat4.makeBasis(_tmpDirX, _tmpDirD, _tmpDirN);
  targetOccQuat.setFromRotationMatrix(_tmpMat4);

  // Pose-quality gate. Two independent bad-pose signals, either one hides.
  //   A) |D.z| high → finger along camera axis (basis degenerate)
  //   B) palmSpan / middleFinger high → fingers fanned wide (landmark noise)
  // Each uses its own hysteresis band so neither flickers near threshold.
  // The spread-ratio is computed from 2D landmarks (camera-space distances)
  // so it's invariant to how close the hand is to the camera.
  const fingerZMag = Math.abs(_tmpDirD.z);
  const _vw = videoElement.videoWidth || 0;
  const _vh = videoElement.videoHeight || 0;
  const palmSpan2D = getSegmentLengthForPose(landmarks, 5, 17, _vw, _vh);
  const middleSeg2D = getSegmentLengthForPose(landmarks, 9, 10, _vw, _vh);
  const spreadRatio = middleSeg2D > 1e-4 ? palmSpan2D / middleSeg2D : 0;

  // -- COMMENTED OUT POSE RESTRICTION --
  // if (_ringPoseAcceptable) {
  //   if (fingerZMag > POSE_D_Z_HIDE || spreadRatio > POSE_SPREAD_HIDE) {
  //     _ringPoseAcceptable = false;
  //   }
  // } else {
  //   if (fingerZMag < POSE_D_Z_SHOW && spreadRatio < POSE_SPREAD_SHOW) {
  //     _ringPoseAcceptable = true;
  //   }
  // }
  // if (!_ringPoseAcceptable) {
  //   // Hide the ring + occluders and surface the hint. We return early so
  //   // the filter/prediction state doesn't get polluted with the degenerate
  //   // side-view samples — they'd poison the smoothing when the user
  //   // returns to a good pose.
  //   ringModel.visible = false;
  //   occluderMesh.visible = false;
  //   hideBlockerOccluders();
  //   _setPoseHintVisible(true);
  //   ringPosFilter.reset();
  //   ringQuatFilter.reset();
  //   ringScaleFilter.reset();
  //   _hasPrevTarget = false;
  //   _posVelocity.set(0, 0, 0);
  //   return;
  // }
  _ringPoseAcceptable = true;
  _setPoseHintVisible(false);

  const rotX = _slRotX.value * Math.PI / 180;
  const rotY = _slRotY.value * Math.PI / 180;
  const rotZ = _slRotZ.value * Math.PI / 180;
  targetQuat.copy(targetOccQuat);
  if (!isReportedRightHand) targetQuat.multiply(ringFrontFlipQuat);
  _tmpEuler.set(rotX, rotY, rotZ, 'XYZ');
  _tmpQuatA.setFromEuler(_tmpEuler);
  targetQuat.multiply(_tmpQuatA);

  // Scale (Auto-computed from finger segment — works for any GLB). Reuses
  // _tmpPosA/B scratches (previously set above for the ring position, value
  // has already been copied into targetPos so we can clobber).
  mapToOrthographicSpaceInto(_tmpPosA, landmarks[anchorStart]);
  mapToOrthographicSpaceInto(_tmpPosB, landmarks[anchorEnd]);
  const segLengthWorld = _tmpPosA.distanceTo(_tmpPosB);
  const sBase = parseFloat(_slScaleBase.value);
  const scaleVal = segLengthWorld * sBase;
  targetScale.set(scaleVal, scaleVal, scaleVal);

  // One-Euro filter pass: replaces targetPos / targetQuat / targetScale with
  // their adaptive-smoothed versions. Stationary hand = heavy smoothing
  // (jitter dies); moving hand = light smoothing (tracking stays snappy).
  // Applied AFTER all the derivation math so the filters see the final
  // composed pose — cleaner than filtering each input landmark.
  // targetOccQuat/targetOccScale are NOT filtered — the occluder is
  // invisible (depth-only), so its jitter doesn't matter visually and
  // filtering it would just add cost.
  const tSec = performance.now() / 1000;
  targetPos.copy(ringPosFilter.filter(targetPos, tSec));
  targetQuat.copy(ringQuatFilter.filter(targetQuat, tSec));
  const smoothedScale = ringScaleFilter.filter(scaleVal, tSec);
  targetScale.set(smoothedScale, smoothedScale, smoothedScale);
  const dist2D = getSegmentLength2D(landmarks, anchorStart, anchorEnd);

  // Velocity update for motion prediction. Uses the FILTERED targetPos so
  // velocity reflects the clean pose stream, not MediaPipe's raw jitter.
  // _tmpPosA is free here (previously used for segStart/End, that work is
  // done). Stored velocity is consumed in animate() — target itself is not
  // modified, so next-frame velocity calculation stays grounded in the
  // filtered signal rather than the predicted one (prevents feedback).
  const _predNowMs = tSec * 1000;
  if (_hasPrevTarget) {
    const dtMs = _predNowMs - _prevTargetTimeMs;
    if (dtMs > 0 && dtMs < 200) {
      const invDt = 1000 / dtMs;
      _tmpPosA.subVectors(targetPos, _prevTargetPos).multiplyScalar(invDt);
      const vLen = _tmpPosA.length();
      if (vLen > MAX_VELOCITY_UNITS_PER_S) {
        _tmpPosA.multiplyScalar(MAX_VELOCITY_UNITS_PER_S / vLen);
      }
      _posVelocity.lerp(_tmpPosA, 0.30);
    }
  }
  _prevTargetPos.copy(targetPos);
  _prevTargetTimeMs = _predNowMs;
  _hasPrevTarget = true;

  const occluderBase = parseFloat(_slOccluderScale.value);

  const dist3D = getSegmentLength3D(worldLandmarks, anchorStart, anchorEnd);

  // Use segLengthWorld (aspect-corrected) instead of dist2D (warped) 
  // to ensure the occluder doesn't grow when the hand is horizontal.
  const foreshortenFactor = segLengthWorld > 1e-4 ? Math.max(1.0, (dist3D * 5.0) / (segLengthWorld * 1.0)) : 1.0;
  const finalForeshorten = Math.min(foreshortenFactor, 2.5);

  targetOccScale.set(
    segLengthWorld * occluderBase * activeFinger.widthMul * 0.5,
    segLengthWorld * 2.0 * finalForeshorten, 
    segLengthWorld * occluderBase * activeFinger.widthMul * 0.5
  );
  updateBlockerOccluders(
    landmarks,
    worldLandmarks,
    activeFinger.id,
    getSegmentDepth(worldLandmarks, anchorStart, anchorEnd)
  );
}

// ---- Phase A: Debounced canvas resize via ResizeObserver ------------------
// Previously checkCanvasSize ran every animation frame. During a DevTools
// responsive-mode toggle or a phone rotation the container changes size on
// many frames in a row, and each change triggers GPU reallocations for
// renderer / composer / bloomPass / sceneRT — this is what was locking up
// the page for seconds.
//
// Replacement: ResizeObserver fires only on real size changes, and we
// further gate the heavy allocations behind a 150ms settle timer so the
// viewport has to stop moving before we actually re-allocate. Cached
// dimensions short-circuit identical sizes so we never pay the cost twice.
let _lastAppliedW = 0;
let _lastAppliedH = 0;
let _resizeTimer = 0;

function applyCanvasSize() {
  _resizeTimer = 0;
  const rect = containerElement.getBoundingClientRect();
  const w = Math.floor(rect.width);
  const h = Math.floor(rect.height);
  if (w <= 0 || h <= 0) return;
  if (w === _lastAppliedW && h === _lastAppliedH) return;
  _lastAppliedW = w;
  _lastAppliedH = h;

  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  bloomPass.setSize(w, h);
  landmarkCanvas.width = w;
  landmarkCanvas.height = h;

  const aspect = w / h;
  camera.left = -aspect;
  camera.right = aspect;
  camera.top = 1;
  camera.bottom = -1;
  camera.updateProjectionMatrix();
  if (loadingElement) loadingElement.style.display = 'none';
}

function scheduleCanvasResize(immediate) {
  if (_resizeTimer) { clearTimeout(_resizeTimer); _resizeTimer = 0; }
  if (immediate) applyCanvasSize();
  else _resizeTimer = setTimeout(applyCanvasSize, 150);
}

// Lock the container's aspect ratio to the camera feed once we know what it
// is. Runs once per stream (camera switches fire loadedmetadata again) and
// triggers an immediate resize so the first frame is correctly sized.
videoElement.addEventListener('loadedmetadata', () => {
  const vw = videoElement.videoWidth, vh = videoElement.videoHeight;
  if (!vw || !vh) return;
  const ratio = `${vw} / ${vh}`;
  if (containerElement.style.aspectRatio !== ratio) {
    containerElement.style.aspectRatio = ratio;
  }
  scheduleCanvasResize(true);
});

// ResizeObserver catches everything else: window resize, DevTools device-
// mode toggle, orientation change, panel open/close, zoom, etc. Debounced
// so rapid intermediate sizes during an animation don't thrash the GPU.
const _canvasResizeObserver = new ResizeObserver(() => scheduleCanvasResize(false));
_canvasResizeObserver.observe(containerElement);

// --- requestVideoFrameCallback hook -----------------------------------------
// Polling `video.currentTime !== lastVideoTime` works but has two issues on
// mobile: (1) it wastes every 60Hz rAF tick's worth of compare work when the
// video decodes at 30Hz, and (2) it gives us no exact frame presentation time
// — which motion prediction (a later task) needs to compute pipeline latency.
//
// rVFC fires exactly once per decoded frame and exposes `metadata.presentationTime`
// (when the browser will scan it out) and `metadata.mediaTime` (video clock).
// We set a "new frame ready" flag here; animate() resets it after consuming.
// Firefox Android and older Safari lack rVFC — `_hasRVFC` is false there and
// the legacy currentTime-polling path stays in effect.
const _hasRVFC = typeof HTMLVideoElement !== 'undefined'
  && typeof HTMLVideoElement.prototype.requestVideoFrameCallback === 'function';
let _newHandFrame = true;          // true on startup so the first detect fires
let _latestFramePresentMs = 0;     // wall-clock when the frame hits the screen
let _latestFrameMediaTimeS = 0;    // decoded-frame timestamp (video clock)
function _onVideoFrame(now, metadata) {
  _newHandFrame = true;
  _latestFramePresentMs = metadata.presentationTime;
  _latestFrameMediaTimeS = metadata.mediaTime;
  videoElement.requestVideoFrameCallback(_onVideoFrame);
}
if (_hasRVFC) {
  videoElement.requestVideoFrameCallback(_onVideoFrame);
  log('rVFC active — frame-accurate detection gate');
} else {
  log('rVFC unavailable — falling back to currentTime polling');
}

function animate() {
  requestAnimationFrame(animate);
  const nowMs = performance.now();

  // rVFC path preferred: the flag is set exactly once per decoded frame.
  // currentTime fallback for browsers without rVFC.
  const hasNewHandFrame = _hasRVFC
    ? _newHandFrame
    : (videoElement.currentTime !== lastVideoTime);

  // Hand tracking — skipped when the Show Ring toggle is off.
  if (isHandTrackingActive() && videoElement.readyState >= 2 && hasNewHandFrame) {
    // Prefer the Web Worker path when it's up. We dispatch an ImageBitmap of
    // the current video frame to the worker and move on — detection happens
    // in parallel with rendering. Results arrive via onmessage and route
    // through _handleWorkerResults → processResults.
    if (_useWorker && _workerReady && !_pendingWorkerDetect) {
      _newHandFrame = false;
      lastVideoTime = videoElement.currentTime;
      _pendingWorkerDetect = true;
      _workerFrameId++;
      const frameId = _workerFrameId;
      const timestamp = performance.now();
      // createImageBitmap is async but usually <2ms — doesn't block rendering.
      // The bitmap is transferred (zero-copy) to the worker.
      createImageBitmap(videoElement).then((bitmap) => {
        // The user may have toggled hand tracking off between dispatch and
        // bitmap creation — drop it cleanly rather than sending a dead frame.
        if (!isHandTrackingActive() || !_useWorker) {
          try { bitmap.close(); } catch (_) { }
          _pendingWorkerDetect = false;
          return;
        }
        _handWorker.postMessage({
          type: 'detect',
          bitmap,
          timestamp,
          frameId
        }, [bitmap]);
      }).catch((err) => {
        console.warn('[HandWorker] createImageBitmap failed, dropping frame:', err);
        _pendingWorkerDetect = false;
      });
    } else if (!_useWorker) {
      if (!handLandmarker) {
        ensureMainHandLandmarker(window.__forceAllCpu ? 'CPU' : 'GPU').catch(() => { });
        return;
      }
      // Fallback: synchronous main-thread detection. Same code as before the
      // worker was introduced — used when the worker failed to init or the
      // browser doesn't support module workers.
      _newHandFrame = false;
      const startTimeMs = performance.now();
      lastVideoTime = videoElement.currentTime;

      let results = null;
      let threw = false;
      try {
        results = handLandmarker.detectForVideo(videoElement, startTimeMs);
      } catch (e) {
        threw = true;
        console.warn("[HandTracking] detectForVideo threw:", e?.message || e);
      }

      const nDetections = results?.landmarks?.length || 0;
      if (nDetections > 0) { _hadDetectionOnce = true; _consecutiveMisses = 0; }
      else { _consecutiveMisses++; }
      if (threw) _consecutiveMisses++;

      const shouldRecover = _currentDelegate === 'GPU' && !_recovering && (
        window.__forceAllCpu ||
        (_hadDetectionOnce && _consecutiveMisses > 20) ||
        (!_hadDetectionOnce && _consecutiveMisses > 60)
      );
      if (shouldRecover) recoverHandToCpu();

      if (!threw && results) {
        processResults(results);
      }
    }
  }

  // Face tracking (earrings) - only update if the category is earrings to save CPU
  if (activeCategory === 'earrings') {
    updateFaceTracking(videoElement, performance.now());
  }

  if (lastRenderTimeMs && (nowMs - lastRenderTimeMs) < (RENDER_FRAME_MS - 1)) {
    return;
  }
  lastRenderTimeMs = nowMs;

  if (ringModel && isHandPresent) {
    // Forward-extrapolate the filtered target pos along smoothed velocity so
    // the ring's visible position matches where the finger WILL be when this
    // frame hits the screen, not where it was when MediaPipe finished. The
    // velocity was computed from the filtered stream (not the predicted one),
    // so there's no feedback loop.
    _tmpPosA.copy(targetPos).addScaledVector(_posVelocity, PREDICTION_HORIZON_S);
    ringModel.position.lerp(_tmpPosA, 0.95); // Slightly snappier lock-on
    occluderMesh.position.copy(ringModel.position);
    // Keep quaternion interpolation on the shortest arc to avoid apparent spin reversals.
    if (ringModel.quaternion.dot(targetQuat) < 0) targetQuat.set(-targetQuat.x, -targetQuat.y, -targetQuat.z, -targetQuat.w);
    if (occluderMesh.quaternion.dot(targetOccQuat) < 0) targetOccQuat.set(-targetOccQuat.x, -targetOccQuat.y, -targetOccQuat.z, -targetOccQuat.w);
    ringModel.quaternion.slerp(targetQuat, 0.45);
    occluderMesh.quaternion.slerp(targetOccQuat, 0.45);
    ringModel.scale.lerp(targetScale, 0.62);
    // Scintillation — rotate the gem shader's env-twist 2× the ring's current
    // rotation so sparkle moves faster than the hand (diamond-like dynamic fire).
    updateDiamondEnvTwist(ringModel, ringModel.quaternion);
    occluderMesh.scale.lerp(targetOccScale, 0.62);
    occluderMesh.visible = true;
  } else if (occluderMesh) {
    occluderMesh.visible = false;
    hideBlockerOccluders();
  }
  // Re-bake the diamond's reflection cube at the ring's current position.
  // Runs BEFORE the alpha snapshot + composer so the sceneRT and composer both
  // see the gem with fresh metal reflections captured this frame.
  updateDiamondReflectionCube();

  // Smart skipping: if no jewelry is visible, skip the heavy composer/render logic.
  // The video backdrop is handled by the browser's video element in CSS fallback,
  // but here it's also a mesh. We check if the hand/face trackers have anything to show.
  const ringVisible = ringModel && isHandPresent;
  const earringVisible = typeof isEarringVisible === 'function' && isEarringVisible();

  if (PERF_TIER.usePostFX) {
    if (!ringVisible && !earringVisible) {
      // Nothing to bloom, skip composer and render the scene once (just video backdrop)
      renderer.setRenderTarget(null);
      renderer.clear();
      renderer.render(scene, camera);
    } else {
      // Double-render collapse: point the alphaRestorePass to the composer's 
      // internal texture instead of a separate sceneRT render.
      // This halves the triangle/draw-call cost on desktop.
      alphaRestorePass.uniforms.tScene.value = composer.renderTarget2.texture;

      // Update time uniform for film noise
      alphaRestorePass.uniforms.time.value = nowMs / 1000.0;

      composer.render();
    }
  } else {
    // Mobile fast path: render the scene directly.
    renderer.setRenderTarget(null);
    renderer.clear();
    renderer.render(scene, camera);
  }

  fpsFrameCount++;
  const fpsElapsed = nowMs - fpsLastSampleTime;
  if (fpsElapsed >= 500) {
    const fps = (fpsFrameCount * 1000) / fpsElapsed;
    fpsMeterElement.textContent = `FPS: ${Math.round(fps)}`;
    fpsFrameCount = 0;
    fpsLastSampleTime = nowMs;
  }
}
animate();

const cameraConstraints = isMobile
  ? {
    facingMode: 'user',
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 30, min: 24 }
  }
  : {
    facingMode: 'user',
    width: { ideal: 640 }, // Drop to 640 to speed up MediaPipe landmark detection
    height: { ideal: 480 }
  };

console.log("[Init] Requesting Camera...");
if (loadingElement) loadingElement.innerText = "Requesting Camera...";

navigator.mediaDevices.getUserMedia({ video: cameraConstraints })
  .then(stream => {
    console.log("[Init] Camera Granted.");
    videoElement.srcObject = stream;
    videoElement.onloadedmetadata = () => {
      console.log("[Init] Video Metadata Loaded.");
      videoElement.play();
    };
  })
  .catch(err => {
    console.error("[Init] Camera Error:", err);
    if (loadingElement) {
      if (err.name === 'NotAllowedError') {
        loadingElement.innerText = "Camera Permission Denied";
      } else if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
        loadingElement.innerText = "HTTPS Required for Camera";
      } else {
        loadingElement.innerText = "Camera Error: " + err.name;
      }
    }
  });
