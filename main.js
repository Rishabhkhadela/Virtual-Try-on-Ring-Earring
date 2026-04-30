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
import { initializeHandOccluder } from './handOccluder.js';

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
const HAND_OCCLUDER_ASSETS = {
  Left: 'assets/hand/LeftHandAndroidXRVisual.glb',
  Right: 'assets/hand/RightHandAndroidXRVisual.glb',
};
const handOccluders = new Map();

for (const [handedness, assetPath] of Object.entries(HAND_OCCLUDER_ASSETS)) {
  initializeHandOccluder({
    scene,
    assetPath,
    procedural: true,
    debug: !!document.getElementById('showHandMesh')?.checked,
    mapToScene: (point2D, out) => mapToOrthographicSpaceInto(out, point2D),
    log,
  }).then((controller) => {
    handOccluders.set(handedness, controller);
  }).catch((error) => {
    log(`[HandOccluder] Failed to initialize ${handedness} hand: ${error?.message || error}`);
  });
}

function getReportedHandedness(results) {
  return results?.handednesses?.[0]?.[0]?.categoryName === 'Right' ? 'Right' : 'Left';
}

function getHandOccluderForResults(results) {
  return handOccluders.get(getReportedHandedness(results)) || null;
}

function hideHandOccluders(except = null) {
  for (const controller of handOccluders.values()) {
    if (controller !== except) controller.hide();
  }
}

// =========================================================================
// PR-α: Rigged-hand bone driver (debug-only, additive)
// -------------------------------------------------------------------------
// Loads LeftHandAndroidXRVisual.glb / RightHandAndroidXRVisual.glb,
// resolves their finger bones by name, and drives bone world positions
// from MediaPipe landmarks every frame. Renders as wireframe so it can
// be visually verified against the real hand before we depend on it for
// occlusion (PR-β) or ring parenting (PR-γ).
//
// IMPORTANT: This system does NOT replace the existing occluder, does NOT
// change ring placement, and does NOT mutate any tracking math. It only
// renders an additional debug mesh when "Show Rigged Hand" is checked.
// =========================================================================

// Bone-name → MediaPipe landmark index map. The hand GLBs follow a
// consistent naming convention: {prefix}_{Finger}{Joint}, prefix = L/R.
// The verts are skinned to these bones, so setting each bone's WORLD
// position to the corresponding landmark's scene position deforms the
// surface to follow the real hand.
const RIGGED_BONE_TO_LANDMARK = {
  // Wrist
  Wrist: 0,
  // Thumb: MP=1, IP=2, DIP=3, TIP=4
  ThumbMetacarpal: 1,
  ThumbProximal: 2,
  ThumbDistal: 3,
  // Index: MCP=5, PIP=6, DIP=7, TIP=8
  IndexMetacarpal: 5,
  IndexProximal: 6,
  IndexIntermediate: 7,
  IndexDistal: 8,
  // Middle: 9 / 10 / 11 / 12
  MiddleMetacarpal: 9,
  MiddleProximal: 10,
  MiddleIntermediate: 11,
  MiddleDistal: 12,
  // Ring: 13 / 14 / 15 / 16
  RingMetacarpal: 13,
  RingProximal: 14,
  RingIntermediate: 15,
  RingDistal: 16,
  // Little / pinky: 17 / 18 / 19 / 20
  LittleMetacarpal: 17,
  LittleProximal: 18,
  LittleIntermediate: 19,
  LittleDistal: 20,
};

const RIGGED_HAND_ASSETS = {
  Left: 'assets/hand/LeftHandAndroidXRVisual.glb',
  Right: 'assets/hand/RightHandAndroidXRVisual.glb',
};
const riggedHands = new Map(); // handedness -> { root, boneByLandmark[], visible }

function _resolveBoneByLandmark(root, prefix) {
  const map = new Array(21).fill(null);
  const boneNames = Object.keys(RIGGED_BONE_TO_LANDMARK);
  root.traverse((obj) => {
    if (!obj.isBone) return;
    for (const suffix of boneNames) {
      // Match exact name "L_RingProximal" / "R_RingProximal".
      // WeightFix bones are skipped — we only drive the primary chain.
      if (obj.name === `${prefix}_${suffix}`) {
        map[RIGGED_BONE_TO_LANDMARK[suffix]] = obj;
        return;
      }
    }
  });
  return map;
}

function _initializeRiggedHand(handedness, assetPath) {
  const prefix = handedness === 'Right' ? 'R' : 'L';
  gltfLoader.load(
    assetPath,
    (gltf) => {
      const root = gltf.scene;
      const boneByLandmark = _resolveBoneByLandmark(root, prefix);
      const resolved = boneByLandmark.filter(Boolean).length;
      if (resolved < 15) {
        log(`[RiggedHand] ${handedness}: only resolved ${resolved}/21 bones — names may not match`);
      }

      // Wireframe debug material for every skinned mesh in the hand. We
      // wrap each material so the rigged hand stays visible and obviously
      // distinct from the real GLB shading, but doesn't write color or
      // depth (won't fight the existing renderer state).
      root.traverse((obj) => {
        if (obj.isMesh || obj.isSkinnedMesh) {
          obj.material = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            wireframe: true,
            transparent: true,
            opacity: 0.55,
            depthWrite: false,
            depthTest: false,
          });
          obj.renderOrder = 9999; // draw on top so debug is always visible
        }
      });

      root.visible = false; // hidden until "Show Rigged Hand" is toggled
      scene.add(root);

      // Capture the authored palm-span (distance between index MCP and
      // little MCP bones at bind pose). Used as the denominator for
      // group-level scaling: liveSpan / authoredSpan = scale factor.
      let authoredPalmSpan = 1;
      const indexMcp = boneByLandmark[5];
      const pinkyMcp = boneByLandmark[17];
      if (indexMcp && pinkyMcp) {
        indexMcp.updateWorldMatrix(true, false);
        pinkyMcp.updateWorldMatrix(true, false);
        const wA = new THREE.Vector3();
        const wB = new THREE.Vector3();
        indexMcp.getWorldPosition(wA);
        pinkyMcp.getWorldPosition(wB);
        authoredPalmSpan = wA.distanceTo(wB) || 1;
      }

      riggedHands.set(handedness, { root, boneByLandmark, prefix, resolved, authoredPalmSpan });
      // Expose to window so DevTools can inspect even though `riggedHands`
      // is module-scoped. Also lets us write one-line `__rigDiag()` tests.
      window.riggedHands = riggedHands;
      log(`[RiggedHand] ${handedness} initialized (${resolved}/21 bones, authoredPalmSpan=${authoredPalmSpan.toFixed(4)})`);
    },
    undefined,
    (err) => {
      log(`[RiggedHand] Failed to load ${handedness}: ${err?.message || err}`);
    }
  );
}
// NOTE: actual loader invocation is deferred to AFTER `gltfLoader` is
// declared (a few lines below). The function is hoisted as a `function`
// declaration, but it captures `gltfLoader` lexically — so we have to
// run the loop from a position where the binding exists.

const _cbShowRiggedHand = document.getElementById('showRiggedHand');

// Reusable scratch — never per-frame allocate.
const _riggedTmpV = new THREE.Vector3();
const _riggedTmpV2 = new THREE.Vector3();
const _riggedTmpV3 = new THREE.Vector3();
const _riggedTmpQ = new THREE.Quaternion();
const _riggedTmpM = new THREE.Matrix4();
const _riggedAxisX = new THREE.Vector3();
const _riggedAxisY = new THREE.Vector3();
const _riggedAxisZ = new THREE.Vector3();

// Bone-to-bone "child landmark" map. Used in the per-bone local-rotation
// step: each bone aims its bind-axis at its child landmark.
const RIGGED_BONE_CHILD = {
  Wrist: 9,                  // wrist points at middle MCP (best palm reference)
  ThumbMetacarpal: 2,
  ThumbProximal: 3,
  ThumbDistal: 4,
  IndexMetacarpal: 6,
  IndexProximal: 7,
  IndexIntermediate: 8,
  MiddleMetacarpal: 10,
  MiddleProximal: 11,
  MiddleIntermediate: 12,
  RingMetacarpal: 14,
  RingProximal: 15,
  RingIntermediate: 16,
  LittleMetacarpal: 18,
  LittleProximal: 19,
  LittleIntermediate: 20,
};

function _hideAllRiggedHands(except = null) {
  for (const entry of riggedHands.values()) {
    if (entry !== except) entry.root.visible = false;
  }
}

// Capture each bone's bind-pose local-axis-toward-child ONCE per asset.
// Stored on the entry as `bindAimAxis: Map<bone, Vector3>`. We then drive
// each bone by rotating that axis to point at the child landmark in the
// bone's parent-local frame — this is the standard puppeteering approach
// and respects the inverseBindMatrix.
function _ensureBindAimCache(entry) {
  if (entry.bindAimAxis) return;
  entry.bindAimAxis = new Map();
  // Build a quick name→bone lookup for child resolution.
  const byName = new Map();
  entry.root.traverse((o) => { if (o.isBone) byName.set(o.name, o); });
  const prefix = entry.prefix;
  // For each bone we drive, compute the local-space direction from this
  // bone's bind position to its child bone's bind position, expressed in
  // THIS bone's local frame.
  for (const [suffix, childIdx] of Object.entries(RIGGED_BONE_CHILD)) {
    const bone = byName.get(`${prefix}_${suffix}`);
    if (!bone) continue;
    // Find bind-time child bone by suffix lookup using the kinematic chain.
    const childSuffix = _suffixForLandmark(childIdx);
    if (!childSuffix) continue;
    const child = byName.get(`${prefix}_${childSuffix}`);
    if (!child) continue;
    // Make sure world matrices reflect the bind pose.
    bone.updateWorldMatrix(true, false);
    child.updateWorldMatrix(true, false);
    // Direction in WORLD space.
    const childWorld = new THREE.Vector3();
    const boneWorld = new THREE.Vector3();
    child.getWorldPosition(childWorld);
    bone.getWorldPosition(boneWorld);
    childWorld.sub(boneWorld).normalize();
    // Convert to the BONE's local space (so we can rotate it later).
    const inv = new THREE.Matrix4().copy(bone.matrixWorld).invert();
    // We only want the rotation part, not translation, so use a transformDirection.
    childWorld.transformDirection(inv);
    childWorld.normalize();
    entry.bindAimAxis.set(bone, childWorld);
    // Also stash the original quaternion so we can compose deltas.
    entry.bindQuat = entry.bindQuat || new Map();
    entry.bindQuat.set(bone, bone.quaternion.clone());
  }
}

// Reverse lookup: given a landmark index, get the bone-name suffix.
function _suffixForLandmark(idx) {
  for (const [suffix, mappedIdx] of Object.entries(RIGGED_BONE_TO_LANDMARK)) {
    if (mappedIdx === idx) return suffix;
  }
  return null;
}

// Compute the rigged group's WORLD pose from the live landmarks: anchor it
// at the wrist, scale it by palm-span vs. authored palm-span, and orient it
// so the palm-normal matches the live palm.
function _computeRiggedGroupPose(landmarks, entry, outPos, outQuat, outScale) {
  // Live palm reference points: wrist (0), index MCP (5), pinky MCP (17).
  mapToOrthographicSpaceInto(_riggedTmpV, landmarks[0]);   // wrist
  mapToOrthographicSpaceInto(_riggedTmpV2, landmarks[5]);  // index MCP
  mapToOrthographicSpaceInto(_riggedTmpV3, landmarks[17]); // pinky MCP

  // Position: wrist.
  outPos.copy(_riggedTmpV);

  // Build palm basis from these three points:
  //   palmAcross = (pinkyMCP - indexMCP).normalize()
  //   palmAlong  = ((indexMCP+pinkyMCP)*0.5 - wrist).normalize()
  //   palmNormal = palmAlong × palmAcross
  _riggedAxisX.subVectors(_riggedTmpV3, _riggedTmpV2).normalize(); // across
  _riggedAxisY.copy(_riggedTmpV2).add(_riggedTmpV3).multiplyScalar(0.5).sub(_riggedTmpV).normalize(); // along
  _riggedAxisZ.crossVectors(_riggedAxisY, _riggedAxisX).normalize(); // normal
  // Re-orthogonalize across so the basis is clean.
  _riggedAxisX.crossVectors(_riggedAxisZ, _riggedAxisY).normalize();

  // Scale: live palm-span vs. asset's authored palm-span (use bbox.x as a
  // reasonable proxy; we cached it on entry at load time).
  const livePalmSpan = _riggedTmpV3.distanceTo(_riggedTmpV2);
  const authoredPalmSpan = entry.authoredPalmSpan || 1;
  const s = livePalmSpan / authoredPalmSpan;
  outScale.set(s, s, s);

  // Rotation: rotation matrix from authored basis (which we treat as
  // identity — the rigged hand was authored with +X right, +Y up, +Z out)
  // to the live basis. The asset authored axes happen to map: model Y →
  // along finger direction (palm-along), model X → across, model Z → normal.
  _riggedTmpM.makeBasis(_riggedAxisX, _riggedAxisY, _riggedAxisZ);
  outQuat.setFromRotationMatrix(_riggedTmpM);
}

// Drive the rigged hand from MediaPipe landmarks. Pure debug feedback.
//
// Path B: this driver is intentionally dormant. The infrastructure (asset
// load, bone resolution, HUD readout) stays in tree so we can resume
// Strategy 2 later, but the per-frame update is a no-op unless the explicit
// `window.__enableRiggedHand` flag is set in DevTools. This keeps the
// existing landmark-driven ring placement as the sole source of truth and
// lets us focus tuning on the existing system without rigged-hand work
// interfering.
function updateRiggedHand(results) {
  if (!window.__enableRiggedHand) {
    _hideAllRiggedHands();
    return;
  }
  if (!_cbShowRiggedHand || !_cbShowRiggedHand.checked) {
    _hideAllRiggedHands();
    return;
  }
  if (!results?.landmarks || results.landmarks.length === 0) {
    _hideAllRiggedHands();
    return;
  }
  const handedness = getReportedHandedness(results);
  const entry = riggedHands.get(handedness);
  if (!entry) return;
  _hideAllRiggedHands(entry);

  const landmarks = results.landmarks[0];
  const { root, boneByLandmark } = entry;
  _ensureBindAimCache(entry);
  root.visible = true;

  // ---- Stage A: place the whole rigged group at the live hand pose. ----
  // Group-level transform handles wrist position, palm orientation, and
  // global scale. The skinning shader keeps using the bind matrix correctly
  // because we never touched bone.position.
  _computeRiggedGroupPose(landmarks, entry, _riggedTmpV, _riggedTmpQ, _riggedAxisX);
  root.position.copy(_riggedTmpV);
  root.quaternion.copy(_riggedTmpQ);
  root.scale.copy(_riggedAxisX); // outScale was reused into _riggedAxisX

  // ---- Stage B: rotate finger bones to match landmark angles. ----
  // For each driven bone we have its bind-pose aim axis (local frame).
  // We compute the desired aim axis from the landmark direction
  // (parent→child) expressed in this bone's parent-local frame, then
  // build a delta rotation that maps bind→desired and prepend it to the
  // bind-pose quaternion.
  for (const [suffix, childLandmarkIdx] of Object.entries(RIGGED_BONE_CHILD)) {
    const boneLandmarkIdx = RIGGED_BONE_TO_LANDMARK[suffix];
    const bone = boneByLandmark[boneLandmarkIdx];
    if (!bone) continue;
    const bindAim = entry.bindAimAxis.get(bone);
    const bindQuat = entry.bindQuat.get(bone);
    if (!bindAim || !bindQuat) continue;

    // Live direction in WORLD scene space.
    mapToOrthographicSpaceInto(_riggedTmpV, landmarks[boneLandmarkIdx]);
    mapToOrthographicSpaceInto(_riggedTmpV2, landmarks[childLandmarkIdx]);
    _riggedTmpV2.sub(_riggedTmpV).normalize();

    // Convert to bone's parent-local space so we can apply via local rot.
    const parent = bone.parent;
    if (parent) {
      parent.updateWorldMatrix(true, false);
      _riggedTmpM.copy(parent.matrixWorld).invert();
      _riggedTmpV2.transformDirection(_riggedTmpM);
      _riggedTmpV2.normalize();
    }

    // Apply bindQuat to bindAim to get bindAim in parent-local space.
    _riggedTmpV3.copy(bindAim).applyQuaternion(bindQuat);

    // Build delta quaternion that maps current bindAim direction → live.
    _riggedTmpQ.setFromUnitVectors(_riggedTmpV3, _riggedTmpV2);

    // Final bone rotation = delta * bindQuat (apply delta in parent space).
    bone.quaternion.copy(_riggedTmpQ).multiply(bindQuat);
  }

  root.updateMatrixWorld(true);
}

// One-liner diagnostic: paste `__rigDiag()` into DevTools console (after
// typing 'allow pasting' once per session). Returns the load/visibility/
// bbox state for whichever rigged hand is in scene.
window.__rigDiag = function () {
  const all = {};
  for (const [hd, entry] of riggedHands.entries()) {
    const bbox = new THREE.Box3().setFromObject(entry.root);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const meshes = [];
    entry.root.traverse((o) => {
      if (o.isSkinnedMesh) meshes.push({ name: o.name, visible: o.visible, frustumCulled: o.frustumCulled });
    });
    all[hd] = {
      prefix: entry.prefix,
      resolvedBones: entry.resolved,
      rootVisible: entry.root.visible,
      inScene: !!entry.root.parent,
      bbox_size: size.toArray().map((n) => +n.toFixed(4)),
      bbox_min: bbox.min.toArray().map((n) => +n.toFixed(4)),
      bbox_max: bbox.max.toArray().map((n) => +n.toFixed(4)),
      meshes,
    };
  }
  console.table(all);
  return all;
};

// --- Load GLTF Model ---
let ringModel = null;
const gltfLoader = new THREE.GLTFLoader();

// PR-α deferred initialization: now that gltfLoader exists, kick off the
// rigged-hand asset loads. Function was declared earlier; only the
// invocation is deferred to here so the lexical `gltfLoader` capture is valid.
for (const [hd, path] of Object.entries(RIGGED_HAND_ASSETS)) _initializeRiggedHand(hd, path);

// --- Mask.glb diagnostic loader (Strategy 2 prep) --------------------------
// One-shot: load assets/Mask.glb, print its structure, then drop the
// reference. NOT added to the scene. Tells us whether the asset is a finger
// sleeve, a multi-segment finger, or a whole hand so we can pick the right
// rigging strategy. Output is grouped under [Mask.glb] in the console — open
// DevTools and copy the block back into chat.
gltfLoader.load(
  'assets/Mask.glb',
  (gltf) => {
    const root = gltf.scene;
    const bbox = new THREE.Box3().setFromObject(root);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    bbox.getSize(size);
    bbox.getCenter(center);

    const nodes = [];
    const meshes = [];
    const materials = new Set();
    let totalVerts = 0;
    let totalTris = 0;

    root.traverse((obj) => {
      const indent = '  '.repeat(_depthOf(obj, root));
      const tag = obj.isMesh ? 'Mesh' : (obj.isBone ? 'Bone' : (obj.isSkinnedMesh ? 'SkinnedMesh' : (obj.type || 'Object3D')));
      nodes.push(`${indent}${tag}: ${obj.name || '(unnamed)'}`);
      if (obj.isMesh || obj.isSkinnedMesh) {
        const geom = obj.geometry;
        const v = geom?.attributes?.position?.count || 0;
        const i = geom?.index?.count || 0;
        const tris = i ? i / 3 : v / 3;
        totalVerts += v;
        totalTris += tris;
        meshes.push({
          name: obj.name || '(unnamed)',
          verts: v,
          tris,
          skinned: !!obj.isSkinnedMesh,
          material: obj.material?.name || obj.material?.type || '?',
        });
        if (obj.material) materials.add(obj.material?.name || obj.material?.type);
      }
    });

    console.groupCollapsed('%c[Mask.glb] structure report', 'color:#7fffd4;font-weight:bold');
    console.log('Bounding box (local units):');
    console.log('  size  :', size.x.toFixed(4), size.y.toFixed(4), size.z.toFixed(4));
    console.log('  center:', center.x.toFixed(4), center.y.toFixed(4), center.z.toFixed(4));
    console.log('  min   :', bbox.min.x.toFixed(4), bbox.min.y.toFixed(4), bbox.min.z.toFixed(4));
    console.log('  max   :', bbox.max.x.toFixed(4), bbox.max.y.toFixed(4), bbox.max.z.toFixed(4));
    const longest = Math.max(size.x, size.y, size.z);
    const shortest = Math.min(size.x, size.y, size.z);
    const ratio = shortest > 1e-6 ? longest / shortest : 0;
    console.log(`  long/short ratio: ${ratio.toFixed(2)}  (>5 = sleeve, ~2-3 = finger, <2 = palm/hand)`);
    console.log('');
    console.log('Mesh summary:');
    console.log('  meshes      :', meshes.length);
    console.log('  total verts :', totalVerts);
    console.log('  total tris  :', totalTris);
    console.log('  materials   :', Array.from(materials).join(', ') || '(none)');
    console.log('  skinned?    :', meshes.some((m) => m.skinned) ? 'yes' : 'no');
    console.log('');
    console.log('Per-mesh:');
    meshes.forEach((m) => console.log(`  - ${m.name}: ${m.verts} v / ${m.tris} t  [${m.material}]${m.skinned ? '  (skinned)' : ''}`));
    console.log('');
    console.log('Node hierarchy:');
    nodes.forEach((n) => console.log(n));
    console.log('');
    console.log('Animations:', gltf.animations?.length || 0);
    console.log('Skins      :', gltf.parser?.json?.skins?.length || 0);
    console.log('');
    console.log('VERDICT (heuristic): ' +
      (ratio > 5 ? 'looks like a FINGER SLEEVE (single segment cylinder/tube)'
        : ratio > 2.5 ? 'looks like a FINGER or finger group'
          : 'looks like a HAND or palm region'));
    console.groupEnd();

    // Stash on window so the next session can also inspect interactively.
    window.__maskGlbReport = { size, center, bbox, meshes, ratio, nodes, gltf };
  },
  undefined,
  (err) => {
    console.warn('[Mask.glb] failed to load:', err?.message || err);
  }
);
function _depthOf(obj, root) {
  let d = 0;
  let cur = obj;
  while (cur && cur !== root) { d++; cur = cur.parent; }
  return d;
}

// --- Hand GLB diagnostic loader (Strategy 2 viability check) --------------
// Inspects the existing rigged hand assets at assets/hand/ to see whether
// they expose bones we can drive directly from MediaPipe landmarks. Reports
// bone count + names, mesh count, skinning state, and bbox proportions.
// One-shot, console-only, no scene changes.
function _diagnoseHandGlb(path) {
  gltfLoader.load(
    path,
    (gltf) => {
      const root = gltf.scene;
      const bbox = new THREE.Box3().setFromObject(root);
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      bbox.getSize(size);
      bbox.getCenter(center);

      const bones = [];
      const meshes = [];
      const skinnedMeshes = [];
      const materials = new Set();
      let totalVerts = 0;
      let totalTris = 0;

      root.traverse((obj) => {
        if (obj.isBone) {
          bones.push({
            name: obj.name || '(unnamed)',
            depth: _depthOf(obj, root),
            parent: obj.parent?.name || '(none)',
          });
        }
        if (obj.isMesh || obj.isSkinnedMesh) {
          const geom = obj.geometry;
          const v = geom?.attributes?.position?.count || 0;
          const i = geom?.index?.count || 0;
          const tris = i ? i / 3 : v / 3;
          totalVerts += v;
          totalTris += tris;
          const entry = {
            name: obj.name || '(unnamed)',
            verts: v,
            tris,
            skinned: !!obj.isSkinnedMesh,
            material: obj.material?.name || obj.material?.type || '?',
          };
          meshes.push(entry);
          if (obj.isSkinnedMesh) skinnedMeshes.push(entry);
          if (obj.material) materials.add(obj.material?.name || obj.material?.type);
        }
      });

      const tag = path.split('/').pop();
      console.groupCollapsed(`%c[${tag}] hand structure report`, 'color:#ffd47f;font-weight:bold');
      console.log('Bounding box (local units):');
      console.log('  size  :', size.x.toFixed(4), size.y.toFixed(4), size.z.toFixed(4));
      console.log('  center:', center.x.toFixed(4), center.y.toFixed(4), center.z.toFixed(4));
      console.log('');
      console.log('Mesh summary:');
      console.log('  meshes        :', meshes.length);
      console.log('  skinned meshes:', skinnedMeshes.length);
      console.log('  total verts   :', totalVerts);
      console.log('  total tris    :', totalTris);
      console.log('  materials     :', Array.from(materials).join(', ') || '(none)');
      console.log('');
      console.log('Bones / skeleton:');
      console.log('  bone count    :', bones.length);
      console.log('  skins (parser):', gltf.parser?.json?.skins?.length || 0);
      if (bones.length) {
        console.log('  bone list (name -> parent):');
        bones.forEach((b) => console.log(`    [${b.depth}] ${b.name}  <- ${b.parent}`));
      } else {
        console.log('  (no bones detected — mesh is static)');
      }
      console.log('');
      console.log('Animations:', gltf.animations?.length || 0);
      console.log('');
      const verdict = bones.length >= 15
        ? 'RIGGED: enough bones to drive from MediaPipe (Strategy 2 viable)'
        : bones.length > 0
          ? 'PARTIALLY RIGGED: some bones, may need careful mapping'
          : 'STATIC: no bones — would need procedural deform, same as current handOccluder.js';
      console.log('VERDICT:', verdict);
      console.groupEnd();

      window.__handGlbReports = window.__handGlbReports || {};
      window.__handGlbReports[tag] = { size, center, bbox, meshes, bones, skinnedMeshes, gltf };
    },
    undefined,
    (err) => {
      console.warn(`[${path}] failed to load:`, err?.message || err);
    }
  );
}
_diagnoseHandGlb('assets/hand/LeftHandAndroidXRVisual.glb');
_diagnoseHandGlb('assets/hand/RightHandAndroidXRVisual.glb');

// Available ring models. Each `preset` captures the slider values that make
// the model sit correctly on the finger — applied on load/swap so the user
// doesn't have to re-tune rotX/rotY/rotZ/scale for every ring.
const RING_MODELS = [
  {
    id: 'ring',
    label: 'Default Ring',
    path: 'assets/ring.glb',
    preset: { rotX: 164, rotY: 90, rotZ: -67, scale: 0.75, anchorT: 0.62, offsetX: 0, offsetY: 0, offsetZ: 0 }
  },
  {
    id: 'Rotation Test 01',
    label: 'Rotation Test 01',
    path: 'assets/Rotation Test 01.glb',
    preset: { rotX: 73, rotY: 90, rotZ: -67, scale: 0.75, anchorT: 0.5, offsetX: 0, offsetY: 0, offsetZ: 0 }
  },
  {
    id: 'Rotation Test 02',
    label: 'Rotation Test 02',
    path: 'assets/Rotation Test 02.glb',
    preset: { rotX: 73, rotY: 90, rotZ: -67, scale: 0.75, anchorT: 0.5, offsetX: 0, offsetY: 0, offsetZ: 0 }
  },
  {
    id: 'Rotation Test 04',
    label: 'Rotation Test 04',
    path: 'assets/Rotation Test 04.glb',
    preset: { rotX: 164, rotY: 90, rotZ: -67, scale: 1.0, anchorT: 0.5, offsetX: 0, offsetY: 0, offsetZ: 0 }
  },
  {
    id: 'Rotation Test 05',
    label: 'Rotation Test 05',
    path: 'assets/Rotation Test 05.glb',
    preset: { rotX: 44, rotY: 86, rotZ: 39, scale: 0.8, anchorT: 0.5, offsetX: 0, offsetY: 0, offsetZ: 0 }
  },
  {
    id: '01 Clean',
    label: '01 Clean',
    path: 'assets/01 Clean.glb',
    preset: { rotX: 101, rotY: 2, rotZ: -2, scale: 0.8, anchorT: 0.5, offsetX: 0, offsetY: 0, offsetZ: 0 }
  },
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
    ['ringAnchorT', 'numRingAnchorT', preset.anchorT ?? 0.5],
    ['ringOffsetX', 'numRingOffsetX', preset.offsetX ?? 0],
    ['ringOffsetY', 'numRingOffsetY', preset.offsetY ?? 0],
    ['ringOffsetZ', 'numRingOffsetZ', preset.offsetZ ?? 0],
  ];
  for (const [rangeId, numId, value] of pairs) {
    if (value === undefined) continue;
    const range = document.getElementById(rangeId);
    const num = document.getElementById(numId);
    if (range) range.value = value;
    if (num) num.value = value;
  }
}

window.captureActiveRingPreset = function captureActiveRingPreset() {
  const entry = RING_MODELS[currentRingIndex];
  if (!entry) return null;
  const preset = {
    rotX: Number(_slRotX?.value ?? entry.preset.rotX),
    rotY: Number(_slRotY?.value ?? entry.preset.rotY),
    rotZ: Number(_slRotZ?.value ?? entry.preset.rotZ),
    scale: Number(_slScaleBase?.value ?? entry.preset.scale),
    anchorT: Number(_slRingAnchorT?.value ?? entry.preset.anchorT ?? 0.5),
    offsetX: Number(_slRingOffsetX?.value ?? entry.preset.offsetX ?? 0),
    offsetY: Number(_slRingOffsetY?.value ?? entry.preset.offsetY ?? 0),
    offsetZ: Number(_slRingOffsetZ?.value ?? entry.preset.offsetZ ?? 0),
  };
  console.log(`[RingPreset:${entry.id}]`, JSON.stringify(preset));
  console.table(preset);
  return preset;
};

function percentile(sortedValues, t) {
  if (!sortedValues.length) return 0;
  const idx = Math.min(sortedValues.length - 1, Math.max(0, (sortedValues.length - 1) * t));
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedValues[lo];
  return sortedValues[lo] + (sortedValues[hi] - sortedValues[lo]) * (idx - lo);
}

function collectModelVertices(root) {
  const vertices = [];
  root.updateMatrixWorld(true);
  root.traverse((node) => {
    if (!node.isMesh || !node.geometry?.attributes?.position) return;
    const pos = node.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const vertex = new THREE.Vector3().fromBufferAttribute(pos, i);
      node.localToWorld(vertex);
      vertices.push(vertex);
    }
  });
  return vertices;
}

function getRobustVertexCenter(vertices) {
  if (!vertices.length) return new THREE.Vector3();

  const xs = vertices.map((v) => v.x).sort((a, b) => a - b);
  const ys = vertices.map((v) => v.y).sort((a, b) => a - b);
  const zs = vertices.map((v) => v.z).sort((a, b) => a - b);

  return new THREE.Vector3(
    (percentile(xs, 0.15) + percentile(xs, 0.85)) * 0.5,
    (percentile(ys, 0.15) + percentile(ys, 0.85)) * 0.5,
    (percentile(zs, 0.15) + percentile(zs, 0.85)) * 0.5
  );
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
  hideHandOccluders();
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
  const vertices = collectModelVertices(rawModel);

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

const showHandMeshToggle = document.getElementById('showHandMesh');
const toggleHandMeshBtn = document.getElementById('toggleHandMeshBtn');

function syncHandMeshButtonState() {
  if (!toggleHandMeshBtn || !showHandMeshToggle) return;
  const enabled = !!showHandMeshToggle.checked;
  toggleHandMeshBtn.classList.toggle('active', enabled);
  toggleHandMeshBtn.setAttribute('aria-pressed', enabled ? 'true' : 'false');
  toggleHandMeshBtn.textContent = enabled ? 'Skin Mesh On' : 'Skin Mesh Off';
}

if (showHandMeshToggle && toggleHandMeshBtn) {
  toggleHandMeshBtn.addEventListener('click', () => {
    showHandMeshToggle.checked = !showHandMeshToggle.checked;
    syncHandMeshButtonState();
  });
  showHandMeshToggle.addEventListener('change', syncHandMeshButtonState);
  syncHandMeshButtonState();
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
const _tmpDirSurface = new THREE.Vector3();
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
const _slRingAnchorT = document.getElementById('ringAnchorT');
const _slRingOffsetX = document.getElementById('ringOffsetX');
const _slRingOffsetY = document.getElementById('ringOffsetY');
const _slRingOffsetZ = document.getElementById('ringOffsetZ');
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

// FIX D: hysteresis state for the occlusion-hide guards (world-length ratio
// + neighbor-finger forward check). Without hysteresis a single noisy frame
// flips visibility, producing visible flicker. Need K consecutive bad frames
// to hide, M consecutive good frames to show again.
let _hideBadStreak = 0;
let _hideGoodStreak = 0;
let _hideActive = false;
const HIDE_REQUIRES_BAD_FRAMES = 2;   // borderline single-frame noise won't hide
const HIDE_RELEASES_GOOD_FRAMES = 4;  // need a clear streak to bring ring back
let _dbgHideThisFrame = false;
let _dbgForwardCount = 0;
let _dbgFingerRatio = 0;

// scaleBoost: per-pose calibration multiplier. Tuned conservatively from
// HUD-evidence screenshots (#42–#48):
//   - back/side-on poses already render correctly at 1.00
//   - 'front' (palm or back to camera) was slightly small on flat poses with
//     palmFacing > 0.95 → +10% lifts those without breaking other fronts
//   - 'side' and 'foreshortened' get small boosts because the foreshorten
//     compensation in the occluder block doesn't propagate to ring scale
//   - 'fist' stays at 1.00 because curled fingers project very short and
//     any boost compounds the existing radius->scale path into oversize
// All values are conservative; tweak by ±0.05 increments if a specific
// pose shows a visible mismatch in side-by-side testing.
// VTO PARITY PASS: side-by-side with ijewel showed our ring floating ABOVE
// the finger and slightly under-sized on back/knuckle poses. Two changes:
//   1) surfaceLift slashed across all poses. The previous values pushed the
//      ring center 34-55% of a finger-radius FORWARD of the visible 2D
//      midpoint — which is already on the finger surface from the camera's
//      POV. Result: ring hovered above skin instead of wrapping. New values
//      keep a small forward bias (so the ring sits "on top of" the finger
//      rather than embedded in it) without floating.
//   2) scaleBoost raised on back and fist where ijewel's render reads as
//      "sized to the knuckle" while ours read as "smaller than the finger".
const RING_POSE_FIT = {
  front: { anchorBias: -0.10, axisX: 0.00, axisY: 0.22, surfaceLift: 0.12, scaleRadiusBlend: 0.12, screenBlend: 0.06, scaleBoost: 1.10 },
  back: { anchorBias: -0.08, axisX: 0.00, axisY: 0.20, surfaceLift: 0.10, scaleRadiusBlend: 0.10, screenBlend: 0.06, scaleBoost: 1.10 },
  side: { anchorBias: -0.04, axisX: 0.02, axisY: 0.18, surfaceLift: 0.20, scaleRadiusBlend: 0.18, screenBlend: 0.16, scaleBoost: 1.05 },
  foreshortened: { anchorBias: -0.14, axisX: 0.00, axisY: 0.24, surfaceLift: 0.22, scaleRadiusBlend: 0.22, screenBlend: 0.18, scaleBoost: 1.05 },
  fist: { anchorBias: -0.16, axisX: 0.00, axisY: 0.26, surfaceLift: 0.18, scaleRadiusBlend: 0.18, screenBlend: 0.12, scaleBoost: 1.10 }
};

function _setPoseHintVisible(show) {
  if (!_poseHintEl) return;
  if (show) _poseHintEl.classList.add('visible');
  else _poseHintEl.classList.remove('visible');
}

// --- Pose diagnostic HUD ---------------------------------------------------
// Pure read-only telemetry for debugging why a given pose was classified the
// way it was. Toggled by the #showPoseDebug checkbox in index.html. Does NOT
// touch any tracking state; only formats and prints values that the
// classifier already produced. Throttled to ~10fps so it doesn't bog the
// layout in the hot render loop.
const _poseDebugHudEl = document.getElementById('poseDebugHud');
const _poseDebugCheckbox = document.getElementById('showPoseDebug');
let _poseDebugLastWriteMs = 0;
function _fmt(n, d = 3) {
  if (typeof n !== 'number' || !isFinite(n)) return ' --- ';
  return (n >= 0 ? ' ' : '') + n.toFixed(d);
}
function updatePoseDebugHud(poseFit, basisNormal, fingerDir, isReportedRightHand, reportedHandedness, fingerZMag, spreadRatio, activeFinger) {
  if (!_poseDebugHudEl || !_poseDebugCheckbox) return;
  if (!_poseDebugCheckbox.checked) {
    if (_poseDebugHudEl.style.display !== 'none') _poseDebugHudEl.style.display = 'none';
    return;
  }
  const nowMs = performance.now();
  if (nowMs - _poseDebugLastWriteMs < 100) return;
  _poseDebugLastWriteMs = nowMs;
  if (_poseDebugHudEl.style.display !== 'block') _poseDebugHudEl.style.display = 'block';

  // basisNormal points along the palm/back axis; basisNormal.z > 0 means the
  // palm faces the camera, < 0 means the back of the hand faces the camera.
  const palmFacingCamera = basisNormal.z > 0;
  const palmFacingMag = Math.abs(basisNormal.z);
  const fit = poseFit.fit || {};

  _poseDebugHudEl.textContent =
`POSE  : ${poseFit.name || '?'}  ${palmFacingCamera ? '(palm)' : '(back)'}
FINGER: ${activeFinger?.id ?? '?'}    HAND : ${reportedHandedness} (${isReportedRightHand ? 'R' : 'L'})

basisNormal  X:${_fmt(basisNormal.x)} Y:${_fmt(basisNormal.y)} Z:${_fmt(basisNormal.z)}
fingerDir    X:${_fmt(fingerDir.x)} Y:${_fmt(fingerDir.y)} Z:${_fmt(fingerDir.z)}

palmFacing   : ${_fmt(palmFacingMag)}
fingerZMag   : ${_fmt(fingerZMag)}
spreadRatio  : ${_fmt(spreadRatio, 2)}
sideAmount   : ${_fmt(poseFit.sideAmount, 2)}
fingerToward : ${_fmt(poseFit.fingerTowardCamera, 2)}
spreadAmount : ${_fmt(poseFit.spreadAmount, 2)}
curledAmount : ${_fmt(poseFit.curledAmount, 2)}

HIDE STATE   : ${_hideActive ? '*** HIDDEN ***' : 'visible'}  frame=${_dbgHideThisFrame ? 'Y' : 'N'}
  fwdCount   : ${_dbgForwardCount}  (>=2 hides)
  fingerRatio: ${_fmt(_dbgFingerRatio, 2)}  (need 0.45-1.80)
  badStreak  : ${_hideBadStreak} / ${HIDE_REQUIRES_BAD_FRAMES}
  goodStreak : ${_hideGoodStreak} / ${HIDE_RELEASES_GOOD_FRAMES}

FIT PROFILE  : ${poseFit.name}
  anchorBias : ${_fmt(fit.anchorBias, 2)}
  axisX      : ${_fmt(fit.axisX, 2)}
  axisY      : ${_fmt(fit.axisY, 2)}
  surfaceLift: ${_fmt(fit.surfaceLift, 2)}
  scaleBoost : ${_fmt(fit.scaleBoost, 2)}

${_riggedHandStatusLines(reportedHandedness)}`;
}

// Returns a multi-line string describing the rigged-hand load + visibility
// state for the active handedness. Lets the on-screen HUD answer the same
// question __rigDiag() answers in DevTools, without a paste step.
function _riggedHandStatusLines(reportedHandedness) {
  if (!riggedHands || riggedHands.size === 0) {
    return 'RIGGED HAND   : (not loaded yet)';
  }
  const entry = riggedHands.get(reportedHandedness);
  if (!entry) {
    return `RIGGED HAND   : (no asset for ${reportedHandedness})`;
  }
  const bbox = new THREE.Box3().setFromObject(entry.root);
  const size = new THREE.Vector3();
  bbox.getSize(size);
  return `RIGGED HAND   : ${entry.prefix}  bones=${entry.resolved}/21  vis=${entry.root.visible ? 'yes' : 'NO'}
  bbox size  : ${_fmt(size.x)} ${_fmt(size.y)} ${_fmt(size.z)}
  bbox center: ${_fmt((bbox.min.x + bbox.max.x) * 0.5)} ${_fmt((bbox.min.y + bbox.max.y) * 0.5)} ${_fmt((bbox.min.z + bbox.max.z) * 0.5)}`;
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

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function smoothstep(edge0, edge1, value) {
  const t = clamp01((value - edge0) / (edge1 - edge0 || 1));
  return t * t * (3 - 2 * t);
}

function mix(a, b, t) {
  return a + (b - a) * t;
}

function getSegmentSceneDirectionInto(out, landmarks, startIdx, endIdx) {
  mapToOrthographicSpaceInto(_tmpPosA, landmarks[endIdx]);
  mapToOrthographicSpaceInto(_tmpPosB, landmarks[startIdx]);
  return out.subVectors(_tmpPosA, _tmpPosB).normalize();
}

function estimateFingerRadiusScene(landmarks, anchorStart, anchorEnd, segmentLengthScene) {
  mapToOrthographicSpaceInto(_tmpPosA, landmarks[5]);
  mapToOrthographicSpaceInto(_tmpPosB, landmarks[17]);
  const palmSpanScene = _tmpPosA.distanceTo(_tmpPosB);
  const fromSegment = segmentLengthScene * 0.23;
  const fromPalm = palmSpanScene * 0.055;
  return Math.max(segmentLengthScene * 0.16, Math.min(segmentLengthScene * 0.34, Math.max(fromSegment, fromPalm)));
}

function classifyRingPose(landmarks, worldLandmarks, basisNormal, fingerDir, spreadRatio, isReportedRightHand) {
  const palmFacing = Math.abs(basisNormal.z);
  const sideAmount = 1 - smoothstep(0.28, 0.72, palmFacing);
  const fingerTowardCamera = smoothstep(0.48, 0.86, Math.abs(fingerDir.z));
  const spreadAmount = smoothstep(1.32, 1.70, spreadRatio);

  const tipToMcp = getSegmentLength3D(worldLandmarks, 13, 16) || getSegmentLength2D(landmarks, 13, 16);
  const mcpToPip = getSegmentLength3D(worldLandmarks, 13, 14) || getSegmentLength2D(landmarks, 13, 14);
  const curledAmount = mcpToPip > 1e-4 ? clamp01(1 - (tipToMcp / (mcpToPip * 2.35))) : 0;

  // FIX A: handedness-aware front/back classification. The basis normal
  // direction depends on K = palmSpan(5→17), which flips sign between right
  // and left hands. Empirically, for a right hand the labels were inverted —
  // back-of-hand poses got 'front' and palm-facing got 'back'. Negate the
  // sign for right hands so both hands resolve to the same physical labels.
  const palmFacingZ = isReportedRightHand ? -basisNormal.z : basisNormal.z;
  let name = palmFacingZ >= 0 ? 'front' : 'back';
  if (curledAmount > 0.30) name = 'fist';
  if (sideAmount > 0.70) name = 'side';
  if (fingerTowardCamera > 0.65) name = 'foreshortened';

  const fit = RING_POSE_FIT[name] || RING_POSE_FIT.front;
  return {
    name,
    fit,
    palmFacing,
    sideAmount,
    fingerTowardCamera,
    spreadAmount,
    curledAmount
  };
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
    hideHandOccluders();
    hideBlockerOccluders();
    updateRiggedHand(results); // hides rigged debug hand when no landmarks
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
  const activeFinger = getActiveFinger();
  // Gate hand-skeleton overlay on the Show Hand Mesh checkbox. When unchecked,
  // clear the canvas instead of drawing — otherwise the last drawn skeleton
  // would linger on screen.
  if (_cbShowHandMesh && _cbShowHandMesh.checked) {
    drawLandmarks(landmarks);
  } else {
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
  }
  const reportedHandedness = getReportedHandedness(results);
  // PR-α: drive the rigged debug hand from landmarks. Pure additive — no
  // effect on tracking or occlusion when the checkbox is off.
  updateRiggedHand(results);
  const activeHandOccluder = getHandOccluderForResults(results);
  if (activeHandOccluder && _cbShowHandMesh && _cbShowHandMesh.checked) {
    hideHandOccluders(activeHandOccluder);
    activeHandOccluder.setMode('debug');
    activeHandOccluder.setVisible(true);
    activeHandOccluder.updatePose(results, { activeFingerId: activeFinger.id });
  } else {
    hideHandOccluders();
  }
  const worldLandmarks = results.worldLandmarks ? results.worldLandmarks[0] : null;
  const isReportedRightHand = reportedHandedness === 'Right';
  const [anchorStart, anchorEnd] = activeFinger.anchor;

  // Position (Anchored to the selected finger segment). Uses _tmpPosA as a
  // scratch for the NDC conversion before copying into targetPos.
  const ringAnchorT = _slRingAnchorT ? Math.min(1, Math.max(0, parseFloat(_slRingAnchorT.value) || 0.5)) : 0.5;
  _tmpMid2D.x = landmarks[anchorStart].x * (1 - ringAnchorT) + landmarks[anchorEnd].x * ringAnchorT;
  _tmpMid2D.y = landmarks[anchorStart].y * (1 - ringAnchorT) + landmarks[anchorEnd].y * ringAnchorT;
  mapToOrthographicSpaceInto(_tmpPosA, _tmpMid2D);
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
  const poseFit = classifyRingPose(landmarks, worldLandmarks, _tmpDirN, _tmpDirD, spreadRatio, isReportedRightHand);
  window.__lastRingPoseFit = poseFit;

  // Patch 2 (Path B): occluded-finger guard. When the active finger is
  // hidden behind extended fingers (peace-sign / scissors poses), MediaPipe
  // still reports world landmarks for it but the values drift sideways and
  // the ring renders in empty space. Detect the situation by comparing the
  // active finger's MCP→TIP world distance against the palm span — a
  // healthy extended finger is roughly 0.7×–1.6× the palm-span; values way
  // below or above mean the model is guessing. Hide the ring rather than
  // place it in the phantom position. Showing nothing > showing wrong.
  let _hideThisFrame = false;
  if (worldLandmarks && activeFinger?.anchor) {
    const [aStart] = activeFinger.anchor;
    const aTip = aStart + 3;
    const fingerWorldLen = getSegmentLength3D(worldLandmarks, aStart, aTip);
    const palmWorldSpan = getSegmentLength3D(worldLandmarks, 5, 17);
    if (palmWorldSpan > 1e-4 && fingerWorldLen > 1e-4) {
      const ratio = fingerWorldLen / palmWorldSpan;
      _dbgFingerRatio = ratio;
      // Outside [0.45, 1.80] → finger landmark is suspect.
      if (ratio < 0.45 || ratio > 1.80) _hideThisFrame = true;
    }

    // Self-occlusion guard for poses like peace-sign / V-sign where the
    // active finger is curled BEHIND two extended fingers. Compare other
    // fingertips against the active finger's TIP Z (not MCP Z) — when all
    // fingers are spread open toward the camera every tip is naturally in
    // front of the MCP, which caused a false-positive hide on open palms.
    if (!_hideThisFrame) {
      const aTipZ = worldLandmarks[aStart + 3].z;  // active finger TIP z
      const aTipIdx = aStart + 3;
      const FINGER_TIPS = [8, 12, 16, 20];
      let forwardCount = 0;
      for (const tipIdx of FINGER_TIPS) {
        if (tipIdx === aTipIdx) continue;
        if (worldLandmarks[tipIdx].z - aTipZ < -0.04) forwardCount++;
      }
      _dbgForwardCount = forwardCount;
      if (forwardCount >= 2) _hideThisFrame = true;
    }
  }
  _dbgHideThisFrame = _hideThisFrame;
  // FIX D: hysteresis. A single bad frame doesn't hide; the ring stays
  // hidden across short noise bursts. State transitions only on streaks.
  if (_hideThisFrame) {
    _hideBadStreak++;
    _hideGoodStreak = 0;
    if (_hideBadStreak >= HIDE_REQUIRES_BAD_FRAMES) _hideActive = true;
  } else {
    _hideGoodStreak++;
    _hideBadStreak = 0;
    if (_hideGoodStreak >= HIDE_RELEASES_GOOD_FRAMES) _hideActive = false;
  }
  if (_hideActive) {
    ringModel.visible = false;
    occluderMesh.visible = false;
    hideBlockerOccluders();
    updatePoseDebugHud(poseFit, _tmpDirN, _tmpDirD, isReportedRightHand, reportedHandedness, fingerZMag, spreadRatio, activeFinger);
    return;
  }

  // Diagnostic HUD: read-only sample of the inputs the classifier just used,
  // plus the chosen profile name. Throttled to ~10 fps so it doesn't burn
  // layout time on every render frame.
  updatePoseDebugHud(poseFit, _tmpDirN, _tmpDirD, isReportedRightHand, reportedHandedness, fingerZMag, spreadRatio, activeFinger);

  // FIX B: for FLAT-OPEN hands (palm or back facing camera, fingers spread,
  // not curled), the screen-blend on _tmpDirD can drag the ring laterally
  // off the finger because it's mixing two unit vectors with different
  // magnitudes when normalized. Detect flat-open and skip the blend — the
  // 2D anchor projection is already accurate for these poses.
  const _isFlatOpen = poseFit.palmFacing > 0.80 && poseFit.curledAmount < 0.20;
  // Screen-space assist keeps the rendered ring visually parallel to the
  // selected finger while preserving most of MediaPipe's 3D depth direction.
  // The blend is deliberately small; it corrects drift without taking over.
  getSegmentSceneDirectionInto(_tmpDirSurface, landmarks, anchorStart, anchorEnd);
  const _effectiveScreenBlend = _isFlatOpen ? 0 : poseFit.fit.screenBlend;
  _tmpDirD.lerp(_tmpDirSurface, _effectiveScreenBlend).normalize();
  _tmpDirN.crossVectors(_tmpDirK, _tmpDirD).normalize();
  if (_tmpDirN.lengthSq() < 0.01) {
    _tmpDirN.crossVectors(Y_AXIS, _tmpDirD).normalize();
  }
  _tmpDirX.crossVectors(_tmpDirD, _tmpDirN).normalize();
  // FIX C: for flat-open hands, override the basis with a screen-space frame
  // so the ring's roll axis stays anchored to the visible image plane. The
  // MediaPipe-world basis can flip sign on tiny landmark noise when the
  // palm is parallel to the image plane, which made the ring appear rolled
  // 90° on otherwise-correct flat poses (e.g. open palm). Screen-space D
  // (already in _tmpDirSurface) plus a discretized N keeps roll locked.
  // Sign of N is taken from the freshly-computed world palm normal so it
  // matches palm/back direction regardless of pose-name handling — the
  // pose-name labels are handedness-inverted by FIX A and not safe to
  // branch on here.
  if (_isFlatOpen) {
    const _origNormSignZ = _tmpDirN.z >= 0 ? 1 : -1;
    _tmpDirD.copy(_tmpDirSurface);
    _tmpDirN.set(0, 0, _origNormSignZ);
    _tmpDirX.crossVectors(_tmpDirD, _tmpDirN).normalize();
    _tmpDirN.crossVectors(_tmpDirX, _tmpDirD).normalize();
  }
  _tmpMat4.makeBasis(_tmpDirX, _tmpDirD, _tmpDirN);
  targetOccQuat.setFromRotationMatrix(_tmpMat4);

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

  const assistedAnchorT = Math.min(0.78, Math.max(0.18, ringAnchorT + poseFit.fit.anchorBias));
  _tmpMid2D.x = landmarks[anchorStart].x * (1 - assistedAnchorT) + landmarks[anchorEnd].x * assistedAnchorT;
  _tmpMid2D.y = landmarks[anchorStart].y * (1 - assistedAnchorT) + landmarks[anchorEnd].y * assistedAnchorT;
  mapToOrthographicSpaceInto(_tmpPosA, _tmpMid2D);
  targetPos.copy(_tmpPosA);

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
  const fingerRadiusScene = estimateFingerRadiusScene(landmarks, anchorStart, anchorEnd, segLengthWorld);
  const sBase = parseFloat(_slScaleBase.value);
  const radiusScale = fingerRadiusScene * 3.25;
  const radiusBlend = poseFit.fit.scaleRadiusBlend;
  const fitMetric = mix(segLengthWorld, radiusScale, radiusBlend);
  // Per-pose calibration. Lives in RING_POSE_FIT so each pose (front/back/
  // side/foreshortened/fist) tunes independently — a global multiplier was
  // over-correcting on back-of-hand and curled-finger poses that were already
  // sized correctly. Default 1.0 if a pose entry doesn't define it.
  const poseScaleBoost = poseFit.fit.scaleBoost ?? 1.0;
  const scaleVal = fitMetric * sBase * mix(1.0, activeFinger.widthMul, 0.35) * poseScaleBoost;
  targetScale.set(scaleVal, scaleVal, scaleVal);

  // Placement offsets mirror Jewel's manual Position controls, but are applied
  // in the live finger basis so they continue to behave correctly as the hand
  // rotates: X = across finger, Y = along finger, Z = palm/back normal.
  // A zero manual offset uses the pose profile. A non-zero slider value acts
  // as an override, which keeps the tweak engine useful during calibration.
  const offsetX = _slRingOffsetX ? parseFloat(_slRingOffsetX.value) || 0 : 0;
  const offsetY = _slRingOffsetY ? parseFloat(_slRingOffsetY.value) || 0 : 0;
  const offsetZ = _slRingOffsetZ ? parseFloat(_slRingOffsetZ.value) || 0 : 0;
  // FIX B (continued): for flat-open hands, zero the axisX/axisY profile push
  // entirely. These offsets are added in MediaPipe-world directions but
  // scaled by SCENE-space distances — a coordinate-frame mix that's small
  // for non-flat poses but blows up on open palms where the world basis can
  // tilt. Manual slider overrides still apply (calibration use).
  const fitAxisX = Math.abs(offsetX) > 0.001 ? offsetX : (_isFlatOpen ? 0 : poseFit.fit.axisX);
  const fitAxisY = Math.abs(offsetY) > 0.001 ? offsetY : (_isFlatOpen ? 0 : poseFit.fit.axisY);
  const fitSurfaceZ = poseFit.fit.surfaceLift + offsetZ;
  // FIX 1 (Path B): on a left hand, MediaPipe's mirrored-image landmarks
  // make _tmpDirX point the OPPOSITE direction across the finger from the
  // right-hand case. The handedness rotation flip already handles ring
  // orientation, but the lateral placement offset (axisX) ends up on the
  // wrong side of the finger. Negate the X contribution for left hands so
  // the ring biases toward the same anatomical side regardless of which
  // hand is in frame.
  const handednessSignX = isReportedRightHand ? 1 : -1;
  _tmpDirSurface.copy(_tmpDirN);
  if (_tmpDirSurface.z < 0) _tmpDirSurface.multiplyScalar(-1);
  targetPos
    .addScaledVector(_tmpDirX, segLengthWorld * fitAxisX * handednessSignX)
    .addScaledVector(_tmpDirD, segLengthWorld * fitAxisY)
    .addScaledVector(_tmpDirSurface, fingerRadiusScene * fitSurfaceZ);

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
    fingerRadiusScene * 2.2 * occluderBase * activeFinger.widthMul,
    segLengthWorld * 2.0 * finalForeshorten, 
    fingerRadiusScene * 2.2 * occluderBase * activeFinger.widthMul
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
