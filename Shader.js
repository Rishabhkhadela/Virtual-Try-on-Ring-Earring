// Shader.js — jewelry shading for the r146 (UMD-global) Three.js setup.
//
// Exports expected by main.js / faceTracking.js:
//   configureRenderer(renderer)
//   loadHDREnvironment(renderer, scene, path) -> Promise<{envMap, equirect}>
//   applyJewelryShading(group, { metal, diamond })
//
// DIAMOND — custom translucent refraction shader (reliable GLSL1).
//
// Earlier BVH ray-trace approach (Shader.ts port) failed to link against
// three-mesh-bvh 0.5.24's GLSL chunks on r146 — the fragment shader never
// compiled, leaving gems invisible. Replaced with a GLSL1 refraction + chroma
// shader that:
//   1. Samples the raw equirect HDR three times (R/G/B rays at different IOR)
//      to produce chromatic dispersion ("fire").
//   2. Uses Schlick fresnel with F0 = 0.17 (IOR 2.42) to mix refraction and
//      reflection the way a real diamond does.
//   3. Alpha-modulates by fresnel so the gem *body* is mostly transparent
//      (video shows through) while facet edges stay opaque and specular.
//      That's the real AR-diamond look — not a chrome ball.
//
// This is GLSL1 (the default). No `uvec4`, no `usampler2D`, no extensions —
// compiles everywhere, no cross-THREE-instance gotchas.

const THREE = window.THREE;
const _REVISION = parseInt(THREE.REVISION, 10) || 0;

// three-mesh-bvh for true internal-reflection ray-trace ("real diamond fire").
// Dynamic top-level import so a CDN hiccup falls back to the sparkle-overlay
// path instead of killing the page.
let _bvhLib = null;
try {
  _bvhLib = await import('three-mesh-bvh');
  console.log('[Shader] three-mesh-bvh loaded — BVH ray-trace enabled for diamonds');
} catch (err) {
  console.warn('[Shader] three-mesh-bvh unavailable — using sparkle-overlay fallback:', err);
}

// ============================================================
// Module state
// ============================================================
let _hdrEquirectTexture = null;
let _diamondPatternOverride = null;

const DIAMOND_NAME_PATTERN = /diamond|gem|stone|crystal|brilliant|jewel|rhinestone|glass|sapphire|ruby|emerald|quartz|zircon|rock/i;

// --- Dynamic cube environment for diamond shaders ----------------------------
// Previously the BVH/custom diamond shaders sampled the raw equirect HDR, which
// means the gem only "sees" the studio sky — not the ring's metal prongs right
// next to it. Real diamonds reflect their mounting as dark structural lines
// inside the stone; without it the gem looks like it's floating in a void.
//
// main.js now owns a CubeCamera + WebGLCubeRenderTarget positioned at the ring.
// It renders the scene (metal + HDR sky background, diamonds/video-backdrop
// hidden) into the cube every frame and calls setDiamondEnvCube() with the
// resulting CubeTexture. The shaders sample this via samplerCube — so refl/refr
// rays that aim at the prongs pick up real dark-metal pixels.
let _dynamicEnvCubeTexture = null;
const _diamondMaterialsWithEnvMap = new Set();

// Lazy fallback cube — pure-black 1×1 cube used until the real cube arrives so
// the samplerCube uniform never points at null.
let _fallbackCubeRT = null;
function getFallbackCube() {
  if (!_fallbackCubeRT) _fallbackCubeRT = new THREE.WebGLCubeRenderTarget(1);
  return _fallbackCubeRT.texture;
}

// 2x2 warm fallback — still used by the equirect-based HDR loader + the custom
// GLSL1 shader until the dynamic cube is hooked up.
const _fallbackEnvTexture = (() => {
  const data = new Uint8Array([
    200, 190, 180, 255, 220, 210, 195, 255,
    120, 115, 110, 255, 140, 135, 125, 255,
  ]);
  const tex = new THREE.DataTexture(data, 2, 2, THREE.RGBAFormat);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
})();

// Call this from main.js with the CubeCamera's render target texture. Rebinds
// the envMap uniform on every already-created diamond material so they all
// start sampling the live cube immediately.
export function setDiamondEnvCube(cubeTexture) {
  _dynamicEnvCubeTexture = cubeTexture;
  for (const mat of _diamondMaterialsWithEnvMap) {
    if (mat.uniforms?.envMap) mat.uniforms.envMap.value = cubeTexture;
  }
}

// Clean HDR equirect, used only by BVH-exit refraction rays (not reflection).
// Reason: in cluster settings each gem's CubeCamera bakes in its neighbor's
// dark back-facet, so rays that exited the gem and want to sample "the world
// out there" end up hitting a neighbor gem and dragging the cluster dim. HDR
// gives those rays a clean studio backdrop, restoring cluster brightness.
// Reflection rays still use the cube (keeps prong-darkness inside the stone).
let _dynamicEnvHDRTexture = null;
export function setDiamondEnvHDR(hdrEquirect) {
  _dynamicEnvHDRTexture = hdrEquirect;
  for (const mat of _diamondMaterialsWithEnvMap) {
    if (mat.uniforms?.envHDR) {
      mat.uniforms.envHDR.value = hdrEquirect;
      mat.uniforms.uUseHDR.value = 1;
    }
  }
}

// ============================================================
// Detection — name regex + transmission flag + (dielectric & smooth) heuristic
// ============================================================
function isDiamondMesh(mesh, material) {
  if (!material) return false;
  const matName = (material.name || '').toLowerCase();
  const meshName = (mesh?.name || '').toLowerCase();
  const pattern = _diamondPatternOverride || DIAMOND_NAME_PATTERN;

  if (pattern.test(matName) || pattern.test(meshName)) return true;
  if (typeof material.transmission === 'number' && material.transmission > 0.3) return true;
  if (typeof material.metalness === 'number' && typeof material.roughness === 'number') {
    if (material.metalness < 0.35 && material.roughness < 0.3) return true;
  }
  return false;
}

// ============================================================
// Renderer config
// ============================================================
export function configureRenderer(renderer) {
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  // 1.3 (dropped from 1.8): with envMapIntensity=10 the metal band was burning
  // past the bloom threshold and Gaussian-spilling white onto adjacent skin.
  // 1.3 keeps gems bright without bleaching the ring's edge halo into the hand.
  renderer.toneMappingExposure = 1.3;
  renderer.physicallyCorrectLights = true;
}

// ============================================================
// Rotate an equirect HDR around Y by `angleRadians`. We re-render the equirect
// into a fresh HalfFloat render target, shifting U by angle/(2π). Result is a
// new equirect texture that every downstream consumer (PMREM for metals,
// CubeCamera seed, scene.background during cube capture, BVH shader's direct
// envHDR sampling) can use without any per-path rotation logic — the rotation
// is baked into the pixels once.
// ============================================================
function rotateEquirectY(renderer, srcTexture, angleRadians) {
  if (!angleRadians) return srcTexture;
  const w = (srcTexture.image?.width) || 1024;
  const h = (srcTexture.image?.height) || 512;
  const rt = new THREE.WebGLRenderTarget(w, h, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
    type: THREE.HalfFloatType,   // preserve HDR range; float equirects routinely exceed 1.0
    // Inherit the source's encoding so downstream PMREM / CubeCamera decode
    // the rotated texture identically to the original. Default LinearEncoding
    // was causing the ring's cube-reflected HDR to read dim — the RT ended up
    // flagged as a different color space than the raw equirect it replaced.
    encoding: srcTexture.encoding || THREE.LinearEncoding,
    generateMipmaps: false,
    wrapS: THREE.RepeatWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
  });

  // (Earlier version set rt.texture.image = {width, height} here thinking PMREM
  // needed it — it doesn't, and that fake image object caused WebGL2 to throw
  // "texSubImage2D: Overload resolution failed" every frame because Three.js
  // tried to upload the texture from a non-DOM/non-ImageData source. Removed.)

  const mat = new THREE.ShaderMaterial({
    uniforms: {
      tSrc: { value: srcTexture },
      uShift: { value: angleRadians / (2.0 * Math.PI) }
    },
    vertexShader: `
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = vec4(position, 1.0); }
    `,
    fragmentShader: `
      uniform sampler2D tSrc;
      uniform float uShift;
      varying vec2 vUv;
      void main() {
        vec2 uv = vUv;
        uv.x = fract(uv.x + uShift);
        gl_FragColor = texture2D(tSrc, uv);
      }
    `,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,   // belt-and-suspenders — keep linear HDR values intact
  });

  const quadScene = new THREE.Scene();
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
  quadScene.add(quad);

  // Save a broader slice of renderer state. ACES tonemap + sRGB outputEncoding
  // from configureRenderer() were compressing HDR values on write to our RT,
  // producing a darker rotated equirect than the source. Linear everything
  // during this one-shot pass.
  const prevRT = renderer.getRenderTarget();
  const prevAutoClear = renderer.autoClear;
  const prevToneMapping = renderer.toneMapping;
  const prevOutputEncoding = renderer.outputEncoding;

  renderer.autoClear = true;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputEncoding = THREE.LinearEncoding;

  renderer.setRenderTarget(rt);
  renderer.render(quadScene, quadCam);

  renderer.setRenderTarget(prevRT);
  renderer.autoClear = prevAutoClear;
  renderer.toneMapping = prevToneMapping;
  renderer.outputEncoding = prevOutputEncoding;

  quad.geometry.dispose();
  mat.dispose();

  // Present the RT texture as a drop-in equirect replacement. Do NOT set
  // needsUpdate — render-target textures receive their pixels from the GPU
  // pass above, not from a CPU-side .image source. Flagging them dirty
  // makes Three.js repeatedly retry a broken CPU upload every frame.
  rt.texture.mapping = THREE.EquirectangularReflectionMapping;
  rt.texture.minFilter = THREE.LinearFilter;
  rt.texture.magFilter = THREE.LinearFilter;
  rt.texture.wrapS = THREE.RepeatWrapping;
  rt.texture.wrapT = THREE.ClampToEdgeWrapping;
  return rt.texture;
}

// ============================================================
// HDR environment — PMREM for PBR, raw equirect for the custom diamond shader
// opts.rotationY — optional Y-axis rotation in radians applied ONCE to the
//   loaded equirect before PMREM generation. Rotates the entire IBL coherently
//   across metals, diamonds, cube capture, and scene.background.
// ============================================================
export function loadHDREnvironment(renderer, scene, path, opts = {}) {
  const rotationY = opts.rotationY || 0;
  return new Promise((resolve, reject) => {
    const loader = new THREE.RGBELoader();
    if (loader.setDataType) loader.setDataType(THREE.HalfFloatType);

    loader.load(path, (rawTexture) => {
      rawTexture.mapping = THREE.EquirectangularReflectionMapping;
      rawTexture.minFilter = THREE.LinearFilter;
      rawTexture.magFilter = THREE.LinearFilter;
      rawTexture.wrapS = THREE.RepeatWrapping;
      rawTexture.wrapT = THREE.ClampToEdgeWrapping;
      rawTexture.needsUpdate = true;

      // Pre-rotate around Y so every downstream path sees the same rotated HDR
      // without needing its own rotation logic.
      const texture = rotateEquirectY(renderer, rawTexture, rotationY);

      const pmrem = new THREE.PMREMGenerator(renderer);
      pmrem.compileEquirectangularShader();
      const envMap = pmrem.fromEquirectangular(texture).texture;
      scene.environment = envMap;
      pmrem.dispose();

      _hdrEquirectTexture = texture;
      // Diamond shaders no longer sample the equirect directly — they read
      // from the dynamic cube fed by main.js. PBR metals still pick up the
      // HDR via scene.environment (set above).

      console.log(`[Shader] HDR loaded: ${path}`);
      resolve({ envMap, equirect: texture });
    }, undefined, (err) => {
      console.warn('[Shader] HDR load failed:', err);
      reject(err);
    });
  });
}

export function setDiamondDetectionPattern(regex) {
  _diamondPatternOverride = regex;
}

// Per-frame scintillation driver. Pass in an euler or quaternion representing
// the ring's current world rotation; this multiplies the angle × 2 internally
// and updates every diamond material's uEnvTwist uniform so sparkle moves
// faster than the ring. Call from animate() after ringModel.quaternion.slerp.
const _scratchEuler = new THREE.Euler();
const _scratchMat4 = new THREE.Matrix4();
const _scratchTwist3 = new THREE.Matrix3();
export function updateDiamondEnvTwist(group, quatOrNull) {
  if (!group) return;
  if (quatOrNull) {
    _scratchEuler.setFromQuaternion(quatOrNull, 'XYZ');
    // × 4 — sparkle moves 4× faster than the hand. The extra speed visually
    // masks MediaPipe's per-frame tracking jitter: the env-twist noise floor
    // looks intentional (scintillation) instead of AR wobble.
    _scratchEuler.set(_scratchEuler.x * 4, _scratchEuler.y * 4, _scratchEuler.z * 4, 'XYZ');
    _scratchMat4.makeRotationFromEuler(_scratchEuler);
    _scratchTwist3.setFromMatrix4(_scratchMat4);
  } else {
    _scratchTwist3.identity();
  }
  group.traverse((node) => {
    if (!node.isMesh || !node.material) return;
    const mats = Array.isArray(node.material) ? node.material : [node.material];
    mats.forEach((m) => {
      const u = m.userData?.diamondUniforms?.uEnvTwist;
      if (u) u.value.copy(_scratchTwist3);
    });
  });
}

// ============================================================
// Diamond — custom translucent refraction shader (default)
// ============================================================
export function createDiamondShaderMaterial(params = {}) {
  const {
    ior = 2.42,
    dispersion = 0,         // aberrationStrength — stronger default = more visible fire
    brightness = 2.8,          // HDR boost so facet peaks push past bloom threshold (1.25)
    color = new THREE.Color(0xffffff),
    sparkleStrength = 0,     // multiplier on the facet-edge scintillation
    causticStrength = 0.5,     // how much the multi-jitter max samples contribute (brilliance)
  } = params;

  const material = new THREE.ShaderMaterial({
    uniforms: {
      // samplerCube — dynamic cube from main.js CubeCamera (metal + HDR sky).
      envMap: { value: _dynamicEnvCubeTexture || getFallbackCube() },
      ior: { value: ior },
      aberrationStrength: { value: dispersion },
      brightness: { value: brightness },
      color: { value: color.clone() },
      envRotation: { value: new THREE.Matrix3() },
      sparkleStrength: { value: sparkleStrength },
      causticStrength: { value: causticStrength },
    },
    extensions: {
      derivatives: true,  // dFdx/dFdy for facet-edge sparkle (GL_OES_standard_derivatives)
    },
    vertexShader: /* glsl */ `
      varying vec3 vWorldPosition;
      varying vec3 vWorldNormal;

      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPosition = wp.xyz;
        // Gem meshes are uniformly scaled in loadRingModel / loadEarringModel,
        // so mat3(modelMatrix) is a valid normal transform.
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;

      varying vec3 vWorldPosition;
      varying vec3 vWorldNormal;

      // samplerCube fed by main.js CubeCamera — contains metal prongs + HDR.
      uniform samplerCube envMap;
      uniform float ior;
      uniform float aberrationStrength;
      uniform float brightness;
      uniform vec3  color;
      uniform mat3  envRotation;
      uniform float sparkleStrength;
      uniform float causticStrength;

      #define PI 3.141592653589793

      vec4 envSample(samplerCube m, vec3 dir) {
        vec3 rd = envRotation * normalize(dir);
        return textureCube(m, rd);
      }

      // Cheap 3D hash — produces a pseudo-random value in [0,1] from any position.
      // Used for view-dependent scintillation (diamond twinkle).
      float hash3(vec3 p) {
        return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
      }

      // Refract with reflection fallback for total-internal-reflection
      vec3 refractOrReflect(vec3 V, vec3 N, float eta, vec3 refl) {
        vec3 r = refract(V, N, eta);
        return length(r) < 0.001 ? refl : r;
      }

      // 7-channel SPECTRAL dispersion — proper prism rainbow.
      // Each wavelength refracts at a slightly different IOR, and each gets
      // weighted by its sRGB-approximated spectral color. Sum produces the
      // classic diamond fire (red at one edge, violet at the opposite edge).
      vec3 spectralRefraction(vec3 V, vec3 N, float baseEta, float strength, vec3 refl) {
        float s = strength;
        vec3 d0 = refractOrReflect(V, N, baseEta * (1.0 + s * 1.00), refl);   // red
        vec3 d1 = refractOrReflect(V, N, baseEta * (1.0 + s * 0.66), refl);   // orange
        vec3 d2 = refractOrReflect(V, N, baseEta * (1.0 + s * 0.33), refl);   // yellow
        vec3 d3 = refractOrReflect(V, N, baseEta,                      refl); // green (base)
        vec3 d4 = refractOrReflect(V, N, baseEta * (1.0 - s * 0.33), refl);   // cyan
        vec3 d5 = refractOrReflect(V, N, baseEta * (1.0 - s * 0.66), refl);   // blue
        vec3 d6 = refractOrReflect(V, N, baseEta * (1.0 - s * 1.00), refl);   // violet

        vec3 s0 = envSample(envMap, d0).rgb;
        vec3 s1 = envSample(envMap, d1).rgb;
        vec3 s2 = envSample(envMap, d2).rgb;
        vec3 s3 = envSample(envMap, d3).rgb;
        vec3 s4 = envSample(envMap, d4).rgb;
        vec3 s5 = envSample(envMap, d5).rgb;
        vec3 s6 = envSample(envMap, d6).rgb;

        // sRGB weights per spectral band — sums to ~3x luminance, divided out below
        vec3 accum = vec3(0.0);
        accum += s0 * vec3(1.00, 0.00, 0.00);   // red
        accum += s1 * vec3(1.00, 0.45, 0.00);   // orange
        accum += s2 * vec3(0.90, 0.90, 0.00);   // yellow
        accum += s3 * vec3(0.20, 1.00, 0.20);   // green
        accum += s4 * vec3(0.00, 0.80, 0.90);   // cyan
        accum += s5 * vec3(0.10, 0.20, 1.00);   // blue
        accum += s6 * vec3(0.55, 0.00, 1.00);   // violet
        return accum / 2.8;
      }

      // Bright-biased multi-sample — finds nearby HDR hotspots. Fakes the bright
      // caustic returns that a BVH internal-bounce trace would produce.
      vec3 envMaxSample(vec3 dir, float radius) {
        vec3 up = abs(dir.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 t = normalize(cross(dir, up));
        vec3 b = cross(dir, t);
        vec3 c0 = envSample(envMap, dir).rgb;
        vec3 c1 = envSample(envMap, normalize(dir + t * radius)).rgb;
        vec3 c2 = envSample(envMap, normalize(dir - t * radius)).rgb;
        vec3 c3 = envSample(envMap, normalize(dir + b * radius)).rgb;
        vec3 c4 = envSample(envMap, normalize(dir - b * radius)).rgb;
        return max(c0, max(max(c1, c2), max(c3, c4)));
      }

      // Procedural studio lights — THIS is the key to the diamond look when the
      // loaded HDR is low-contrast. Three bright "virtual lights" at fixed
      // directions; the refracted/reflected ray dot-products against each,
      // producing sharp highlights exactly where a real gem would catch a
      // studio strobe. Works even in a dim uniform HDR.
      float studioLights(vec3 dir) {
        float l1 = pow(max(0.0, dot(dir, normalize(vec3( 1.0,  1.0,  0.4)))), 40.0);
        float l2 = pow(max(0.0, dot(dir, normalize(vec3(-1.0,  0.6,  0.6)))), 40.0);
        float l3 = pow(max(0.0, dot(dir, normalize(vec3( 0.2, -0.8,  1.0)))), 40.0);
        float l4 = pow(max(0.0, dot(dir, normalize(vec3(-0.4, -0.3, -1.0)))), 40.0);
        return l1 * 1.0 + l2 * 0.85 + l3 * 0.7 + l4 * 0.6;
      }

      void main() {
        vec3 V = normalize(vWorldPosition - cameraPosition);
        vec3 N = normalize(vWorldNormal);
        float eta = 1.0 / ior;
        vec3 refl = reflect(V, N);

        // Simple 3-channel chromatic refraction
        float s = aberrationStrength;
        vec3 refrG = refractOrReflect(V, N, eta,               refl);
        vec3 refrR = refractOrReflect(V, N, eta * (1.0 - s),   refl);
        vec3 refrB = refractOrReflect(V, N, eta * (1.0 + s),   refl);

        vec3 refrColor = vec3(
          envSample(envMap, refrR).r,
          envSample(envMap, refrG).g,
          envSample(envMap, refrB).b
        );
        vec3 reflColor = envSample(envMap, refl).rgb;

        // Schlick fresnel — F0 ≈ 0.17 for diamond IOR 2.42
        float cosI    = max(0.0, dot(-V, N));
        float F0      = 0.17;
        float fresnel = F0 + (1.0 - F0) * pow(1.0 - cosI, 5.0);

        // ==== White-biased base ====
        // 60% env contribution + 45% white lift. Gives a clean bright
        // crystalline body with visible facet shading (since env still varies).
        vec3 envColor = mix(refrColor, reflColor, fresnel);
        envColor = envColor * 0.6 + vec3(0.45);
        envColor *= color * brightness;

        // ==== Procedural studio-light highlights ====
        // Sample the same "virtual studio lights" once through refraction and
        // once through reflection, and add as bright white specular.
        // These are what actually look like the bright hits on a real diamond.
        float refrLights = studioLights(refrG);
        float reflLights = studioLights(refl);
        vec3 highlights = vec3(1.0, 0.97, 0.93) * (refrLights * 2.2 + reflLights * 2.8 * fresnel);
        envColor += highlights;

        // ==== Rainbow fringe at edges ====
        // Spectral dispersion by pulling apart RGB wavelengths proportional to
        // how hot each studio-light hit is per-channel. This gives rainbow
        // fringing concentrated around the bright facets (where fire happens).
        float lightsR = studioLights(refrR);
        float lightsB = studioLights(refrB);
        envColor.r += lightsR * 0.8;
        envColor.b += lightsB * 0.8;

        // ==== Facet-edge scintillation ====
        vec3 dN = abs(dFdx(N)) + abs(dFdy(N));
        float edge = length(dN);
        float edgeSparkle = smoothstep(0.005, 0.1, edge);
        envColor += vec3(1.4, 1.35, 1.2) * edgeSparkle * sparkleStrength * 1.8;

        // ==== View-dependent twinkle ====
        vec3 twinkleSeed = floor(vWorldPosition * 180.0) + floor(V * 22.0);
        float twinkle = hash3(twinkleSeed);
        float twinkleFlash = pow(max(twinkle - 0.92, 0.0) / 0.08, 3.0);
        envColor += vec3(1.6, 1.5, 1.3) * twinkleFlash * sparkleStrength * 5.0;

        // Rim fresnel boost — edges get extra punch
        envColor += envColor * fresnel * 0.5;

        // Solid opaque white diamond
        gl_FragColor = vec4(envColor, 1.0);
        #include <tonemapping_fragment>
        #include <encodings_fragment>
      }
    `,
    transparent: false,   // bright white diamond — opaque object, not a glass shell
    depthWrite: true,
    side: THREE.FrontSide,
    toneMapped: true,
  });

  _diamondMaterialsWithEnvMap.add(material);
  return material;
}

// ============================================================
// Diamond — PBR transmission fallback (opt-in via opts.diamond.usePBR === true)
// ============================================================
function autoThickness(geometry) {
  if (!geometry) return 0.5;
  if (!geometry.boundingBox) geometry.computeBoundingBox();
  if (!geometry.boundingBox) return 0.5;
  const size = new THREE.Vector3();
  geometry.boundingBox.getSize(size);
  const dim = Math.max(size.x, size.y, size.z);
  return THREE.MathUtils.clamp(dim * 0.5, 0.1, 5.0);
}

const _SUPPORTS_THICKNESS = _REVISION >= 129;
const _SUPPORTS_ATTEN_COLOR = _REVISION >= 131;
const _SUPPORTS_DISPERSION = _REVISION >= 162;

// ============================================================
// Diamond — BVH ray-traced ShaderMaterial (GLSL3, pure)
// ============================================================
// Why a separate ShaderMaterial instead of injecting into MeshPhysicalMaterial:
// MeshPhysicalMaterial's built-in fragment shader uses GLSL1-style identifiers
// (`gl_FragColor`, `modelMatrix` only in vertex, no fragment-side samplers
// with uint precision). Setting `glslVersion: GLSL3` on it triggers compile
// errors in r146 ("'usampler2D' : No precision specified", "'modelMatrix' :
// undeclared identifier", "'gl_FragColor' : undeclared identifier").
// Pure ShaderMaterial under our own control avoids every one of those.
export function createDiamondBVHShaderMaterial(geometry, params = {}) {
  if (!_bvhLib) return null;
  const { MeshBVH, MeshBVHUniformStruct, shaderStructs, shaderIntersectFunction } = _bvhLib;

  let bvh = geometry.boundsTree;
  if (!bvh) {
    try {
      bvh = new MeshBVH(geometry, { maxLeafTris: 10 });
      geometry.boundsTree = bvh;
    } catch (err) {
      console.warn('[Shader] MeshBVH build failed — caller should fall back:', err);
      return null;
    }
  }

  const bvhStruct = new MeshBVHUniformStruct();
  bvhStruct.updateFrom(bvh);

  const {
    ior = 2.417,
    bounces = 5,              // +1 bounce → richer internal reflection variety
    fringeStrength = 0.12,
    sparkleStrength = 1.0,
    brightness = 1.5,
    color = new THREE.Color(0xffffff),
    fillAmount = 0.0,
  } = params;

  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      // Dynamic cube map from main.js CubeCamera — contains metal prongs + HDR
      // sky. Used by REFLECTION (refl) sampling so the gem reflects its mounting.
      envMap: { value: _dynamicEnvCubeTexture || getFallbackCube() },
      // Clean HDR equirect — used by REFRACTION-exit sampling so rays that left
      // the gem see the studio sky, not an adjacent cluster gem's dark back.
      // Read current module state (not hardcoded fallback): materials created
      // AFTER HDR loads — e.g. the earring, which comes in via MediaPipe setup
      // long after the ring — would otherwise miss setDiamondEnvHDR's one-shot
      // update and sample the wrong-perspective cube, going dark.
      envHDR: { value: _dynamicEnvHDRTexture || _fallbackEnvTexture },
      uUseHDR: { value: _dynamicEnvHDRTexture ? 1 : 0 },
      // 1 = reflection samples the CubeCamera (good for ring — gets prong lines).
      // 0 = reflection samples HDR instead (for earrings or any piece where the
      // cube is captured from the wrong position; otherwise the gem body goes
      // dark because the cube data is from a different place in the scene).
      uUseCubeRefl: { value: 1 },
      uBVH: { value: bvhStruct },
      uBounces: { value: Math.min(8, Math.max(1, Math.floor(bounces))) },
      uIOR: { value: ior },
      uFringe: { value: fringeStrength },
      uSparkle: { value: sparkleStrength },
      uBrightness: { value: brightness },
      uColor: { value: color.clone() },
      uFillAmount: { value: Math.max(0, fillAmount) },
      uModelInv: { value: new THREE.Matrix4() },
      uEnvTwist: { value: new THREE.Matrix3() },
    },
    vertexShader: /* glsl */ `
      out vec3 vWorldPos;
      out vec3 vWorldNormal;
      out vec3 vLocalPos;

      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        vLocalPos = position;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      // Integer samplers in GLSL3 ES have NO DEFAULT PRECISION — must declare.
      // This was the missing bit last time (and the reason the shader never
      // compiled on r146).
      precision highp float;
      precision highp int;
      precision highp sampler2D;
      precision highp usampler2D;
      precision highp isampler2D;

      ${shaderStructs}
      ${shaderIntersectFunction}

      in vec3 vWorldPos;
      in vec3 vWorldNormal;
      in vec3 vLocalPos;

      // samplerCube — fed by the CubeCamera in main.js. Contains the ring's
      // metal reflected back at the diamond (the "prong-inside-stone" look).
      // Used by direct reflection sampling.
      uniform samplerCube envMap;
      // HDR equirect — clean studio backdrop. Used by BVH-exit refraction rays
      // so cluster gems don't see each other's dark backs.
      uniform sampler2D envHDR;
      uniform int uUseHDR;
      uniform int uUseCubeRefl;
      uniform BVH uBVH;
      uniform int uBounces;
      uniform float uIOR;
      uniform float uFringe;
      uniform float uSparkle;
      uniform float uBrightness;
      uniform vec3  uColor;
      uniform float uFillAmount;
      uniform mat4  uModelInv;
      uniform mat3  uEnvTwist;

      // Three.js auto-declares modelMatrix in the VERTEX shader but not the
      // fragment for ShaderMaterial in GLSL3 mode. Declaring it here tells
      // Three.js to populate it with the mesh's matrixWorld per-object.
      uniform mat4 modelMatrix;

      out vec4 fragColor;

      #define PI 3.141592653589793

      // Equirect UV mapping for sampling the HDR sky.
      vec2 equirectUv(vec3 dir) {
        float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
        float v = asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;
        return vec2(u, v);
      }

      // Cube sample — sees metal prongs, adjacent cluster gems, everything.
      // Correct for REFLECTION (what the outside of the stone reflects).
      vec3 envSampleCube(vec3 dir) {
        return texture(envMap, uEnvTwist * normalize(dir)).rgb;
      }

      // HDR sample — pure studio sky. Correct for REFRACTION-EXIT (light that
      // traveled through the gem and is now out in the world). Prevents cluster
      // gems from sampling each other's dark back-facets in the cube.
      vec3 envSampleHDR(vec3 dir) {
        vec3 rd = uEnvTwist * normalize(dir);
        return texture(envHDR, equirectUv(rd)).rgb;
      }

      // Default sampler — used by BVH refractive exit rays. Prefers the HDR
      // when available, so each gem sees clean studio not its neighbors.
      vec3 envSample(vec3 dir) {
        return uUseHDR == 1 ? envSampleHDR(dir) : envSampleCube(dir);
      }
      float hash3d(vec3 p) {
        return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
      }

      vec3 coloredLights(vec3 dir) {
        // Tighter exponent (100) + whiter tints → reference-photo look:
        // bright crystalline pinpoints, not a rainbow carnival.
        float p = 100.0;
        vec3 c = vec3(0.0);
        c += vec3(1.55, 1.50, 1.35) * pow(max(0.0, dot(dir, normalize(vec3( 1.0,  1.0,  0.4)))), p);
        c += vec3(1.30, 1.45, 1.60) * pow(max(0.0, dot(dir, normalize(vec3(-1.0,  0.7,  0.6)))), p);
        c += vec3(1.50, 1.45, 1.50) * pow(max(0.0, dot(dir, normalize(vec3( 0.3, -0.8,  1.0)))), p);
        c += vec3(1.40, 1.55, 1.45) * pow(max(0.0, dot(dir, normalize(vec3(-0.4, -0.3, -1.0)))), p);
        c += vec3(1.60, 1.35, 1.25) * pow(max(0.0, dot(dir, normalize(vec3( 0.6,  0.5, -0.9)))), p);
        c += vec3(1.25, 1.45, 1.60) * pow(max(0.0, dot(dir, normalize(vec3(-0.7,  1.0, -0.2)))), p);
        c += vec3(1.60, 1.60, 1.60) * pow(max(0.0, dot(dir, normalize(vec3( 0.0,  0.2,  1.0)))), p);
        c += vec3(1.45, 1.35, 1.55) * pow(max(0.0, dot(dir, normalize(vec3( 0.9, -0.4, -0.2)))), p);
        return c;
      }

      // THE CORE: BVH internal-reflection ray-trace.
      // Ray enters gem (world space → local), bounces via TIR off internal
      // facets until it refracts out or hits the bounce budget, returns
      // WORLD-space exit direction. Sampling the envMap along this direction
      // gives the "arrows and hearts" facet pattern.
      vec3 bvhTrace(vec3 V, vec3 N, float iorVal) {
        float eta = 1.0 / iorVal;
        vec3 refrW = refract(V, N, eta);
        if (length(refrW) < 0.001) return reflect(V, N);

        vec3 rayDir = normalize((uModelInv * vec4(refrW, 0.0)).xyz);
        vec3 rayOrig = vLocalPos + rayDir * 0.001;

        for (int i = 0; i < 8; i++) {
          if (i >= uBounces) break;
          uvec4 faceIdx = uvec4(0u);
          vec3  faceNorm = vec3(0.0);
          vec3  bary = vec3(0.0);
          float side = 1.0;
          float dist = 0.0;
          bvhIntersectFirstHit(uBVH, rayOrig, rayDir, faceIdx, faceNorm, bary, side, dist);
          if (dist <= 0.0) break;
          vec3 hitPos = rayOrig + rayDir * max(dist - 0.001, 0.0);

          vec3 refrNext = refract(rayDir, faceNorm, iorVal);
          if (length(refrNext) > 0.001) {
            rayDir = refrNext;
            break;                               // ray exits the gem
          }
          rayDir = reflect(rayDir, faceNorm);   // total internal reflection
          rayOrig = hitPos + rayDir * 0.001;
        }

        return normalize((modelMatrix * vec4(rayDir, 0.0)).xyz);
      }

      void main() {
        vec3 V = normalize(vWorldPos - cameraPosition);
        vec3 N = normalize(vWorldNormal);
        vec3 refl = reflect(V, N);

        float cosI = max(0.0, dot(-V, N));
        float F0 = 0.17;
        float fresn = F0 + (1.0 - F0) * pow(1.0 - cosI, 5.0);

        // Per-wavelength BVH trace → true chromatic dispersion through N bounces.
        // envSample() → HDR (clean studio) so the refractive body reads bright
        // even in cluster rings where the cube would show neighbor-gem darkness.
        float df = uFringe * 0.06;
        vec3 exitR = bvhTrace(V, N, uIOR * (1.0 - df));
        vec3 exitG = bvhTrace(V, N, uIOR);
        vec3 exitB = bvhTrace(V, N, uIOR * (1.0 + df));
        vec3 bvhColor = vec3(envSample(exitR).r, envSample(exitG).g, envSample(exitB).b);

        // Rim reflection. For rings (uUseCubeRefl = 1) sample the CubeCamera
        // so the stone reflects its own metal prongs. For earrings (= 0) fall
        // back to HDR because the cube is captured at the ring's position and
        // would give garbage reflections from the earring's viewpoint.
        vec3 reflColor = uUseCubeRefl == 1 ? envSampleCube(refl) : envSample(refl);

        // Fresnel mix: refraction (body) ↔ reflection (rim)
        vec3 col = mix(bvhColor, reflColor, fresn);

        // Contrast CRUNCH — 2.0: slightly softer than 2.2 to restore midtone
        // variation on cluster stones (they were reading bimodal, pure-white
        // or pure-dark, with no shading in between).
        col = pow(max(col, vec3(0.0)), vec3(2.0));

        // Mobile fill compensation: when the bounce budget is lower, some
        // facets lose secondary internal light. Lift only the darkest facets
        // so the stone stays bright while preserving contrast and fire.
        float fillMask = (1.0 - smoothstep(0.02, 0.20, dot(col, vec3(0.2126, 0.7152, 0.0722)))) * uFillAmount;
        col += vec3(0.10, 0.105, 0.11) * fillMask;

        col *= uColor * uBrightness;

        // Procedural studio-light pinpoints — bumped to x1.5/x0.9 for a
        // more obvious "bright hit" on specific facets
        col += (coloredLights(refl) * 1.5 + coloredLights(refract(V, N, 1.0 / uIOR)) * 0.9) * uSparkle;

        // Facet-edge scintillation via normal derivatives
        vec3 dN = abs(dFdx(N)) + abs(dFdy(N));
        float edge = length(dN);
        float edgeSp = smoothstep(0.005, 0.10, edge);
        col += vec3(1.3, 1.28, 1.22) * edgeSp * uSparkle * 1.0;

        // View-dependent twinkle — threshold 0.98 (was 0.96) makes flashes
        // RARER but SHARPER: only the top 2% of cells flash, so each flash
        // reads as a distinct star burst instead of fuzzy ambient sparkle.
        vec3 twSeed = floor(vWorldPos * 40.0) + floor(V * 10.0);
        float tw = hash3d(twSeed);
        float twFlash = pow(max(tw - 0.98, 0.0) / 0.02, 2.0);
        col += vec3(1.5, 1.45, 1.35) * twFlash * uSparkle * 3.0;

        // Rim rainbow — stronger (× 0.7 vs 0.4) so the prism effect is visible
        vec3 rimRainbow = vec3(
          sin(fresn * 12.0 + 0.0),
          sin(fresn * 12.0 + 2.1),
          sin(fresn * 12.0 + 4.2)
        ) * 0.5 + 0.5;
        col += rimRainbow * fresn * fresn * uFringe * 0.7;

        // Rim punch
        col += col * fresn * 0.4;

        // Quadratic highlight booster — pixels that are already bright get
        // squared (scaled), pushing them hard past the ACES tonemap knee so
        // they BLAZE instead of just glow. Dark pixels stay dark.
        // 0.06 (was 0.12): cluster stones were saturating their brightest 5%
        // of facets into pure white with no detail. Halved so peaks still pop
        // but don't bleach out surrounding tonal variation.
        col += col * col * 0.06;

        fragColor = vec4(col, 1.0);

        // Tonemap + encoding (use Three.js chunks via gl_FragColor alias)
        // In GLSL3 ShaderMaterial, Three.js aliases gl_FragColor → pc_fragColor
        // when we declare our own 'out vec4' named differently; to be safe,
        // apply the tonemap manually using the same ACES approximation.
        // --- ACES Filmic approximation (matches r146's ACESFilmicToneMapping) ---
        vec3 x = fragColor.rgb * 0.6;
        vec3 aces = (x * (2.51 * x + vec3(0.03))) / (x * (2.43 * x + vec3(0.59)) + vec3(0.14));
        fragColor.rgb = clamp(aces, vec3(0.0), vec3(1.0));
        // Linear → sRGB (matches outputEncoding = sRGBEncoding)
        fragColor.rgb = pow(fragColor.rgb, vec3(1.0 / 2.2));
      }
    `,
    transparent: false,
    side: THREE.FrontSide,
    toneMapped: false,   // we did our own ACES above
  });

  material.onBeforeRender = function (_r, _s, _c, _g, object) {
    material.uniforms.uModelInv.value.copy(object.matrixWorld).invert();
  };

  material.userData.isDiamondBVH = true;   // so updateDiamondEnvTwist can find it
  material.userData.diamondUniforms = {
    uEnvTwist: material.uniforms.uEnvTwist, // alias for scintillation API
  };

  // Register so setDiamondEnvCube() can swap in the live cube texture when
  // main.js wires up the CubeCamera. Until then the uniform already points at
  // _dynamicEnvCubeTexture (if set) or the black fallback cube.
  _diamondMaterialsWithEnvMap.add(material);
  return material;
}

export function createDiamondPhysicalMaterial(geometry, params = {}) {
  // IOR 2.417 — slightly off from physical 2.42 to produce sharp chromatic
  // fringes at facet edges (user tuning request — marks the "fire" sweet spot).
  const iorValue = params.ior ?? 2.417;
  const sparkleStrength = params.sparkleStrength ?? 0.85;
  const fringeStrength = params.fringeStrength ?? 0.30;

  const mat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(params.color ?? 0xffffff),
    metalness: 0.0,
    roughness: params.roughness ?? 0.0,     // must stay 0 for sharp facet highlights
    transmission: params.transmission ?? 0.9,
    ior: iorValue,
    envMapIntensity: params.envMapIntensity ?? 1.6, // peak-attraction tuning
    clearcoat: 0.0,
    clearcoatRoughness: params.clearcoatRoughness ?? 0.0,
    emissive: new THREE.Color(0xffffff),
    emissiveIntensity: params.emissiveIntensity ?? 0.0,
    // transparent:false + transmission>0 → Three.js routes this material through
    // the transmission render pass, where it samples the opaque backdrop RT
    // with IOR-offset UVs. That's how real refraction through the hand works.
    transparent: false,
    side: THREE.FrontSide,
  });

  // MeshPhysicalMaterial.transmission gives us the hand refracted through the
  // gem. Sparkle overlay layered on top via onBeforeCompile adds pinpoint
  // studio lights + edge scintillation + twinkle + rim rainbow.
  // BVH ray-trace is NOT applied here — it lives in createDiamondBVHShaderMaterial
  // because MeshPhysicalMaterial + glslVersion:GLSL3 is broken on r146.
  mat.userData.diamondUniforms = {
    uDiamondSparkle: { value: sparkleStrength },
    uDiamondFringe: { value: fringeStrength },
    uDiamondIOR: { value: iorValue },
    uEnvTwist: { value: new THREE.Matrix3() },
  };

  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, mat.userData.diamondUniforms);

    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
         varying vec3 vDiamondWPos;
         varying vec3 vDiamondWNormal;`
      )
      .replace(
        '#include <fog_vertex>',
        `#include <fog_vertex>
         vDiamondWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
         vDiamondWNormal = normalize(mat3(modelMatrix) * objectNormal);`
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
         varying vec3 vDiamondWPos;
         varying vec3 vDiamondWNormal;
         uniform float uDiamondSparkle;
         uniform float uDiamondFringe;
         uniform float uDiamondIOR;
         uniform mat3  uEnvTwist;

         float diamondHash3(vec3 p) {
           return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
         }

         // Eight virtual studio lights at fixed world directions, each with a
         // distinct spectral tint. Real diamonds show DIFFERENT COLORS on
         // different facets — one facet flashes yellow, the next flashes blue.
         // With 8 colored lights, whichever facet points toward light N catches
         // that specific tint, giving natural color variation across the gem.
         // Sharp pow(dot,80) → tight pinpoint peaks (sparkle, not wash).
         vec3 diamondColoredLights(vec3 dir) {
           // Whiter tints + tighter exponent (100) → clean bright pinpoints like
           // a studio product shot. Each facet still catches its own hue, just
           // de-saturated toward white so the gem reads crystalline, not rainbow.
           float p = 100.0;
           vec3 c = vec3(0.0);
           c += vec3(1.55, 1.50, 1.35) * pow(max(0.0, dot(dir, normalize(vec3( 1.0,  1.0,  0.4)))), p);  // warm
           c += vec3(1.30, 1.45, 1.60) * pow(max(0.0, dot(dir, normalize(vec3(-1.0,  0.7,  0.6)))), p);  // cool blue
           c += vec3(1.50, 1.45, 1.50) * pow(max(0.0, dot(dir, normalize(vec3( 0.3, -0.8,  1.0)))), p);  // rose
           c += vec3(1.40, 1.55, 1.45) * pow(max(0.0, dot(dir, normalize(vec3(-0.4, -0.3, -1.0)))), p);  // mint
           c += vec3(1.60, 1.35, 1.25) * pow(max(0.0, dot(dir, normalize(vec3( 0.6,  0.5, -0.9)))), p);  // amber
           c += vec3(1.25, 1.45, 1.60) * pow(max(0.0, dot(dir, normalize(vec3(-0.7,  1.0, -0.2)))), p);  // sky
           c += vec3(1.60, 1.60, 1.60) * pow(max(0.0, dot(dir, normalize(vec3( 0.0,  0.2,  1.0)))), p);  // white
           c += vec3(1.45, 1.35, 1.55) * pow(max(0.0, dot(dir, normalize(vec3( 0.9, -0.4, -0.2)))), p);  // lilac
           return c;
         }`
      )
      // Inject sparkle additions right before the final output chunk, so they
      // layer on top of the PBR transmission result.
      .replace(
        '#include <output_fragment>',
        `{
          vec3 N_raw_d = normalize(vDiamondWNormal);
          // Env-twist rotation (set from JS per frame, tied to ring rotation × 2
          // for faster-than-hand scintillation). Identity when not updated.
          vec3 N_d = normalize(uEnvTwist * N_raw_d);
          vec3 V_d = normalize(vDiamondWPos - cameraPosition);
          vec3 refl_d = reflect(V_d, N_d);
          float cosI_d = max(0.0, dot(-V_d, N_d));
          float fresn_d = pow(1.0 - cosI_d, 5.0);

          // ==== Contrast CRUNCH ====
          // 2.0 — slightly softer than 2.2 so cluster stones retain midtone
          // facet shading instead of going bimodal white/dark.
          outgoingLight = pow(max(outgoingLight, vec3(0.0)), vec3(2.0));

          // ==== Spectrally-tinted studio-light specular ====
          vec3 lightsRefl = diamondColoredLights(refl_d);
          vec3 lightsRefr = diamondColoredLights(refract(V_d, N_d, 1.0 / uDiamondIOR));
          outgoingLight += (lightsRefl * 1.0 + lightsRefr * 0.6) * uDiamondSparkle;

          // ==== Facet-edge scintillation via derivatives ====
          vec3 dN_d = abs(dFdx(N_d)) + abs(dFdy(N_d));
          float edge_d = length(dN_d);
          float edgeSp_d = smoothstep(0.005, 0.10, edge_d);
          outgoingLight += vec3(1.2, 1.18, 1.15) * edgeSp_d * uDiamondSparkle * 0.8;

          // ==== Big sparse fire flashes ====
          // Threshold 0.98 (was 0.96) → only 2% of cells flash — rarer but
          // SHARPER pinpoints that read as individual star bursts.
          vec3 twSeed_d = floor(vDiamondWPos * 40.0) + floor(V_d * 10.0);
          float tw_d = diamondHash3(twSeed_d);
          float twFlash_d = pow(max(tw_d - 0.98, 0.0) / 0.02, 2.0);
          outgoingLight += vec3(1.4, 1.38, 1.32) * twFlash_d * uDiamondSparkle * 2.5;

          // ==== Fresnel-modulated RGB fringe — SHARPENED ====
          // Narrow offset (× 0.25 instead of × 0.6) gives tight rainbow
          // fringes at facet edges only — crisper, less blurry than before.
          #ifdef USE_ENVMAP
            vec3 reflR_d = reflect(V_d, N_d + dN_d * 0.25);
            vec3 reflB_d = reflect(V_d, N_d - dN_d * 0.25);
            vec3 fringe_d = vec3(
              textureCubeUV(envMap, reflR_d, 0.0).r,
              textureCubeUV(envMap, refl_d,  0.0).g,
              textureCubeUV(envMap, reflB_d, 0.0).b
            );
            outgoingLight += (fringe_d - dot(fringe_d, vec3(0.333))) * fresn_d * uDiamondFringe;
          #endif

          // ==== Rim rainbow hue shift (subtle) ====
          vec3 rimRainbow = vec3(
            sin(fresn_d * 12.0 + 0.0),
            sin(fresn_d * 12.0 + 2.1),
            sin(fresn_d * 12.0 + 4.2)
          ) * 0.5 + 0.5;
          outgoingLight += rimRainbow * fresn_d * fresn_d * uDiamondFringe * 0.4;

          // Rim fresnel punch — restored to × 0.4 for a crisp white outline
          // that separates the gem from the skin in AR.
          outgoingLight += outgoingLight * fresn_d * 0.4;
        }
        #include <output_fragment>`
      );
  };

  // needed so Three.js re-caches the program with our onBeforeCompile changes
  mat.customProgramCacheKey = () => `diamond_${iorValue}_${sparkleStrength}_${fringeStrength}`;
  if (_SUPPORTS_THICKNESS) {
    mat.thickness = typeof params.thickness === 'number' ? params.thickness : autoThickness(geometry);
    // Shorter attenuation distance → stronger whitening of the refracted hand.
    // 10.0 leaves the hand's skin tone mostly intact (looks like brown glass);
    // 1.5 pushes it toward white (looks like a cloudy-bright diamond).
    mat.attenuationDistance = params.attenuationDistance ?? 1.5;
  }
  if (_SUPPORTS_ATTEN_COLOR) {
    mat.attenuationColor = new THREE.Color(params.attenuationColor ?? 0xffffff);
  }
  if (_SUPPORTS_DISPERSION && typeof params.dispersion === 'number') {
    mat.dispersion = params.dispersion;
  }
  mat.name = 'diamond-pbr';
  return mat;
}

// ============================================================
// Metal presets — user-facing tint options for the ring/earring bands.
// Values are perceptually tuned to look correct through ACES tonemap at
// envMapIntensity ~1.8. Keep hex ≤0xffffff; use clearcoat to get the "polished"
// look, not over-bright color.
// ============================================================
export const METAL_PRESETS = {
  silver: 0xf5f5f8,   // white gold / platinum — very slight cool cast
  gold: 0xffd89b,   // warm yellow gold (softer than raw 0xffd700 — ACES eats the saturation otherwise)
  'rose-gold': 0xeec0b0,   // rose gold — pink-bronze blush
};

// Apply a preset (or raw hex) to every METAL material in a group. Diamond
// materials are skipped (detected via userData marker + name). Color-only —
// metalness / roughness / clearcoat are untouched, so a gold ring keeps the
// same polish as silver.
export function setMetalColor(group, preset) {
  if (!group) return;
  const hex = typeof preset === 'string' ? (METAL_PRESETS[preset] ?? 0xf5f5f8) : preset;
  const color = new THREE.Color(hex);
  group.traverse((node) => {
    if (!node.isMesh || !node.material) return;
    const mats = Array.isArray(node.material) ? node.material : [node.material];
    for (const m of mats) {
      if (!m || !m.color) continue;
      // Skip diamond materials — they carry our diamondUniforms marker, or
      // one of the three known diamond material names.
      if (m.userData?.diamondUniforms || m.userData?.isDiamondBVH) continue;
      if (m.name === 'diamond-bvh' || m.name === 'diamond-pbr' || m.name === 'diamond-shader') continue;
      // Only touch true metals — skip anything non-metallic.
      if (typeof m.metalness === 'number' && m.metalness < 0.5) continue;
      m.color.copy(color);
      // Keep the mobile emissive-lift tinted to the current metal color.
      // Without this, picking gold while emissive was initialized as silver
      // leaves the emissive shining silver light on top of the gold diffuse
      // — the ring looks like gold with a silver halo. Copying color into
      // emissive keeps the lift the same hue as the metal.
      if (m.emissive) m.emissive.copy(color);
      // Most jewelry GLBs ship with a baked-in albedo texture whose RGB
      // encodes the metal's colour. MeshPhysicalMaterial's final albedo is
      // color × map, so a gold-coloured map turns every color we set back
      // into a slightly-tinted gold — the user sees "no change". Drop the
      // map entirely when the user explicitly picks a metal preset so only
      // our color decides the tint.
      if (m.map) {
        m.map = null;
      }
      // Force a genuinely new shader binary per color.
      //
      // Adreno / Mali / Xclipse / Apple mobile GPU drivers cache uniform
      // state tied to the compiled program object. If two recompiles
      // produce byte-identical GLSL, the driver sometimes links to the
      // same program object and the old uniform cache remains live — so
      // m.color.copy(goldColor) + m.needsUpdate=true updates the JS side
      // but the GPU keeps drawing with the previously cached silver.
      //
      // `customProgramCacheKey` alone fails here because it only changes
      // Three.js's JS-side cache — the driver sees identical source text
      // and can still cache-hit. Injecting a `#define` that embeds the
      // color's hex into the GLSL source guarantees the preprocessed
      // shader is literally different per color → driver linker cannot
      // cache-match → fresh program on GPU → fresh uniform state.
      //
      // The define is unused by the actual shader logic; it's purely a
      // cache-buster. Zero runtime cost; compiles identically fast.
      m.defines = Object.assign({}, m.defines || {}, {
        METAL_COLOR_V: hex.toString(16)
      });
      m.customProgramCacheKey = () => `metal_${hex.toString(16)}`;
      m.needsUpdate = true;
    }
  });
}

// ============================================================
// applyJewelryShading
// ============================================================
export function applyJewelryShading(group, opts = {}) {
  const metal = opts.metal || {};
  const diamond = opts.diamond || {};
  if (opts.diamondPattern instanceof RegExp) _diamondPatternOverride = opts.diamondPattern;
  // PBR MeshPhysicalMaterial.transmission is now the default — it refracts
  // the in-scene video backdrop, which is what produces a real diamond look.
  // Custom shader kept as opt-in via diamond.useCustomShader === true.
  const usePBR = diamond.useCustomShader !== true;

  let diamondCount = 0;
  let metalCount = 0;
  const diamondNames = [];
  const metalNames = [];

  // Per-group material instancing. Cluster rings often contain many Mesh
  // nodes that share the same BufferGeometry (GLTF instancing, cloned
  // prongs, etc). Previously each such mesh got its OWN ShaderMaterial —
  // which in three.js r146 means its own compiled WebGL program, its own
  // uniform upload, and a full material-state change between draw calls.
  //
  // Keying the cache on geometry means two meshes with the same geometry
  // share one material object → one program, one uniform block, one
  // material-state change shared across draws. Saves 10-30% GPU time on
  // cluster rings where half a dozen prongs are identical.
  //
  // Cache is scoped to this applyJewelryShading() call so different rings
  // don't cross-pollute (each ring's shading settings are independent).
  const bvhMaterialByGeom = new Map();
  const pbrMaterialByGeom = new Map();
  const shaderMaterialByGeom = new Map();

  group.traverse((node) => {
    if (!node.isMesh || !node.material) return;

    const oldMats = Array.isArray(node.material) ? node.material : [node.material];
    const newMats = oldMats.map((m) => {
      if (isDiamondMesh(node, m)) {
        diamondCount++;
        diamondNames.push(`${node.name || '?'} → ${m.name || '(unnamed)'}`);

        if (usePBR) {
          // If BVH library is available AND the caller didn't veto BVH,
          // prefer the pure ShaderMaterial BVH path. It does real internal
          // ray-trace ("arrows and hearts") — the definitive diamond look.
          // Falls back to MeshPhysicalMaterial + sparkle overlay if BVH build
          // throws or the lib isn't loaded.
          if (_bvhLib && diamond.useBVH !== false) {
            // Reuse if another mesh in this group already got a BVH material
            // for this exact geometry (BVH struct is geometry-specific, so
            // sharing is safe and avoids rebuilding the bounds hierarchy).
            let bvhMat = bvhMaterialByGeom.get(node.geometry);
            if (!bvhMat) {
              bvhMat = createDiamondBVHShaderMaterial(node.geometry, {
                ior: diamond.ior,
                bounces: diamond.bounces,
                fringeStrength: diamond.fringeStrength,
                sparkleStrength: diamond.sparkleStrength,
                brightness: diamond.envMapIntensity,
                color: diamond.color,
                fillAmount: diamond.fillAmount,
              });
              if (bvhMat) {
                bvhMat.name = m.name || 'diamond-bvh';
                // useCubeReflection=false → gem samples HDR for its reflection
                // instead of the CubeCamera. Use this for any piece whose
                // position doesn't match the cube's capture point (earrings,
                // necklaces — cube lives at the ring).
                if (diamond.useCubeReflection === false) {
                  bvhMat.uniforms.uUseCubeRefl.value = 0;
                }
                bvhMaterialByGeom.set(node.geometry, bvhMat);
              }
            }
            if (bvhMat) return bvhMat;
          }

          // PBR fallback — also shared per geometry (thickness auto-derives
          // from geometry bounds inside createDiamondPhysicalMaterial, so
          // two identical geometries produce identical materials anyway).
          let pbrMat = pbrMaterialByGeom.get(node.geometry);
          if (!pbrMat) {
            pbrMat = createDiamondPhysicalMaterial(node.geometry, {
              transmission: diamond.transmission,
              ior: diamond.ior,
              roughness: diamond.roughness,
              envMapIntensity: diamond.envMapIntensity,
              thickness: diamond.thickness,
              attenuationDistance: diamond.attenuationDistance,
              attenuationColor: diamond.attenuationColor,
              clearcoatRoughness: diamond.clearcoatRoughness,
              dispersion: diamond.dispersion,
              color: diamond.color,
              sparkleStrength: diamond.sparkleStrength,
              fringeStrength: diamond.fringeStrength,
            });
            pbrMaterialByGeom.set(node.geometry, pbrMat);
          }
          return pbrMat;
        }

        let d = shaderMaterialByGeom.get(node.geometry);
        if (!d) {
          d = createDiamondShaderMaterial({
            ior: diamond.ior ?? 2.42,
            dispersion: diamond.dispersion ?? 0.08,
            brightness: diamond.envMapIntensity ?? 1.1,
            color: new THREE.Color(diamond.color ?? 0xffffff),
            sparkleStrength: diamond.sparkleStrength ?? 1.2,
            causticStrength: diamond.causticStrength ?? 0.5,
          });
          d.name = m.name || 'diamond-shader';
          shaderMaterialByGeom.set(node.geometry, d);
        }
        // Render gems after opaque metal so transparent blending works correctly
        // (caller applies renderOrder on the group; per-material bump is safe here).
        return d;
      }

      metalCount++;
      metalNames.push(`${node.name || '?'} → ${m.name || '(unnamed)'}`);
      const originalIsMetallic = typeof m.metalness === 'number' ? m.metalness > 0.3 : true;
      // metal.color (string preset or hex number) wins over the GLB's baked
      // color — so the user's Silver/Gold/Rose-gold selection applies on load.
      // If not provided, fall back to the GLB color, then to a safe gold default.
      const metalColor = metal.color !== undefined
        ? new THREE.Color(
          typeof metal.color === 'string' ? (METAL_PRESETS[metal.color] ?? metal.color) : metal.color
        )
        : ((m.color && m.color.clone) ? m.color.clone() : new THREE.Color(0xffd700));
      // If the user explicitly picked a metal preset, skip the GLB's baked
      // albedo texture — otherwise MeshPhysicalMaterial would compute
      // final = color × map, and a gold-coloured baked map would mute every
      // color we set. Normal/roughness/metalness maps are still preserved for
      // surface detail; only the color-carrying albedo is dropped.
      const preserveAlbedoMap = metal.color === undefined;
      const newMat = new THREE.MeshPhysicalMaterial({
        color: metalColor,
        map: preserveAlbedoMap ? (m.map || null) : null,
        normalMap: m.normalMap || null,
        roughnessMap: m.roughnessMap || null,
        metalnessMap: m.metalnessMap || null,
        metalness: metal.metalness ?? (originalIsMetallic ? 1.0 : 0.0),
        roughness: metal.roughness ?? 0.08,
        envMapIntensity: metal.envMapIntensity ?? 2.4,
        clearcoat: metal.clearcoat ?? 0.6,
        clearcoatRoughness: metal.clearcoatRoughness ?? 0.05,
        // Optional emissive lift — used on mobile to ensure thin band geometry
        // survives sub-pixel rasterization at the lower mobile pixel ratio.
        // Set via metal.emissive / metal.emissiveIntensity by the caller;
        // tinted to the metal color so the "lift" never reads as a different
        // color (gold stays gold, silver stays silver, just slightly brighter).
        emissive: metal.emissive !== undefined
          ? new THREE.Color(metal.emissive)
          : metalColor.clone(),
        emissiveIntensity: metal.emissiveIntensity ?? 0,
      });
      newMat.name = m.name || 'metal';

      // Subtle Ambient Occlusion / Base Darkening injection.
      // Darkens pixels that are close to the center of the ring (the finger hole).
      // This grounds the model and prevents the "floating" look.
      newMat.onBeforeCompile = (shader) => {
        shader.vertexShader = shader.vertexShader.replace(
          '#include <common>',
          `#include <common>
           varying vec3 vAOPos;`
        ).replace(
          '#include <begin_vertex>',
          `#include <begin_vertex>
           vAOPos = position;`
        );
        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <common>',
          `#include <common>
           varying vec3 vAOPos;`
        ).replace(
          '#include <opaque_fragment>',
          `#include <opaque_fragment>
           // Ring hole is along the Y axis (normalized in buildRingGroup).
           // Darken based on proximity to the central axis (XZ distance).
           float aoDist = length(vAOPos.xz);
           float ao = smoothstep(0.35, 0.55, aoDist);
           gl_FragColor.rgb *= mix(0.65, 1.0, ao);`
        );
      };

      return newMat;
    });

    node.material = newMats.length === 1 ? newMats[0] : newMats;
  });

  const bvhPath = _bvhLib ? 'BVH ray-trace' : 'sparkle overlay (BVH lib unavailable)';
  console.log(`[Shader] applyJewelryShading → diamonds: ${diamondCount}, metals: ${metalCount} — path: ${bvhPath}`);
  if (diamondNames.length) console.log('[Shader]   diamonds:', diamondNames);
  if (metalNames.length) console.log('[Shader]   metals:', metalNames);
  if (diamondCount === 0) {
    console.warn('[Shader] No diamonds detected. Pass opts.diamondPattern = /<regex>/i to target gems by name.');
  }
}
