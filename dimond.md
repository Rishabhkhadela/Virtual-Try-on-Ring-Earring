Diamond Shading Parameters — Full Reference

  1. Call-site values (what the app actually passes in)

  Ring diamonds — main.js → applyJewelryShading(ringModel, { diamond: ... })

  ┌─────────────────────┬───────┬───────────────────────────────────────────────────────┐
  │      Parameter      │ Value │                        Purpose                        │
  ├─────────────────────┼───────┼───────────────────────────────────────────────────────┤
  │ transmission        │ 1.0   │ Full light transmission (PBR path only)               │
  ├─────────────────────┼───────┼───────────────────────────────────────────────────────┤
  │ ior                 │ 2.417 │ Index of refraction — real diamond is 2.42            │
  ├─────────────────────┼───────┼───────────────────────────────────────────────────────┤
  │ roughness           │ 0.0   │ Mirror-sharp facets; any >0 kills sparkle             │
  ├─────────────────────┼───────┼───────────────────────────────────────────────────────┤
  │ envMapIntensity     │ 3.2   │ Brightness multiplier on HDR env reflections          │
  ├─────────────────────┼───────┼───────────────────────────────────────────────────────┤
  │ attenuationDistance │ 0.55  │ Short light-path → white body (was 5.0 = brown glass) │
  ├─────────────────────┼───────┼───────────────────────────────────────────────────────┤
  │ clearcoat           │ 1.0   │ Extra glossy outer layer                              │
  ├─────────────────────┼───────┼───────────────────────────────────────────────────────┤
  │ clearcoatRoughness  │ 0.0   │ Keeps clearcoat mirror-like                           │
  ├─────────────────────┼───────┼───────────────────────────────────────────────────────┤
  │ dispersion          │ 0.025 │ r146+ built-in prism fringe                           │
  ├─────────────────────┼───────┼───────────────────────────────────────────────────────┤
  │ sparkleStrength     │ 0.85  │ Studio-light pinpoint intensity                       │
  ├─────────────────────┼───────┼───────────────────────────────────────────────────────┤
  │ fringeStrength      │ 0.22  │ Rim rainbow + RGB fringe amount                       │
  └─────────────────────┴───────┴───────────────────────────────────────────────────────┘

  Earring diamonds — faceTracking.js

  Same as ring diamonds, plus thickness: 0.5 (manual — ring path uses auto-compute).

  ---
  2. Detection — Shader.js → isDiamondMesh()

  A mesh is auto-treated as diamond if any of these match:
  - Material/mesh name matches regex: /diamond|gem|stone|crystal|brilliant|jewel|rhinestone|glass|sapphire|ruby|emerald|quartz|zircon|rock/i
  - material.transmission > 0.3
  - material.metalness < 0.35 AND material.roughness < 0.3 (dielectric + smooth heuristic)

  Override via opts.diamondPattern = /<regex>/i.

  ---
  3. Three diamond render paths (auto-selected)

  BVH available?  ─yes→  BVH ShaderMaterial  (GLSL3, true ray-trace)
         │
         no
         ↓
    PBR path?  ─yes→  MeshPhysicalMaterial + sparkle overlay (default)
         │
         useCustomShader: true
         ↓
    Custom ShaderMaterial (GLSL1 refraction)

  Current build: BVH path is preferred (three-mesh-bvh is loaded).

  ---
  4. BVH ShaderMaterial — createDiamondBVHShaderMaterial()

  Uniforms

  ┌─────────────┬────────────────────────────┬──────────────────────────────────────────┐
  │   Uniform   │          Default           │                 Meaning                  │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ envMap      │ HDR equirect               │ Environment sampled by exit rays         │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ uBVH        │ per-geometry               │ Bounds hierarchy for ray intersection    │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ uBounces    │ 5 (clamped 1–8)            │ Internal TIR bounces before exit         │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ uIOR        │ 2.417                      │ Refraction index                         │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ uFringe     │ 0.30 default / 0.22 passed │ Per-wavelength IOR split (0.06 × fringe) │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ uSparkle    │ 1.0 default / 0.85 passed  │ Studio-light multiplier                  │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ uBrightness │ 1.5 default / 3.2 passed   │ Final body-color multiplier              │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ uColor      │ 0xffffff                   │ Body tint                                │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ uModelInv   │ auto per-frame             │ World→local for local-space BVH          │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ uEnvTwist   │ identity / driven          │ 2× ring rotation for scintillation       │
  └─────────────┴────────────────────────────┴──────────────────────────────────────────┘

  In-shader constants

  ┌────────────────────────────┬───────────────────────────┬───────────────────────────────────────┐
  │          Constant          │           Value           │                Purpose                │
  ├────────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
  │ F0 (fresnel)               │ 0.17                      │ Matches IOR 2.42                      │
  ├────────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
  │ Contrast crunch pow        │ 1.7                       │ Sharpens TIR-dark / exit-bright split │
  ├────────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
  │ Quadratic highlight boost  │ × 0.12                    │ Pushes hot pixels past ACES knee      │
  ├────────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
  │ Colored-light pow exponent │ 100                       │ Tight pinpoint highlights             │
  ├────────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
  │ Edge-sparkle smoothstep    │ 0.005 → 0.10              │ Facet-edge normal-derivative range    │
  ├────────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
  │ Twinkle frequency          │ floor(pos * 40.0)         │ Spatial cell size for flashes         │
  ├────────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
  │ Twinkle threshold          │ > 0.96                    │ Only 4% of cells flash                │
  ├────────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
  │ Rim rainbow period         │ sin(fresn * 12.0 + phase) │ Prism fringe frequency                │
  ├────────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
  │ Rim punch                  │ × 0.4                     │ Extra fresnel-weighted boost          │
  └────────────────────────────┴───────────────────────────┴───────────────────────────────────────┘

  ---
  5. PBR path — createDiamondPhysicalMaterial()

  MeshPhysicalMaterial properties

  ┌─────────────────────┬────────────────────────────────────────┐
  │      Property       │                 Value                  │
  ├─────────────────────┼────────────────────────────────────────┤
  │ color               │ 0xffffff                               │
  ├─────────────────────┼────────────────────────────────────────┤
  │ metalness           │ 0.0                                    │
  ├─────────────────────┼────────────────────────────────────────┤
  │ roughness           │ 0.0                                    │
  ├─────────────────────┼────────────────────────────────────────┤
  │ transmission        │ 0.9 (caller overrides to 1.0)          │
  ├─────────────────────┼────────────────────────────────────────┤
  │ ior                 │ 2.417                                  │
  ├─────────────────────┼────────────────────────────────────────┤
  │ envMapIntensity     │ 1.6 default / 3.2 passed               │
  ├─────────────────────┼────────────────────────────────────────┤
  │ clearcoat           │ 0.0 (material-level)                   │
  ├─────────────────────┼────────────────────────────────────────┤
  │ emissive            │ 0xffffff                               │
  ├─────────────────────┼────────────────────────────────────────┤
  │ emissiveIntensity   │ 0.0                                    │
  ├─────────────────────┼────────────────────────────────────────┤
  │ thickness           │ auto — clamp(maxDim × 0.5, 0.1, 5.0)   │
  ├─────────────────────┼────────────────────────────────────────┤
  │ attenuationDistance │ 1.5 default / 0.55 passed              │
  ├─────────────────────┼────────────────────────────────────────┤
  │ attenuationColor    │ 0xffffff                               │
  ├─────────────────────┼────────────────────────────────────────┤
  │ dispersion          │ 0.025 (r162+)                          │
  ├─────────────────────┼────────────────────────────────────────┤
  │ transparent         │ false (routes through transmission RT) │
  ├─────────────────────┼────────────────────────────────────────┤
  │ side                │ FrontSide                              │
  └─────────────────────┴────────────────────────────────────────┘

  Sparkle-overlay uniforms (injected via onBeforeCompile)

  ┌─────────────────┬───────────────────┐
  │     Uniform     │       Value       │
  ├─────────────────┼───────────────────┤
  │ uDiamondSparkle │ 0.85              │
  ├─────────────────┼───────────────────┤
  │ uDiamondFringe  │ 0.22              │
  ├─────────────────┼───────────────────┤
  │ uDiamondIOR     │ 2.417             │
  ├─────────────────┼───────────────────┤
  │ uEnvTwist       │ identity / driven │
  └─────────────────┴───────────────────┘

  In-shader constants

  ┌───────────────────────────────┬──────────────────────────┐
  │           Constant            │          Value           │
  ├───────────────────────────────┼──────────────────────────┤
  │ Contrast crunch pow           │ 1.5                      │
  ├───────────────────────────────┼──────────────────────────┤
  │ Colored-light pow exponent    │ 100                      │
  ├───────────────────────────────┼──────────────────────────┤
  │ Colored-light refl multiplier │ × 1.0                    │
  ├───────────────────────────────┼──────────────────────────┤
  │ Colored-light refr multiplier │ × 0.6                    │
  ├───────────────────────────────┼──────────────────────────┤
  │ Edge scintillation multiplier │ × 0.8                    │
  ├───────────────────────────────┼──────────────────────────┤
  │ Twinkle threshold             │ > 0.96 (4%), flash × 2.5 │
  ├───────────────────────────────┼──────────────────────────┤
  │ RGB fringe offset             │ × 0.25 (tight)           │
  ├───────────────────────────────┼──────────────────────────┤
  │ Rim rainbow multiplier        │ × 0.4                    │
  ├───────────────────────────────┼──────────────────────────┤
  │ Rim punch                     │ × 0.4                    │
  └───────────────────────────────┴──────────────────────────┘

  ---
  6. Custom GLSL1 Shader — createDiamondShaderMaterial() (opt-in only)

  Uniforms

  ┌────────────────────┬──────────────────────────────────┐
  │      Uniform       │             Default              │
  ├────────────────────┼──────────────────────────────────┤
  │ envMap             │ HDR equirect                     │
  ├────────────────────┼──────────────────────────────────┤
  │ ior                │ 2.42                             │
  ├────────────────────┼──────────────────────────────────┤
  │ aberrationStrength │ 0.12 (3-channel R/G/B IOR split) │
  ├────────────────────┼──────────────────────────────────┤
  │ brightness         │ 2.8                              │
  ├────────────────────┼──────────────────────────────────┤
  │ color              │ 0xffffff                         │
  ├────────────────────┼──────────────────────────────────┤
  │ envRotation        │ Matrix3 identity                 │
  ├────────────────────┼──────────────────────────────────┤
  │ sparkleStrength    │ 1.5                              │
  ├────────────────────┼──────────────────────────────────┤
  │ causticStrength    │ 0.5                              │
  └────────────────────┴──────────────────────────────────┘

  In-shader constants

  - Env base mix: 60% envColor + 45% white lift
  - Studio-light refr multiplier: × 2.2; refl: × 2.8 × fresnel
  - Studio-light pow exponent: 40
  - RGB fringe per-channel: × 0.8
  - Edge sparkle multiplier: × 1.8
  - Twinkle: spatial floor(pos * 180), flash × 5.0, threshold > 0.92
  - Rim fresnel boost: × 0.5
  - Spectral 7-channel dispersion: weights normalize by / 2.8

  ---
  7. Scene-level factors that affect diamond appearance

  Renderer (configureRenderer)

  ┌─────────────────────────┬───────────────────────┐
  │         Setting         │         Value         │
  ├─────────────────────────┼───────────────────────┤
  │ outputEncoding          │ sRGBEncoding          │
  ├─────────────────────────┼───────────────────────┤
  │ toneMapping             │ ACESFilmicToneMapping │
  ├─────────────────────────┼───────────────────────┤
  │ toneMappingExposure     │ 1.0                   │
  ├─────────────────────────┼───────────────────────┤
  │ physicallyCorrectLights │ true                  │
  └─────────────────────────┴───────────────────────┘

  HDR environment

  - File: assets/brown_photostudio_02_1k.hdr
  - Loaded as both PMREM (for PBR scene.environment) and raw equirect (for custom/BVH shaders)
  - Fallback: 2×2 warm DataTexture until HDR resolves

  Fill / sparkle lights

  - AmbientLight(0xffffff, 0.3)
  - DirectionalLight(0xffffff, 0.4) — neutral fill
  - sparkleKey: DirectionalLight(0xffffff, 1.6) at (2.5, 3, 4)
  - sparkleRim: DirectionalLight(0xffe8d6, 1.2) at (-2.5, -1.5, 3)

  Bloom post-pass (UnrealBloomPass)

  ┌───────────┬───────┐
  │ Parameter │ Value │
  ├───────────┼───────┤
  │ strength  │ 0.22  │
  ├───────────┼───────┤
  │ radius    │ 0.3   │
  ├───────────┼───────┤
  │ threshold │ 0.98  │
  └───────────┴───────┘

  Scintillation driver (updateDiamondEnvTwist)

  - Env rotation = 2 × ring.quaternion (XYZ euler), updated every frame → sparkle moves faster than hand.

  ---
  8. Quick-tune cheat sheet

  ┌────────────────────────────────┬───────────────────────────────────────────────┐
  │             Want…              │                    Change                     │
  ├────────────────────────────────┼───────────────────────────────────────────────┤
  │ Whiter body                    │ attenuationDistance ↓ (0.3–0.7)               │
  ├────────────────────────────────┼───────────────────────────────────────────────┤
  │ More skin-through (glass)      │ attenuationDistance ↑ (2.0–5.0)               │
  ├────────────────────────────────┼───────────────────────────────────────────────┤
  │ Brighter sparkle               │ envMapIntensity ↑                             │
  ├────────────────────────────────┼───────────────────────────────────────────────┤
  │ More rainbow fire              │ dispersion ↑, fringeStrength ↑                │
  ├────────────────────────────────┼───────────────────────────────────────────────┤
  │ Cleaner crystalline look       │ fringeStrength ↓, tighter colored-light tints │
  ├────────────────────────────────┼───────────────────────────────────────────────┤
  │ Sharper dark/light facet split │ Contrast crunch pow ↑ (1.7–2.0)               │
  ├────────────────────────────────┼───────────────────────────────────────────────┤
  │ More star-flash twinkle        │ Twinkle threshold ↓ (e.g. > 0.92)             │
  ├────────────────────────────────┼───────────────────────────────────────────────┤
  │ Harder rim outline             │ Rim punch × fresn ↑                           │
  └────────────────────────────────┴───────────────────────────────────────────────┘
