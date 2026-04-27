 Diamond Shading — Full Parameter Reference (Current State)                                                                                                                                                                                                                                                
  1. Call-site values — main.js ring block                                                                                                               
  applyJewelryShading(ringModel, {                                                                                                                    
    metal: {
      metalness: 1.0,
      roughness: 0.06,
      envMapIntensity: 1.8,
      clearcoat: 0.7,
      clearcoatRoughness: 0.04
    },
    diamond: {
      transmission: 1.0,
      ior: 2.417,
      roughness: 0.0,
      envMapIntensity: 3.5,
      attenuationDistance: 2.5,
      clearcoat: 1.0,
      clearcoatRoughness: 0.0,
      dispersion: 0.010,
      sparkleStrength: 0.85,
      fringeStrength: 0.07
    }
  });

  2. Call-site values — faceTracking.js earring block

  applyJewelryShading(raw, {
    metal: {
      metalness: 1.0,
      roughness: 0.05,
      envMapIntensity: 1.8,
      clearcoat: 0.7,
      clearcoatRoughness: 0.04
    },
    diamond: {
      transmission: 1.0,
      thickness: 0.5,
      ior: 2.417,
      roughness: 0.0,
      envMapIntensity: 3.5,
      attenuationDistance: 2.5,
      clearcoat: 1.0,
      clearcoatRoughness: 0.0,
      dispersion: 0.010,
      sparkleStrength: 0.85,
      fringeStrength: 0.07,
      useCubeReflection: false   // earring samples HDR, not the ring-anchored cube
    }
  });

  ---
  3. Per-parameter meaning + why this value

  ┌─────────────────────┬────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐   
  │      Parameter      │       Value        │                                            What it does                                            │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ transmission        │ 1.0                │ Full light transmission — PBR fallback uses this to route refraction through the video/scene       │   
  │                     │                    │ backdrop. Ignored by BVH path (uses its own raytrace).                                             │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ ior                 │ 2.417              │ Index of refraction. Real diamond = 2.42; the 0.003 offset gives sharp chromatic fringes at facet  │   
  │                     │                    │ edges.                                                                                             │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ roughness           │ 0.0                │ Must be 0 — any value > 0 kills the sharp facet highlights that make a diamond read as a diamond.  │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ envMapIntensity     │ 3.5                │ BVH path: uBrightness multiplier on sampled env. Was 10.0 (blew out facets), dropped to 3.5 for    │   
  │                     │                    │ midtone variation without losing luminance.                                                        │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ attenuationDistance │ 2.5                │ Short light-path through the gem → whitens dim refracted rays. Too low (0.55) = milky. Too high    │   
  │                     │                    │ (10.0) = cloudy dark cluster stones. 2.5 is the "clear but with a white floor" sweet spot.         │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ clearcoat           │ 1.0                │ Full dielectric outer layer → rim-edge brilliance.                                                 │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ clearcoatRoughness  │ 0.0                │ Mirror-smooth clearcoat — no blur on the rim.                                                      │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ dispersion          │ 0.010              │ PBR path only (r162+). Prism fringe for MeshPhysicalMaterial fallback. Matched tight to the BVH    │   
  │                     │                    │ fringe.                                                                                            │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ sparkleStrength     │ 0.85               │ Multiplier on studio-light pinpoints + edge scintillation + twinkle flashes.                       │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ fringeStrength      │ 0.07               │ Rainbow-fringe intensity. Was 0.45 (pink/purple rim wash), now 0.07 — hint of fire only. Also      │   
  │                     │                    │ drives the BVH per-wavelength IOR split via uFringe × 0.06.                                        │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ thickness           │ 0.5 (earring) /    │ Attenuation depth for PBR transmission. Ring auto-computes from geometry; earring manually set.    │   
  │                     │ auto (ring)        │                                                                                                    │   
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤   
  │ useCubeReflection   │ true (ring) /      │ Ring samples the CubeCamera (gets prong-darkness). Earring samples HDR (cube is anchored to ring's │   
  │                     │ false (earring)    │  position, wrong-perspective for ear).                                                             │   
  └─────────────────────┴────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘   

  ---
  4. Shader-side uniforms (BVH path — createDiamondBVHShaderMaterial)

  ┌──────────────┬──────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐   
  │   Uniform    │              Initial value               │                                       Source                                        │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ envMap       │ samplerCube, fed by main.js CubeCamera   │ Cube re-rendered every frame at ring's position, diamonds + video backdrop hidden   │   
  │              │ (128×128 HalfFloat)                      │ during capture. Used by reflection sampling (if uUseCubeRefl = 1).                  │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ envHDR       │ sampler2D, photo_studio_01_1k.hdr        │ Used by refraction-exit rays so cluster gems don't darken each other via            │   
  │              │ equirect                                 │ adjacent-gem cube data.                                                             │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uUseHDR      │ 1 (after HDR loads)                      │ Gates HDR sampling in envSample().                                                  │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uUseCubeRefl │ 1 (ring) / 0 (earring)                   │ Gates cube vs HDR for the reflection path.                                          │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uBVH         │ per-geometry BVH struct                  │ Bounds hierarchy for internal TIR ray-trace.                                        │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uBounces     │ 5 (clamped 1–8)                          │ Internal TIR bounce budget before ray exits.                                        │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uIOR         │ 2.417                                    │ Refraction index inside the ray-trace.                                              │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uFringe      │ 0.07                                     │ Per-wavelength IOR spread for BVH + rim rainbow multiplier.                         │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uSparkle     │ 0.85                                     │ Studio-light + edge + twinkle multiplier.                                           │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uBrightness  │ 3.5                                      │ Final body-color multiplier (= envMapIntensity).                                    │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uColor       │ 0xffffff                                 │ Body tint.                                                                          │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uModelInv    │ auto per-frame                           │ World→local for local-space BVH.                                                    │   
  ├──────────────┼──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ uEnvTwist    │ identity / ring.quaternion × 4           │ Per-frame scintillation — env rotates 4× faster than hand, masks MediaPipe jitter.  │   
  └──────────────┴──────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘   

  ---
  5. In-shader constants (baked values)

  BVH shader (/* fragmentShader */ in createDiamondBVHShaderMaterial)

  ┌───────────────────────────┬───────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐    
  │         Constant          │           Value           │                                       Purpose                                        │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ F0 (fresnel)              │ 0.17                      │ Matches IOR 2.42                                                                     │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Contrast crunch exponent  │ 2.0                       │ Softer than 2.2 — preserves midtone variation on cluster stones. Was 3.8 (bimodal),  │    
  │                           │                           │ 2.2 (slightly harsh).                                                                │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Quadratic highlight       │ × 0.06                    │ Peaks pop without bleaching surrounding detail. Was 0.12 (too hot).                  │    
  │ booster                   │                           │                                                                                      │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Colored-light pow         │ 100                       │ Tight pinpoint highlights.                                                           │    
  │ exponent                  │                           │                                                                                      │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Colored-light refl        │ × 1.5                     │                                                                                      │    
  │ multiplier                │                           │                                                                                      │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Colored-light refr        │ × 0.9                     │                                                                                      │    
  │ multiplier                │                           │                                                                                      │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Edge-sparkle smoothstep   │ 0.005 → 0.10              │ Facet-edge detection via normal derivatives.                                         │    
  │ range                     │                           │                                                                                      │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Edge sparkle multiplier   │ × 1.0                     │                                                                                      │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Twinkle spatial frequency │ floor(pos × 40) + floor(V │ Cell size for flash locations.                                                       │    
  │                           │  × 10)                    │                                                                                      │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Twinkle threshold         │ > 0.98                    │ Only top 2% of cells flash — rare but sharp.                                         │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Twinkle flash multiplier  │ × 3.0                     │                                                                                      │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Rim rainbow multiplier    │ × 0.7                     │                                                                                      │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Rim punch                 │ × 0.4                     │ Fresnel-weighted boost.                                                              │    
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤    
  │ Final ACES scale          │ × 0.6                     │ Pre-ACES multiplier before the tonemap knee.                                         │    
  └───────────────────────────┴───────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘    

  PBR overlay (onBeforeCompile in createDiamondPhysicalMaterial)

  ┌──────────────────────┬───────────────────────────────────┐
  │       Constant       │               Value               │
  ├──────────────────────┼───────────────────────────────────┤
  │ Contrast crunch      │ 2.0                               │
  ├──────────────────────┼───────────────────────────────────┤
  │ Colored-light pow    │ 100                               │
  ├──────────────────────┼───────────────────────────────────┤
  │ Colored-light refl × │ 1.0                               │
  ├──────────────────────┼───────────────────────────────────┤
  │ Colored-light refr × │ 0.6                               │
  ├──────────────────────┼───────────────────────────────────┤
  │ Edge scintillation × │ 0.8                               │
  ├──────────────────────┼───────────────────────────────────┤
  │ Twinkle threshold    │ > 0.98 (window 0.02), flash × 2.5 │
  ├──────────────────────┼───────────────────────────────────┤
  │ RGB fringe offset    │ × 0.25                            │
  ├──────────────────────┼───────────────────────────────────┤
  │ Rim rainbow ×        │ 0.4                               │
  ├──────────────────────┼───────────────────────────────────┤
  │ Rim punch            │ × 0.4                             │
  └──────────────────────┴───────────────────────────────────┘

  ---
  6. Scene-level factors that shape the diamond

  Renderer (configureRenderer)

  ┌─────────────────────────┬──────────────────────────────────────────────────┐
  │         Setting         │                      Value                       │
  ├─────────────────────────┼──────────────────────────────────────────────────┤
  │ outputEncoding          │ sRGBEncoding                                     │
  ├─────────────────────────┼──────────────────────────────────────────────────┤
  │ toneMapping             │ ACESFilmicToneMapping                            │
  ├─────────────────────────┼──────────────────────────────────────────────────┤
  │ toneMappingExposure     │ 1.3 (was 1.8 — caused bloom spillover onto skin) │
  ├─────────────────────────┼──────────────────────────────────────────────────┤
  │ physicallyCorrectLights │ true                                             │
  └─────────────────────────┴──────────────────────────────────────────────────┘

  Lighting

  ┌────────────────────────┬────────────────────────────────────────┬──────────────────────────────────┐
  │         Light          │               Intensity                │               Note               │
  ├────────────────────────┼────────────────────────────────────────┼──────────────────────────────────┤
  │ AmbientLight           │ 0.3                                    │ Fill floor while HDR async-loads │
  ├────────────────────────┼────────────────────────────────────────┼──────────────────────────────────┤
  │ DirectionalLight (dl1) │ 0.4                                    │ Fill key                         │
  ├────────────────────────┼────────────────────────────────────────┼──────────────────────────────────┤
  │ sparkleKey             │ 1.6 at (2.5, 3, 4)                     │ Catchlight                       │
  ├────────────────────────┼────────────────────────────────────────┼──────────────────────────────────┤
  │ sparkleRim             │ 1.2 at (-2.5, -1.5, 3), color 0xffe8d6 │ Rim catchlight                   │
  └────────────────────────┴────────────────────────────────────────┴──────────────────────────────────┘

  HDR environment

  - File: assets/photo_studio_01_1k.hdr (neutral — switched from warm brown_photostudio_02)
  - Usage: PMREM for scene.environment (all PBR materials incl. metal), raw equirect for BVH refraction sampling, seeded into cube RT as base sky.    

  CubeCamera (diamond-local reflections)

  ┌─────────────────────────────────┬─────────────────────────────────────────────┐
  │             Setting             │                    Value                    │
  ├─────────────────────────────────┼─────────────────────────────────────────────┤
  │ Resolution                      │ 128×128                                     │
  ├─────────────────────────────────┼─────────────────────────────────────────────┤
  │ Format                          │ RGBAFormat                                  │
  ├─────────────────────────────────┼─────────────────────────────────────────────┤
  │ Type                            │ HalfFloatType                               │
  ├─────────────────────────────────┼─────────────────────────────────────────────┤
  │ Near / Far                      │ 0.01 / 100                                  │
  ├─────────────────────────────────┼─────────────────────────────────────────────┤
  │ Position                        │ Tracks ringModel.position every frame       │
  ├─────────────────────────────────┼─────────────────────────────────────────────┤
  │ Hidden during capture           │ Diamonds, video backdrop, axes/line helpers │
  ├─────────────────────────────────┼─────────────────────────────────────────────┤
  │ Tone mapping during capture     │ NoToneMapping (preserves linear HDR values) │
  ├─────────────────────────────────┼─────────────────────────────────────────────┤
  │ scene.background during capture │ HDR equirect (temporarily swapped in)       │
  └─────────────────────────────────┴─────────────────────────────────────────────┘

  Bloom post-pass (UnrealBloomPass)

  ┌───────────┬───────┐
  │ Parameter │ Value │
  ├───────────┼───────┤
  │ strength  │ 0.22  │
  ├───────────┼───────┤
  │ radius    │ 0.2   │
  ├───────────┼───────┤
  │ threshold │ 0.90  │
  └───────────┴───────┘

  Scintillation driver (updateDiamondEnvTwist)

  - uEnvTwist = rotation(euler × 4) applied to ring's quaternion each frame. Makes sparkle move 4× faster than the hand; masks tracking jitter.       

  ---
  7. Detection regex (which materials get treated as diamond)

  /diamond|gem|stone|crystal|brilliant|jewel|rhinestone|glass|sapphire|ruby|emerald|quartz|zircon|rock/i

  Plus heuristics:
  - material.transmission > 0.3 → diamond
  - metalness < 0.35 AND roughness < 0.3 → diamond

  ---
  8. Path selection

  BVH lib loaded?  ─yes→  createDiamondBVHShaderMaterial  (default, GLSL3 ray-trace)
         │
         no
         ↓
    usePBR default? ─yes→  createDiamondPhysicalMaterial  (MeshPhysicalMaterial + onBeforeCompile overlay)
         │
         diamond.useCustomShader: true
         ↓
    createDiamondShaderMaterial  (GLSL1 equirect refraction — legacy, opt-in)

  Current build: BVH path is active for both ring and earring (three-mesh-bvh loaded at page start).

  ---
  9. Quick-tune cheatsheet (if you want to nudge things)

  ┌───────────────────────────────┬───────────────────────────────────────────────────┐
  │             Want…             │                      Change                       │
  ├───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Whiter / more opaque body     │ attenuationDistance ↓ (1.5–2.0)                   │
  ├───────────────────────────────┼───────────────────────────────────────────────────┤
  │ More clarity / "sees through" │ attenuationDistance ↑ (5.0+)                      │
  ├───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Brighter body                 │ envMapIntensity ↑ (4.5–6.0)                       │
  ├───────────────────────────────┼───────────────────────────────────────────────────┤
  │ More rainbow fire             │ fringeStrength ↑ (0.15–0.30)                      │
  ├───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Sharper facet split           │ Contrast crunch exponent ↑ (2.5–3.5) in Shader.js │
  ├───────────────────────────────┼───────────────────────────────────────────────────┤
  │ More star-flash twinkle       │ Twinkle threshold ↓ (0.94) in Shader.js           │
  ├───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Harder rim outline            │ Rim punch × fresn ↑ in Shader.js                  │
  ├───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Neutralize warm band          │ metal.envMapIntensity ↓ (1.3–1.5)                 │
  └───────────────────────────────┴───────────────────────────────────────────────────┘
