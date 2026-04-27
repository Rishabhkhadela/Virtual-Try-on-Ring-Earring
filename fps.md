FPS Improvement Plan — No Quality Reduction                                                                                                         
                                          
  Given your hardware (Intel UHD 630 desktop, mid-range mobile) and that every "obvious" lever (Web Worker for MediaPipe, material instancing, shader 
  pre-compile, stationary cube skip, rVFC) is already in place, what remains are mostly small-to-medium wins. There's no single change left that
  doubles FPS without touching quality. But stacking 4–6 of these should get desktop from ~10 FPS to ~15–18 FPS on the flower cluster, and mobile from
   ~10–15 to ~20–28.                                                                                                                                  
                                                                                                                                                      
  Ordered by impact / risk ratio — highest impact, lowest risk first.                                                                                 
                                                                                                                                                      
  ---
  Tier 1 — Safe, measurable wins

  1. Collapse the double scene render (desktop only) — est. +30–40% desktop FPS

  Current: desktop renders the full scene TWICE per frame — once to sceneRT for alpha preservation, once through composer.RenderPass for bloom. On the
   UHD 630 this is the single biggest redundant cost.

  Plan: point alphaRestorePass.uniforms.tScene at composer.renderTarget2.texture (which already holds the pre-bloom scene after RenderPass), delete
  the explicit sceneRT render. Also delete sceneRT itself — it's no longer needed.

  Risk: if composer's buffer-swap behavior differs from my analysis, alpha could be wrong on the first frame. Verification: check RGB output and
  video-alpha-blending are identical pre/post change. Easy to revert.

  Quality cost: zero if my analysis is correct.

  ---
  2. Skip cube camera entirely when hand is off-screen — est. +1–2 FPS when hand absent

  Current: updateDiamondReflectionCube runs every frame regardless of whether the ring is even visible. When no hand is detected, the cube still
  captures 6 face renders every 2–4 frames.

  Plan: early-return in updateDiamondReflectionCube when !ringModel.visible || !isHandPresent. Already partially checks ringModel.visible but worth
  tightening.

  Quality cost: zero — reflections only matter when the ring is on screen.

  ---
  3. Skip composer when nothing to draw — est. +5–10% desktop FPS when hand absent

  Current: when isHandPresent=false, the composer still runs BloomPass (6 passes for the mip pyramid) on essentially-empty scene data.

  Plan: short-circuit composer.render() to renderer.render(scene, camera) when there's no ring visible. Video still renders correctly (it's just the
  videoBackdrop plane). Once the hand reappears, composer returns.

  Quality cost: zero — nothing to bloom when no ring.

  ---
  4. Drop desktop MediaPipe input resolution — est. +2–4 FPS desktop

  Current: desktop camera constraint is width: 1280, height: 960. MediaPipe's hand tracking doesn't need this resolution — its model internally
  downsamples to 192×192. Extra pixels cost:
  - Camera decode time (more GPU memory bandwidth)
  - createImageBitmap time (copies full frame for Worker)
  - MediaPipe's internal preprocessing

  Plan: drop to width: 640, height: 480 or 960, 720 on desktop. Landmark precision essentially unchanged (validated empirically). videoBackdrop plane
  still displays at full canvas resolution (texture is scaled up).

  Quality cost: tiny — the video plate in the background becomes fractionally less sharp. Most users won't notice. The ring geometry is rendered at
  full pixel ratio so ring sharpness unchanged.

  ---
  5. Remove console.log calls in hot path — est. +1–3 FPS mobile, 0 desktop

  Current: processResults, buildRingGroup, and cube update functions have console.log calls that fire every frame or on ring swap. Desktop devtools
  handle these cheaply. Mobile Chrome's JS-to-native console bridge is surprisingly slow (~2–5 ms per log on some Androids when devtools-over-USB
  isn't attached).

  Plan: wrap every console.log in a DEBUG flag (module-level const false). Leave console.warn and console.error alone.

  Quality cost: zero — loses debug logs only.

  ---
  6. Cube update stride tuning — est. +3–5% FPS (both platforms)

  Current: desktop stride 2, mobile stride 4. Stationary-skip already bypasses updates when ring is still.

  Plan: bump desktop to stride 3. The stationary skip already makes 90% of frames skip anyway; the stride setting only matters during motion. At
  stride 3, motion-frame cube updates happen at 20 FPS (assuming 60 FPS render) — imperceptible lag since reflections blur-merge across frames anyway.

  Quality cost: near-zero — reflections one frame staler during fast motion, which the sparkle+twinkle overlay already masks.

  ---
  Tier 2 — Moderate impact, low risk

  7. Reduce fill-light count — est. +2–4% FPS

  Current: AmbientLight + DirectionalLight (fallback) + sparkleKey + sparkleRim = 4 lights. Each light adds 2–8 fragment ops to every PBR shader
  invocation. HDR IBL (via scene.environment) does 95% of the lighting work — the directional lights are a fallback for before HDR loads.

  Plan: remove dl1 (0.3 intensity fallback DirectionalLight). Keep sparkle key + rim since they produce visible specular highlights. HDR IBL covers
  the rest. If HDR fails to load, ring will look slightly flatter until HDR is available, but that's a 100ms window anyway.

  Quality cost: imperceptible unless HDR load fails.

  ---
  8. Pre-bake setFromEuler on slider values — est. <1% FPS

  Current: every frame _tmpEuler.set(rotX, rotY, rotZ, 'XYZ'); _tmpQuatA.setFromEuler(_tmpEuler). For a stationary ring preset, this is wasted work.

  Plan: cache the computed quaternion in a module-scope variable. Invalidate only when a slider value changes (via input listener). Read cached value
  per frame.

  Quality cost: zero.

  ---
  9. Trim MeshPhysicalMaterial feature set — est. +2–5% FPS on metal-heavy rings

  Current: metal materials have clearcoat: 1.0 (mobile) / 0.7 (desktop) which adds a second BRDF layer. On PBR pixels that's ~30 extra fragment ops.

  Plan — nuanced: currently mobile needs the clearcoat for thin-band visibility (my previous fix). Desktop doesn't actually need it visually — the
  bloom halo does the same job at lower cost. Drop desktop clearcoat to 0. Mobile stays at 1.0.

  Quality cost (desktop only): metal loses its glossy outer shell. Reads as slightly less polished. Subtle enough most users won't notice but a
  jewelry-focused eye would.

  ---
  10. Shrink PMREM mipmap count — est. +VRAM savings, 0–2% FPS

  Current: PMREM generates 8 mip levels for IBL. That's 8 cube textures at 512px → 256 → 128 → ... The lowest mips are used only for rough surfaces
  (roughness > 0.3). Our metals are roughness 0.04–0.06 — they use the highest-res mip almost exclusively.

  Plan: configure PMREM to generate fewer mips. Three.js's PMREMGenerator doesn't expose this directly, but we can generate PMREM at a smaller source
  texture size (e.g., 512 equirect → 128 cubemap mips) and save VRAM + lookup cache pressure on mobile.

  Quality cost: imperceptible for our metals (they sample top mips).

  ---
  Tier 3 — Bigger changes, higher risk

  11. OffscreenCanvas worker rendering (Phase 2 from before) — est. 0–3 FPS desktop, +5–10 sustained mobile

  Current: all rendering runs on main thread. Mobile phones thermally throttle after 2–3 minutes, dropping the whole main thread's clock speed.

  Plan: move Three.js scene, renderer, and animate loop into a render worker. Main thread only handles video capture, UI events, and MediaPipe (which
  is already in its own worker). Three independent threads = better CPU core spread, less thermal concentration.

  Desktop impact: tiny — desktop CPU is already cold, not thermally limited. UHD 630 GPU is the bottleneck and workers don't touch the GPU directly.

  Mobile impact: the real win. Mobile typically holds sustained FPS much better (no 40% FPS drop at the 2-minute mark).

  Risk: ~300–500 line rewrite. Three.js OffscreenCanvas support is good in Chrome/Brave but flaky in Safari < 16.4. Imports need restructuring. UI
  state flows through postMessage. Ring-swap, metal-pick, and slider changes all need new messaging.

  Quality cost: zero.

  ---
  12. WebGPU renderer (experimental) — est. +20–50% FPS on supported hardware

  Current: we use THREE.WebGLRenderer. Three.js r146 has a preview WebGPURenderer (unstable API, not recommended for prod).

  Plan: conditionally instantiate WebGPURenderer on browsers that support it (navigator.gpu). Fall back to WebGL otherwise.

  Hardware limitations: UHD 630 on Windows has WebGPU through D3D12 — but Intel's WebGPU driver for that generation is known slower than their WebGL
  driver (yes, really). Modern Adreno/Apple mobile GPUs can see 30%+ gain from WebGPU.

  Risk: r146's WebGPURenderer is marked experimental. Post-processing passes (UnrealBloomPass) might not work. BVH shader would need conversion from
  GLSL to WGSL (significant rewrite). Three.js r160+ has much more mature WebGPU — upgrading the Three.js version is its own risk.

  Quality cost: zero when it works, breaks the app entirely when it doesn't.

  ---
  13. Instancing for repeated diamond meshes — est. +5–15% FPS on heavy cluster rings

  Current: my previous material-instancing change deduped the MATERIALS. But each gem is still a separate draw call.

  Plan: walk the GLB scene graph at load time; detect meshes that share geometry (e.g., repeated accent stones in a cluster). Merge them into a single
   InstancedMesh with per-instance transforms.

  Complication: the BVH shader uses modelMatrix to compute uModelInv per draw. Instanced meshes don't have per-instance modelMatrix in the same way —
  you'd need an instanced attribute for the inverse matrix. Moderate shader refactor.

  Quality cost: zero if done correctly.

  ---
  Tier 4 — Last-resort, user has vetoed

  (listing for completeness, not recommending)

  - Half-res diamond RT — I tried, destroyed gem detail. You rejected.
  - Shader LOD / PBR fallback on mobile — You rejected.
  - Pixel ratio reduction — "Quality drops over time" problem.
  - precision lowp float — banding on gradients.
  - Reduce BVH bounces — flatter facet pattern.

  ---
  Recommended execution order

  If you say "do these," I would do them in this order to minimize risk:

  ┌─────┬───────────────────────────────────────┬──────────┬────────────────────────┬──────────────────┐
  │  #  │                Change                 │  Effort  │      Tier 1 Risk       │    Est. Gain     │
  ├─────┼───────────────────────────────────────┼──────────┼────────────────────────┼──────────────────┤
  │ 1   │ Skip cube when hand absent            │ 5 min    │ none                   │ +1–2 FPS (idle)  │
  ├─────┼───────────────────────────────────────┼──────────┼────────────────────────┼──────────────────┤
  │ 2   │ Skip composer when hand absent        │ 5 min    │ none                   │ +5–10% (idle)    │
  ├─────┼───────────────────────────────────────┼──────────┼────────────────────────┼──────────────────┤
  │ 3   │ Remove hot-path console.log           │ 10 min   │ none                   │ +1–3 FPS mobile  │
  ├─────┼───────────────────────────────────────┼──────────┼────────────────────────┼──────────────────┤
  │ 4   │ Cache preset quaternion               │ 15 min   │ none                   │ <1%              │
  ├─────┼───────────────────────────────────────┼──────────┼────────────────────────┼──────────────────┤
  │ 5   │ Drop desktop MediaPipe res to 960×720 │ 5 min    │ none                   │ +2–4 FPS desktop │
  ├─────┼───────────────────────────────────────┼──────────┼────────────────────────┼──────────────────┤
  │ 6   │ Remove dl1 fallback light             │ 5 min    │ none                   │ +2–4%            │
  ├─────┼───────────────────────────────────────┼──────────┼────────────────────────┼──────────────────┤
  │ 7   │ Collapse double scene render          │ 30 min   │ medium (visual verify) │ +30–40% desktop  │
  ├─────┼───────────────────────────────────────┼──────────┼────────────────────────┼──────────────────┤
  │ 8   │ PMREM mip count reduction             │ 20 min   │ low                    │ 0–2% + VRAM      │
  ├─────┼───────────────────────────────────────┼──────────┼────────────────────────┼──────────────────┤
  │ 9   │ Desktop clearcoat → 0                 │ 5 min    │ low (visual verify)    │ +2–5%            │
  ├─────┼───────────────────────────────────────┼──────────┼────────────────────────┼──────────────────┤
  │ 10  │ OffscreenCanvas worker                │ 1–2 days │ high                   │ mobile thermal   │
  └─────┴───────────────────────────────────────┴──────────┴────────────────────────┴──────────────────┘

  Doing #1–#9 takes maybe 2 hours and stacks to give you desktop ~15 FPS (cluster), ~25+ FPS (solitaire); mobile ~25 FPS (cluster), ~35+ FPS
  (solitaire).

  #10 (OffscreenCanvas) is the only remaining architecturally-big move and its impact is mostly felt on mobile during long sessions.

  Tell me which of 1–9 you want done (or "all of them") and I'll execute. I won't touch 10–13 without explicit approval — they're significant
  rewrites.