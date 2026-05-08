# AR Ring Pro — Zero-Drift Tracking

Real-time AR jewelry virtual try-on. MediaPipe hand/face tracking overlays 3D rings and earrings on a live camera feed with physics-based diamond rendering, chromatic dispersion, and procedural hand occlusion.

## Running the Project

Serve `index.html` over HTTPS (MediaPipe requires a secure context):

```bash
python3 -m http.server 8000
# or
npx http-server -p 8000 --ssl
```

Open `https://localhost:8000` in Chrome/Edge ≥ 90. Allow camera access.

## Architecture

### Threading Model
- `handWorker.js` — Dedicated Web Worker running MediaPipe `HandLandmarker` inference (30–80ms per frame on mobile). Keeps main thread unblocked so render loop runs at vsync even during slow inference. Falls back to CPU delegate if GPU worker fails.
- `faceTracking.js` — `FaceLandmarker` on the main thread, lazy-activated only when earrings are enabled.
- **Tracker mutex**: MediaPipe can run HandLandmarker OR FaceLandmarker per frame, not both. The product picker switches which tracker is active.

### Render Loop (`main.js`)
```
animate()
  → hand tracking updates
  → ring/earring pose computation
  → occlusion pass (depth write only)
  → diamond cube capture (every Nth frame)
  → render scene (ring + earring + video backdrop)
  → post-processing (bloom + SS AO contact shadows)
```

### Hand Occlusion System (3 layers)

**Layer 1 — Main ring occluder** (`occluderMesh`)
- Single `colorWrite: false, depthWrite: true` cylinder at the ring anchor
- `renderOrder = -1` so it renders before ring geometry
- Scaled to `fingerRadius × 2.2` to cover the full diameter

**Layer 2 — Blocker occluders** (`blockerOccluderMeshes`)
- 10 cylinders (2 per finger) that hide the ring when neighboring fingers pass in front
- Depth-delta check (`> 0.004`) prevents coplanar flicker
- `renderOrder = -2`

**Layer 3 — Procedural hand mesh** (`handOccluder.js`)
- Full landmark-driven capsule reconstruction; only active in debug mode ("Show Hand Mesh")

**Key invariant:** Occluders write depth but NOT color — invisible depth masking hides ring geometry behind fingers without leaving visible artifacts.

### Diamond Shader (`Shader.js`)

Two implementations (only GLSL1 is used in production):

1. **GLSL1 Chromatic Dispersion** (active) — Samples equirect HDR at three different IORs for red/green/blue rays. Produces "fire" (rainbow sparkle). Compiles everywhere without dependency issues.
2. **BVH Ray-Trace** (`Shader.ts`, reference only) — True internal reflections via `three-mesh-bvh`. Failed to link on Three.js r146. Kept as reference; not used.

**Dynamic cube environment:** A 64×64 `CubeCamera` renders the scene (minus video/diamonds to avoid feedback) every 6 frames. Diamonds sample this cube so reflections show real metal prongs. Skips frames when ring is stationary (pos delta < 0.5 mm, quat dot > 0.9998).

### Post-Processing
- `UnrealBloomPass` for sparkle/highlights
- Custom `AlphaRestoreShader` composites bloom with film grain (ring pixels only) and SS AO contact shadows (skin pixels only)
- SS AO: 16-sample Poisson disk on a separate ring-mask render target → darkens skin under the ring without affecting the ring itself

### Performance (Mobile)
| Setting | Value | Reason |
|---|---|---|
| Target FPS | 30 | Matches camera cadence; faster just repeats stale frames |
| Pixel ratio cap | 2× | Even on 3× screens |
| Cube update stride | 6 frames | ~20ms per capture; amortized |
| MSAA on bloom RT | off | Saves bandwidth |
| Antialias on main renderer | off | Major speedup on Intel UHD |

## File Map

| File | Role |
|---|---|
| `index.html` | Entry point, Three.js canvas, UI layout |
| `main.js` | Core render loop, hand tracking, occlusion updates, post-FX |
| `Shader.js` | Diamond/metal shading, HDR loading, cube environment |
| `Shader.ts` | BVH ray-trace reference (not linked, not used) |
| `handOccluder.js` | Procedural hand mesh, landmark-driven bone rigging |
| `handWorker.js` | Web Worker wrapper for MediaPipe HandLandmarker |
| `faceTracking.js` | Face tracking, earring pose, One-Euro filters |
| `wristTracking.js` | Stub (no-op; reserved for future wrist jewelry) |
| `style.css` | UI panels, overlays, responsive layout |

## Key Dependencies

- **Three.js r146** — `MeshPhysicalMaterial` (transmission/IOR), `EffectComposer`, `CubeCamera`, `GLTFLoader`, `RGBELoader`
- **MediaPipe Tasks Vision 0.10.0** — `HandLandmarker` (21 keypoints), `FaceLandmarker` (~468 keypoints)
- **three-mesh-bvh 0.5.24** — BVH ray-trace (reference only; doesn't link on r146)

## Common Workflows

### Adding a Ring Model
1. Export as `.glb`, place in `assets/`
2. Add to ring dropdown in `main.js` (`ringSelect` change handler)
3. Confirm diamond node names match the regex: `/diamond|gem|stone|crystal|brilliant|jewel|rhinestone|glass|sapphire|ruby|emerald|quartz|zircon|rock/i`

### Tuning Occlusion
- Ring hides too early/late → adjust `targetOccScale` (`fingerRadius × 2.2`) in `main.js`
- Blocker fingers not activating → check `depthDelta > 0.004` threshold
- Coplanar flicker → increase depth threshold or add hysteresis to blocker Z placement

### Tuning Diamond Appearance
- Brightness: `sparkleStrength` (0.85), `fringeStrength` (0.07)
- Refraction: `ior` (2.417 = diamond)
- Rainbow fire: `dispersion` (0.010)
- Clearcoat sheen: `clearcoatRoughness` (0.0 = mirror)

### Debugging Hand Tracking
- Enable "Show Pose Debug" checkbox → HUD shows landmark confidence and worker delegate (GPU vs CPU)
- Enable "Show Hand Mesh" → overlay of procedural hand to verify occlusion geometry
- Worker posts `detectMs` per frame; > 40ms means tracking is CPU-bound

## Troubleshooting

| Symptom | Check |
|---|---|
| Black screen | HDR load errors in console; `Shader.js` logs "Metal HDR loaded" / "Gem HDR loaded" |
| Ring invisible | "Show Ring" enabled? Diamond regex matches model node names? |
| Tracking jumps | Worker `detectMs` > 40ms; CPU fallback triggered; check `minHandDetectionConfidence` (0.6) |
| Occlusion flicker | Increase `depthDelta` threshold from 0.004; add hysteresis |
| Earring missing | "Show Earring" must be on; face tracking activates lazily (wait 2–3s) |

## Notes

- **V2 branch** — production; active occlusion bug fixes and mobile optimizations
- **Wrist tracking** — `wristTracking.js` stub ready for future wrist-mounted jewelry
- **Multi-hand** — currently tracks one hand (`numHands: 1` in `handWorker.js`); configurable
- **BVH upgrade path** — if upgrading to Three.js r160+, re-evaluate BVH ray-trace linking
