Here's a full breakdown of all the occluder logic currently in place:                                                                              
                                                            
  ---                                                                                                                                                
  Occluder System — 3 Layers                                                                                                                         
                                                                                                                                                     
  Layer 1: Main Ring Occluder (occluderMesh)                                                                                                         
                                                            
  What it is: A single CylinderGeometry(0.5, 0.5, 2, 32) mesh.

  Material: colorWrite: false, depthWrite: true — completely invisible to the camera, but writes to the depth buffer. It "punches a hole" so the ring
   geometry that falls behind the finger is hidden.

  How it's positioned: Same targetPos and targetOccQuat as the ring itself — it follows the ring exactly, but uses the pure tracking basis (no slider
   rotation offsets) so it always aligns with the actual finger, not the ring's display rotation.

  How it's sized:
  // Width = fingerRadius × 2.2 (covers full finger diameter + small margin)
  targetOccScale.x = fingerRadiusScene * 2.2 * occluderBase * widthMul
  targetOccScale.z = fingerRadiusScene * 2.2 * occluderBase * widthMul

  // Height = foreshorten-stable world segment length × 0.7
  // (worldFingerLen × (palmSpanScene / worldPalmSpan) × 0.7)
  targetOccScale.y = occluderHeightScale

  Z placement: renderOrder = -1 — renders before the ring, so depth values are written first.

  ---
  Layer 2: Blocker Occluders (blockerOccluderMeshes)

  What it is: 10 additional cylinder meshes, one per BLOCKER_SEGMENTS entry (2 segments per finger × 5 fingers).

  Purpose: Occlude the ring when a neighboring finger passes in front of it (e.g., middle finger blocking the ring finger).

  Logic in updateBlockerOccluders():

  For each blocker segment:
    1. Skip if it belongs to the ACTIVE finger (that's Layer 1's job)
    2. Compute depthDelta = depth(this segment) − depth(ring finger)
    3. If depthDelta ≤ 0.004 → skip (this finger is BEHIND the ring finger, won't block)
    4. Place cylinder at 2D midpoint of the segment
    5. Z = min(0.35, 0.05 + depthDelta × 4.0)  ← closer blocking finger = higher Z = renders more in front
    6. Scale = segmentLength2D × 0.7 height, fingerWidth × occluderBase for radius

  renderOrder = -2 (renders before even Layer 1).

  ---
  Layer 3: Full Hand Occluder (handOccluder.js)

  What it is: A full procedural hand mesh loaded from handOccluder.js using landmark-driven capsule segments for all 5 fingers + palm.

  When active: Only when "Show Hand Mesh" debug checkbox is checked. In production it's hidden (hideHandOccluders() is always called unless the
  checkbox is on).

  Purpose: Debug visualization only — shows the reconstructed hand surface. Not used for depth masking in production.

  ---
  Key Design Decisions

  ┌─────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                  Decision                   │                                             Reason                                             │
  ├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ colorWrite: false                           │ Occluder is completely invisible — only writes depth                                           │
  ├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ renderOrder = -1 / -2                       │ Ensures occluder depth pass happens before ring is drawn                                       │
  ├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Height uses worldFingerLen ×                │ Foreshorten-stable: cylinder stays the right visual length on tilted hands                     │
  │ sceneUnitsPerWorld                          │                                                                                                │
  ├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Width uses fingerRadiusScene * 2.2          │ Covers full finger diameter + small buffer so ring edges that dip behind the finger are        │
  │                                             │ properly hidden                                                                                │
  ├─────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Blocker depth check > 0.004                 │ Small threshold avoids flickering when two fingers are nearly coplanar                         │
  └─────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────┘
