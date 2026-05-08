// maskOccluder.js — Phase 3: pixel-perfect hand depth mask
//
// Reads latestSegMask (Uint8Array, 1 = hand pixel) from the YOLO11 segWorker
// every render frame, uploads it as a single-channel DataTexture, and renders
// a fullscreen plane at world z = PLANE_Z with colorWrite:false / depthWrite:true.
//
// The fragment shader discards non-hand pixels so only the hand silhouette
// reaches the depth buffer.  The ring (renderOrder 0, depthTest true) is then
// automatically occluded wherever its projected depth exceeds the plane's depth.
//
// Depth ordering:
//   renderOrder -1000  videoBackdrop   (depthWrite:false — transparent backdrop)
//   renderOrder    -5  maskOccluder    ← this plane  (stamps coarse hand depth)
//   renderOrder    -2  blockerCylinders              (refine depth per-segment)
//   renderOrder    -1  ringCylinders                 (refine depth at ring finger)
//   renderOrder     0  ring model      (depth-tested against all of the above)
//
// World z convention (camera at z=5, looking toward −z):
//   z =  0.0   ring / all landmark positions
//   z =  0.05  this plane  (just in front of ring → ring at z=0 is occluded)
//   z =  0.35  blocker cylinders max  (closer still → even stronger occlusion)
//
// The plane uses depthTest:false so it stamps hand pixels unconditionally;
// cylinder occluders (rendered later, at higher z) will overwrite with more
// precise, closer depth values where needed.

import { latestSegMask, latestSegMaskW, latestSegMaskH } from './main.js';

// Just in front of the ring (world z=0). Fragments at z < PLANE_Z (e.g. the
// ring top after surfaceLift) will still pass the ring's own depth test.
const PLANE_Z = 0.05;

function _makeTexture(data, w, h) {
  const tex = new THREE.DataTexture(
    data, w, h,
    THREE.RedFormat,          // single-channel — Uint8 hand mask
    THREE.UnsignedByteType,
  );
  tex.minFilter  = THREE.LinearFilter;
  tex.magFilter  = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

export function createMaskOccluder(scene) {
  // Start with a 1×1 all-zero placeholder so the plane is inert until the
  // first YOLO11 result arrives.
  let _tw = 1, _th = 1;

  const material = new THREE.ShaderMaterial({
    uniforms: {
      tMask: { value: _makeTexture(new Uint8Array([0]), 1, 1) },
    },
    vertexShader: /* glsl */`
      varying vec2 vUv;
      void main() {
        vUv        = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      uniform sampler2D tMask;
      varying vec2      vUv;
      void main() {
        // The camera feed is CSS-mirrored (scaleX(-1)); flip u to match.
        float hand = texture2D(tMask, vec2(1.0 - vUv.x, vUv.y)).r;
        if (hand < 0.5) discard;          // background pixel — leave buffer alone
        gl_FragColor = vec4(0.0);         // never written (colorWrite:false)
      }
    `,
    colorWrite: false,   // invisible — depth write only
    depthWrite: true,    // stamps z=PLANE_Z into the depth buffer at hand pixels
    depthTest:  false,   // write unconditionally; cylinders refine later
    side: THREE.FrontSide,
  });

  // PlaneGeometry(2,2) spans NDC [-1,1] × [-1,1] exactly for the ortho camera
  // (left=-1, right=1, top=1, bottom=-1).  Placing it at z=PLANE_Z lets Three.js
  // project the correct depth through the ortho projection matrix automatically.
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  mesh.position.z    = PLANE_Z;
  mesh.frustumCulled = false;
  mesh.renderOrder   = -5;
  scene.add(mesh);

  // ── per-frame update ────────────────────────────────────────────────────────

  function _replaceTex(data, w, h) {
    material.uniforms.tMask.value.dispose();
    material.uniforms.tMask.value = _makeTexture(data, w, h);
    _tw = w;
    _th = h;
  }

  function update() {
    const mask = latestSegMask;

    if (!mask) {
      // No hand detected — reset to 1×1 black so all fragments discard.
      if (_tw !== 1 || _th !== 1) {
        _replaceTex(new Uint8Array([0]), 1, 1);
      } else {
        // Already 1×1; just zero the single pixel and re-upload.
        material.uniforms.tMask.value.image.data[0] = 0;
        material.uniforms.tMask.value.needsUpdate   = true;
      }
      return;
    }

    const w = latestSegMaskW;
    const h = latestSegMaskH;

    if (w !== _tw || h !== _th) {
      // Resolution changed — allocate a new texture.
      _replaceTex(mask, w, h);
    } else {
      // Same size — reuse the existing texture object, push new pixel data.
      material.uniforms.tMask.value.image.data = mask;
      material.uniforms.tMask.value.needsUpdate = true;
    }
  }

  return { mesh, update };
}
