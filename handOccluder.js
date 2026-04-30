const THREE_NS = globalThis.THREE;

if (!THREE_NS) {
  throw new Error('handOccluder.js requires global THREE to be loaded first.');
}

const THREE = THREE_NS;

const DEFAULT_ASSET_PATH = 'assets/handOccluder.glb';
const EPSILON = 1e-5;

const BONE_SEGMENTS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
];

const BONE_NAME_ALIASES = new Map([
  ['wrist', 0],
  ['hand', 0],
  ['hand.r', 0],
  ['palm.01', 5],
  ['palm.02', 9],
  ['palm.03', 13],
  ['palm.04', 17],
  ['thumb_cmc', 1],
  ['thumb_mcp', 2],
  ['thumb_ip', 3],
  ['thumb_tip', 4],
  ['index_mcp', 5],
  ['index_pip', 6],
  ['index_dip', 7],
  ['index_tip', 8],
  ['middle_mcp', 9],
  ['middle_pip', 10],
  ['middle_dip', 11],
  ['middle_tip', 12],
  ['ring_mcp', 13],
  ['ring_pip', 14],
  ['ring_dip', 15],
  ['ring_tip', 16],
  ['pinky_mcp', 17],
  ['pinky_pip', 18],
  ['pinky_dip', 19],
  ['pinky_tip', 20],
  ['f_index.01', 6],
  ['f_index.02', 7],
  ['f_index.03', 8],
  ['thumb.01', 2],
  ['thumb.02', 3],
  ['thumb.03', 4],
  ['f_middle.01', 10],
  ['f_middle.02', 11],
  ['f_middle.03', 12],
  ['f_ring.01', 14],
  ['f_ring.02', 15],
  ['f_ring.03', 16],
  ['f_pinky.01', 18],
  ['f_pinky.02', 19],
  ['f_pinky.03', 20],
  ['thumbmetacarpal', 1],
  ['thumbproximal', 2],
  ['thumbdistal', 3],
  ['thumbtip', 4],
  ['indexproximal', 5],
  ['indexintermediate', 6],
  ['indexdistal', 7],
  ['indextip', 8],
  ['middleproximal', 9],
  ['middleintermediate', 10],
  ['middledistal', 11],
  ['middletip', 12],
  ['ringproximal', 13],
  ['ringintermediate', 14],
  ['ringdistal', 15],
  ['ringtip', 16],
  ['littleproximal', 17],
  ['littleintermediate', 18],
  ['littledistal', 19],
  ['littletip', 20],
  ['pinkyproximal', 17],
  ['pinkyintermediate', 18],
  ['pinkydistal', 19],
  ['pinkytip', 20],
  ['bone_armature', 0],
  ['bone.001_armature', 2],
  ['bone.002_armature', 3],
  ['bone.003_armature', 4],
  ['bone.004_armature', 5],
  ['bone.005_armature', 6],
  ['bone.006_armature', 7],
  ['bone.007_armature', 8],
  ['bone.008_armature', 9],
  ['bone.009_armature', 10],
  ['bone.010_armature', 11],
  ['bone.011_armature', 12],
  ['bone.012_armature', 13],
  ['bone.013_armature', 14],
  ['bone.014_armature', 15],
  ['bone.015_armature', 16],
  ['bone.016_armature', 17],
  ['bone.017_armature', 18],
  ['bone.018_armature', 19],
  ['bone.019_armature', 20],
]);

const PREFIX_ALIAS_MIN_LENGTH = 5;

function normalizeBoneName(name) {
  return String(name).trim().toLowerCase().replace(/\s+/g, '_');
}

function stripHandednessPrefix(name) {
  return name.replace(/^(?:l|r|left|right)[_.-]+/, '');
}

function compactBoneName(name) {
  return name.replace(/[_.-]/g, '');
}

function shouldIgnoreBoneName(name) {
  const normalized = stripHandednessPrefix(normalizeBoneName(name));
  return normalized.includes('_weightfix') ||
    /^thumbdistal\.\d+$/.test(normalized) ||
    /^(?:index|middle|ring|little|pinky)metacarpal$/.test(compactBoneName(normalized));
}

function isAliasPrefixMatch(name, alias) {
  if (alias.length < PREFIX_ALIAS_MIN_LENGTH || !name.startsWith(alias)) return false;
  const next = name.charAt(alias.length);
  return next === '' || next === '_' || next === '-' || next === '.';
}

function buildDebugMaterial() {
  return new THREE.MeshBasicMaterial({
    color: 0x00e5ff,
    transparent: true,
    opacity: 0.55,
    depthTest: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    skinning: true,
    wireframe: true,
  });
}

function buildOccluderMaterial() {
  return new THREE.MeshBasicMaterial({
    colorWrite: false,
    depthWrite: true,
    depthTest: true,
    side: THREE.DoubleSide,
    skinning: true,
  });
}

function parseLandmarkIndex(name) {
  if (!name) return null;
  if (shouldIgnoreBoneName(name)) return null;

  const normalized = normalizeBoneName(name);
  const sideAgnostic = stripHandednessPrefix(normalized);
  const compact = compactBoneName(sideAgnostic);

  for (const candidate of [normalized, sideAgnostic, compact]) {
    if (BONE_NAME_ALIASES.has(candidate)) return BONE_NAME_ALIASES.get(candidate);
  }

  for (const [prefix, index] of BONE_NAME_ALIASES.entries()) {
    if (isAliasPrefixMatch(normalized, prefix) || isAliasPrefixMatch(sideAgnostic, prefix)) {
      return index;
    }
  }

  const match = normalized.match(/(?:^|[_-])(?:lm|mp|landmark|joint|bone)?[_-]?(\d{1,2})(?:$|[_-])/);
  if (!match) return null;

  const index = parseInt(match[1], 10);
  return Number.isFinite(index) && index >= 0 && index <= 20 ? index : null;
}

function makePalmNormal(worldLandmarks, out) {
  const wrist = worldLandmarks[0];
  const indexMcp = worldLandmarks[5];
  const pinkyMcp = worldLandmarks[17];
  if (!wrist || !indexMcp || !pinkyMcp) return null;

  const a = new THREE.Vector3(
    -(indexMcp.x - wrist.x),
    -(indexMcp.y - wrist.y),
    -(indexMcp.z - wrist.z)
  );
  const b = new THREE.Vector3(
    -(pinkyMcp.x - wrist.x),
    -(pinkyMcp.y - wrist.y),
    -(pinkyMcp.z - wrist.z)
  );

  out.crossVectors(a, b);
  if (out.lengthSq() < EPSILON) return null;
  return out.normalize();
}

export async function initializeHandOccluder({
  scene,
  assetPath = DEFAULT_ASSET_PATH,
  procedural = false,
  debug = false,
  renderOrder = -10,
  mapToScene = null,
  log = () => {},
} = {}) {
  const state = {
    scene,
    assetPath,
    procedural,
    renderOrder,
    mapToScene,
    log,
    group: new THREE.Group(),
    root: null,
    meshes: [],
    bonesByLandmark: new Map(),
    restBasisByLandmark: new Map(),
    restRootBasis: null,
    restPalmWidth: 1,
    liveRootPosition: new THREE.Vector3(),
    livePalmWidthA: new THREE.Vector3(),
    livePalmWidthB: new THREE.Vector3(),
    livePalmAnchor: new THREE.Vector3(),
    tempRestAnchor: new THREE.Vector3(),
    tempPalmNormal: new THREE.Vector3(),
    tempForward: new THREE.Vector3(),
    tempUp: new THREE.Vector3(),
    tempRight: new THREE.Vector3(),
    tempParentQuat: new THREE.Quaternion(),
    tempWorldQuat: new THREE.Quaternion(),
    tempDeltaQuat: new THREE.Quaternion(),
    tempInvRestQuat: new THREE.Quaternion(),
    tempRootQuat: new THREE.Quaternion(),
    tempInvRootRestQuat: new THREE.Quaternion(),
    tempMatrix: new THREE.Matrix4(),
    debugMaterial: buildDebugMaterial(),
    occluderMaterial: buildOccluderMaterial(),
    mode: debug ? 'debug' : 'occluder',
    ready: false,
    visible: false,
    ringAnchor: null,
    proceduralGeometry: null,
    proceduralPoints: Array.from({ length: 21 }, () => new THREE.Vector3()),
    proceduralAcross: new THREE.Vector3(),
    proceduralForward: new THREE.Vector3(),
    proceduralCenter: new THREE.Vector3(),
  };

  state.group.name = 'HandOccluderRoot';
  state.group.visible = false;
  state.group.renderOrder = renderOrder;
  scene.add(state.group);

  const api = {
    async ready() {
      return loadModel(state);
    },
    setMode(mode) {
      state.mode = mode === 'debug' ? 'debug' : 'occluder';
      applyMode(state);
    },
    setVisible(visible) {
      state.visible = !!visible;
      state.group.visible = !!visible && state.ready;
    },
    hide() {
      state.visible = false;
      state.group.visible = false;
    },
    updatePose(results, options = {}) {
      if (!state.ready || !results?.worldLandmarks?.length || !results?.landmarks?.length) return false;
      if (state.procedural) return updateProceduralPose(state, results, options);

      const worldLandmarks = results.worldLandmarks[0];
      const palmNormal = makePalmNormal(worldLandmarks, state.tempPalmNormal);
      if (!palmNormal) return false;

      const rootBasis = makeRootBasis(worldLandmarks, palmNormal, state);
      if (!rootBasis || !state.restRootBasis) return false;

      state.tempInvRootRestQuat.copy(state.restRootBasis).invert();
      state.tempRootQuat.copy(rootBasis).multiply(state.tempInvRootRestQuat);
      state.group.quaternion.copy(state.tempRootQuat);

      if (state.mapToScene) {
        const landmarks = results.landmarks[0];
        state.mapToScene(landmarks[0], state.liveRootPosition);
        state.mapToScene(landmarks[5], state.livePalmWidthA);
        state.mapToScene(landmarks[17], state.livePalmWidthB);

        state.livePalmAnchor.copy(state.liveRootPosition)
          .add(state.livePalmWidthA)
          .add(state.livePalmWidthB)
          .multiplyScalar(1 / 3);
        state.group.position.copy(state.livePalmAnchor);

        const livePalmWidth = state.livePalmWidthA.distanceTo(state.livePalmWidthB);
        const safeRestPalmWidth = Math.max(state.restPalmWidth, EPSILON);
        const scale = livePalmWidth / safeRestPalmWidth;
        state.group.scale.setScalar(scale);
      }

      for (const [startIdx, endIdx] of BONE_SEGMENTS) {
        const bone = state.bonesByLandmark.get(endIdx);
        const restBasis = state.restBasisByLandmark.get(endIdx);
        if (!bone || !restBasis) continue;

        const start = worldLandmarks[startIdx];
        const end = worldLandmarks[endIdx];
        if (!start || !end) continue;

        state.tempForward.set(
          -(end.x - start.x),
          -(end.y - start.y),
          -(end.z - start.z)
        );
        if (state.tempForward.lengthSq() < EPSILON) continue;
        state.tempForward.normalize();

        state.tempUp.copy(palmNormal);
        const projection = state.tempUp.dot(state.tempForward);
        state.tempUp.addScaledVector(state.tempForward, -projection);
        if (state.tempUp.lengthSq() < EPSILON) continue;
        state.tempUp.normalize();

        state.tempRight.crossVectors(state.tempForward, state.tempUp);
        if (state.tempRight.lengthSq() < EPSILON) continue;
        state.tempRight.normalize();

        state.tempMatrix.makeBasis(state.tempRight, state.tempUp, state.tempForward);
        state.tempWorldQuat.setFromRotationMatrix(state.tempMatrix);

        state.tempInvRestQuat.copy(restBasis.worldQuat).invert();
        state.tempDeltaQuat.copy(state.tempWorldQuat).multiply(state.tempInvRestQuat);

        if (bone.parent?.isBone) {
          bone.parent.getWorldQuaternion(state.tempParentQuat);
          state.tempParentQuat.invert();
          bone.quaternion.copy(state.tempParentQuat).multiply(state.tempDeltaQuat);
        } else {
          bone.quaternion.copy(state.tempDeltaQuat);
        }
      }

      state.root?.updateMatrixWorld(true);
      state.group.visible = state.visible;
      return true;
    },
    getRingAnchor() {
      return state.ringAnchor;
    },
    isReady() {
      return state.ready;
    },
  };

  await loadModel(state);
  applyMode(state);
  return api;
}

async function loadModel(state) {
  if (state.ready) return true;

  if (state.procedural) {
    buildProceduralModel(state);
    state.ready = true;
    state.group.visible = state.visible;
    state.log('[HandOccluder] Using landmark-driven procedural hand mesh');
    return true;
  }

  const loader = new THREE.GLTFLoader();

  try {
    const gltf = await new Promise((resolve, reject) => {
      loader.load(state.assetPath, resolve, undefined, reject);
    });

    state.root = gltf.scene;
    state.root.name = state.root.name || 'HandOccluderModel';
    state.group.add(state.root);

    state.root.traverse((node) => {
      if (node.isBone) {
        const landmarkIndex = parseLandmarkIndex(node.name);
        if (landmarkIndex !== null && !state.bonesByLandmark.has(landmarkIndex)) {
          state.bonesByLandmark.set(landmarkIndex, node);
        }
      }

      if (node.isMesh || node.isSkinnedMesh) {
        node.renderOrder = state.renderOrder;
        node.frustumCulled = false;
        state.meshes.push(node);
      }
    });

    centerModelAtTrackedPalm(state);
    captureRestPose(state);
    state.ringAnchor = state.bonesByLandmark.get(14) || state.bonesByLandmark.get(13) || null;
    state.ready = true;
    state.group.visible = state.visible;
    state.log(`[HandOccluder] Loaded ${state.assetPath} with ${state.bonesByLandmark.size} mapped bones`);
    return true;
  } catch (error) {
    state.ready = false;
    state.group.visible = false;
    state.log(`[HandOccluder] Asset not available at ${state.assetPath}: ${error?.message || error}`);
    return false;
  }
}

function captureRestPose(state) {
  state.root.updateMatrixWorld(true);

  state.restRootBasis = captureRootBasis(state);
  state.restPalmWidth = captureRestPalmWidth(state);

  for (const [startIdx, endIdx] of BONE_SEGMENTS) {
    const bone = state.bonesByLandmark.get(endIdx);
    if (!bone) continue;

    const anchorBone = findAnchorBone(state, startIdx, bone);
    if (!anchorBone) continue;

    const parentPos = new THREE.Vector3();
    const bonePos = new THREE.Vector3();
    anchorBone.getWorldPosition(parentPos);
    bone.getWorldPosition(bonePos);

    const forward = bonePos.clone().sub(parentPos);
    if (forward.lengthSq() < EPSILON) continue;
    forward.normalize();

    const worldQuat = new THREE.Quaternion();
    bone.getWorldQuaternion(worldQuat);

    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(worldQuat);
    up.addScaledVector(forward, -up.dot(forward));
    if (up.lengthSq() < EPSILON) {
      up.set(1, 0, 0).addScaledVector(forward, -forward.x);
    }
    up.normalize();

    const right = new THREE.Vector3().crossVectors(forward, up).normalize();
    const basisMatrix = new THREE.Matrix4().makeBasis(right, up, forward);
    const basisQuat = new THREE.Quaternion().setFromRotationMatrix(basisMatrix);

    state.restBasisByLandmark.set(endIdx, {
      worldQuat: basisQuat,
    });
  }
}

function centerModelAtTrackedPalm(state) {
  const wristBone = state.bonesByLandmark.get(0);
  const indexBone = state.bonesByLandmark.get(5);
  const pinkyBone = state.bonesByLandmark.get(17);
  if (!wristBone) return;

  state.tempRestAnchor.set(0, 0, 0);
  const anchorBones = indexBone && pinkyBone
    ? [wristBone, indexBone, pinkyBone]
    : [wristBone];

  for (const bone of anchorBones) {
    const bonePos = new THREE.Vector3();
    bone.getWorldPosition(bonePos);
    state.tempRestAnchor.add(bonePos);
  }

  state.tempRestAnchor.multiplyScalar(1 / anchorBones.length);
  state.root.position.sub(state.tempRestAnchor);
  state.root.updateMatrixWorld(true);
}

function captureRootBasis(state) {
  const wristBone = state.bonesByLandmark.get(0);
  const middleBone = state.bonesByLandmark.get(9) || state.bonesByLandmark.get(10);
  const pinkyBone = state.bonesByLandmark.get(17) || state.bonesByLandmark.get(18);
  const indexBone = state.bonesByLandmark.get(5) || state.bonesByLandmark.get(6);
  if (!wristBone || !middleBone || !pinkyBone || !indexBone) return null;

  const wrist = new THREE.Vector3();
  const middle = new THREE.Vector3();
  const pinky = new THREE.Vector3();
  const index = new THREE.Vector3();
  wristBone.getWorldPosition(wrist);
  middleBone.getWorldPosition(middle);
  pinkyBone.getWorldPosition(pinky);
  indexBone.getWorldPosition(index);

  const forward = middle.sub(wrist);
  if (forward.lengthSq() < EPSILON) return null;
  forward.normalize();

  const palmAcross = pinky.sub(index);
  if (palmAcross.lengthSq() < EPSILON) return null;
  palmAcross.normalize();

  const normal = new THREE.Vector3().crossVectors(palmAcross, forward);
  if (normal.lengthSq() < EPSILON) return null;
  normal.normalize();

  const up = normal.clone().addScaledVector(forward, -normal.dot(forward)).normalize();
  const right = new THREE.Vector3().crossVectors(forward, up).normalize();
  const basisMatrix = new THREE.Matrix4().makeBasis(right, up, forward);
  return new THREE.Quaternion().setFromRotationMatrix(basisMatrix);
}

function captureRestPalmWidth(state) {
  const indexBone = state.bonesByLandmark.get(5) || state.bonesByLandmark.get(6);
  const pinkyBone = state.bonesByLandmark.get(17) || state.bonesByLandmark.get(18);
  if (!indexBone || !pinkyBone) return 1;

  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  indexBone.getWorldPosition(a);
  pinkyBone.getWorldPosition(b);
  return Math.max(a.distanceTo(b), EPSILON);
}

function makeRootBasis(worldLandmarks, palmNormal, state) {
  const wrist = worldLandmarks[0];
  const middle = worldLandmarks[9];
  const index = worldLandmarks[5];
  const pinky = worldLandmarks[17];
  if (!wrist || !middle || !index || !pinky) return null;

  state.tempForward.set(
    -(middle.x - wrist.x),
    -(middle.y - wrist.y),
    -(middle.z - wrist.z)
  );
  if (state.tempForward.lengthSq() < EPSILON) return null;
  state.tempForward.normalize();

  state.tempUp.copy(palmNormal);
  state.tempUp.addScaledVector(state.tempForward, -state.tempUp.dot(state.tempForward));
  if (state.tempUp.lengthSq() < EPSILON) return null;
  state.tempUp.normalize();

  state.tempRight.crossVectors(state.tempForward, state.tempUp);
  if (state.tempRight.lengthSq() < EPSILON) return null;
  state.tempRight.normalize();

  state.tempMatrix.makeBasis(state.tempRight, state.tempUp, state.tempForward);
  return new THREE.Quaternion().setFromRotationMatrix(state.tempMatrix);
}

function findAnchorBone(state, startIdx, bone) {
  if (startIdx === 0) {
    return state.bonesByLandmark.get(0) || bone.parent || null;
  }
  return state.bonesByLandmark.get(startIdx) || bone.parent || null;
}

function buildProceduralModel(state) {
  const geometry = new THREE.BufferGeometry();
  const mesh = new THREE.Mesh(geometry, state.mode === 'debug' ? state.debugMaterial : state.occluderMaterial);
  mesh.name = 'LandmarkHandOccluderMesh';
  mesh.renderOrder = state.renderOrder;
  mesh.frustumCulled = false;

  state.proceduralGeometry = geometry;
  state.root = new THREE.Group();
  state.root.name = 'LandmarkHandOccluder';
  state.root.add(mesh);
  state.group.add(state.root);
  state.meshes.push(mesh);
}

const PROCEDURAL_FINGER_CHAINS = [
  { id: 'thumb', indices: [1, 2, 3, 4], width: 0.115 },
  { id: 'index', indices: [5, 6, 7, 8], width: 0.078 },
  { id: 'middle', indices: [9, 10, 11, 12], width: 0.085 },
  { id: 'ring', indices: [13, 14, 15, 16], width: 0.078 },
  { id: 'pinky', indices: [17, 18, 19, 20], width: 0.064 },
];

function addVertex(vertices, point) {
  vertices.push(point.x, point.y, point.z);
  return (vertices.length / 3) - 1;
}

function pushTri(indices, a, b, c) {
  indices.push(a, b, c);
}

function offsetPoint(point, dir, amount) {
  return new THREE.Vector3(
    point.x + dir.x * amount,
    point.y + dir.y * amount,
    point.z
  );
}

function makeExpandedPoint(point, center, padding) {
  const dx = point.x - center.x;
  const dy = point.y - center.y;
  const length = Math.hypot(dx, dy);
  if (length < EPSILON) return point.clone();

  return new THREE.Vector3(
    point.x + (dx / length) * padding,
    point.y + (dy / length) * padding,
    point.z
  );
}

function convexHull2D(points) {
  if (points.length <= 3) return points.slice();

  const sorted = points.slice().sort((a, b) => (a.x - b.x) || (a.y - b.y));
  const cross = (origin, a, b) =>
    (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);

  const lower = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  }

  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const point = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  }

  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function addPalmHull(vertices, indices, state, palmWidth) {
  const points = state.proceduralPoints;
  const across = state.proceduralAcross;
  const forward = state.proceduralForward;
  const center = state.proceduralCenter;

  center.copy(points[0])
    .add(points[1])
    .add(points[2])
    .add(points[5])
    .add(points[9])
    .add(points[13])
    .add(points[17])
    .multiplyScalar(1 / 7);

  const seed = [
    offsetPoint(points[0], across, -0.24 * palmWidth).addScaledVector(forward, -0.02 * palmWidth),
    offsetPoint(points[0], across, 0.24 * palmWidth).addScaledVector(forward, -0.02 * palmWidth),
    makeExpandedPoint(points[1], center, 0.09 * palmWidth),
    makeExpandedPoint(points[2], center, 0.07 * palmWidth),
    makeExpandedPoint(points[5], center, 0.08 * palmWidth),
    makeExpandedPoint(points[9], center, 0.05 * palmWidth),
    makeExpandedPoint(points[13], center, 0.06 * palmWidth),
    makeExpandedPoint(points[17], center, 0.09 * palmWidth),
  ];
  const boundary = convexHull2D(seed);

  const centerIndex = addVertex(vertices, center);
  const boundaryIndices = boundary.map((point) => addVertex(vertices, point));
  for (let i = 0; i < boundaryIndices.length; i++) {
    pushTri(indices, centerIndex, boundaryIndices[i], boundaryIndices[(i + 1) % boundaryIndices.length]);
  }
}

function addSegmentQuad(vertices, indices, a, b, radiusA, radiusB) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy);
  if (len < EPSILON) return;

  const nx = -dy / len;
  const ny = dx / len;

  const i0 = addVertex(vertices, new THREE.Vector3(a.x + nx * radiusA, a.y + ny * radiusA, a.z));
  const i1 = addVertex(vertices, new THREE.Vector3(a.x - nx * radiusA, a.y - ny * radiusA, a.z));
  const i2 = addVertex(vertices, new THREE.Vector3(b.x - nx * radiusB, b.y - ny * radiusB, b.z));
  const i3 = addVertex(vertices, new THREE.Vector3(b.x + nx * radiusB, b.y + ny * radiusB, b.z));

  pushTri(indices, i0, i1, i2);
  pushTri(indices, i0, i2, i3);
}

function addJointDisc(vertices, indices, center, radius, sides = 8) {
  if (radius < EPSILON) return;

  const centerIndex = addVertex(vertices, center);
  const ring = [];
  for (let i = 0; i < sides; i++) {
    const angle = (i / sides) * Math.PI * 2;
    ring.push(addVertex(vertices, new THREE.Vector3(
      center.x + Math.cos(angle) * radius,
      center.y + Math.sin(angle) * radius,
      center.z
    )));
  }

  for (let i = 0; i < sides; i++) {
    pushTri(indices, centerIndex, ring[i], ring[(i + 1) % sides]);
  }
}

function updateProceduralPose(state, results, options) {
  if (!state.mapToScene || !state.proceduralGeometry) return false;

  const landmarks = results.landmarks[0];
  const worldLandmarks = results.worldLandmarks?.[0] || null;
  if (!landmarks?.length) return false;

  const wristDepth = worldLandmarks?.[0] ? -worldLandmarks[0].z : 0;
  for (let i = 0; i < 21; i++) {
    const point = state.proceduralPoints[i];
    state.mapToScene(landmarks[i], point);
    const landmarkDepth = worldLandmarks?.[i] ? -worldLandmarks[i].z : wristDepth;
    point.z = 0.02 + (landmarkDepth - wristDepth) * 1.8;
  }

  const points = state.proceduralPoints;
  const palmWidth = Math.max(points[5].distanceTo(points[17]), EPSILON);
  state.proceduralAcross.subVectors(points[17], points[5]);
  state.proceduralAcross.z = 0;
  if (state.proceduralAcross.lengthSq() < EPSILON) state.proceduralAcross.set(1, 0, 0);
  else state.proceduralAcross.normalize();

  state.proceduralForward.subVectors(points[9], points[0]);
  state.proceduralForward.z = 0;
  if (state.proceduralForward.lengthSq() < EPSILON) state.proceduralForward.set(0, 1, 0);
  else state.proceduralForward.normalize();

  const vertices = [];
  const indices = [];
  const activeFingerId = options?.activeFingerId || null;
  const skipActiveFinger = state.mode !== 'debug' && activeFingerId;

  addPalmHull(vertices, indices, state, palmWidth);

  for (const chain of PROCEDURAL_FINGER_CHAINS) {
    if (skipActiveFinger && chain.id === activeFingerId) continue;

    for (let i = 0; i < chain.indices.length - 1; i++) {
      const a = points[chain.indices[i]];
      const b = points[chain.indices[i + 1]];
      const segmentLength = a.distanceTo(b);
      const baseRadius = Math.min(palmWidth * chain.width, segmentLength * 0.42);
      const radiusA = baseRadius * (i === 0 ? 1.08 : 1.0);
      const radiusB = baseRadius * (i === chain.indices.length - 2 ? 0.72 : 0.92);
      addSegmentQuad(vertices, indices, a, b, radiusA, radiusB);
      addJointDisc(vertices, indices, a, radiusA * 0.82, 8);
      if (i === chain.indices.length - 2) addJointDisc(vertices, indices, b, radiusB, 8);
    }
  }

  state.proceduralGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  state.proceduralGeometry.setIndex(indices);
  state.proceduralGeometry.computeBoundingSphere();

  state.group.position.set(0, 0, 0);
  state.group.quaternion.identity();
  state.group.scale.setScalar(1);
  state.group.visible = state.visible;
  return true;
}

function applyMode(state) {
  const material = state.mode === 'debug' ? state.debugMaterial : state.occluderMaterial;
  for (const mesh of state.meshes) {
    mesh.material = material;
  }
  state.group.visible = state.visible && state.ready;
}
