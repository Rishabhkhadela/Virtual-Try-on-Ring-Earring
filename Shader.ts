import * as THREE from "three";
import {
  MeshBVH,
  MeshBVHUniformStruct,
  shaderStructs,
  shaderIntersectFunction,
} from "three-mesh-bvh";
import { createColorFromHex } from "./utils/commonUtils";

const scratchResolution = new THREE.Vector2();

declare module "three" {
  interface BufferGeometry {
    boundsTree?: MeshBVH;
  }
}

const MAX_SHADER_BOUNCES = 8;

export function createDiamondShaderMaterial(
  geometry: THREE.BufferGeometry,
  camera: THREE.Camera,
  envMap: THREE.Texture,
  diamondColor?: string,
  params?: {
    ior?: number;
    bounces?: number;
    aberrationStrength?: number;
    fastChroma?: boolean;
    color?: THREE.Color;
    brightnessMultiplier?: number;
  }
): THREE.ShaderMaterial {
  const {
    ior = 2.4,
    bounces = 4,
    aberrationStrength = 0,
    fastChroma = true,
    color,
    brightnessMultiplier = 1,
  } = params || {};

  const defaultColor = createColorFromHex(diamondColor, brightnessMultiplier);

  const effectiveColor = color ?? defaultColor;

  let bvh = geometry.boundsTree;
  if (!bvh) {
    bvh = new MeshBVH(geometry, { maxLeafTris: 10 });
    geometry.boundsTree = bvh;
  }

  const clampedBounces = Math.min(
    MAX_SHADER_BOUNCES,
    Math.max(1, Math.floor(bounces))
  );

  const material = new THREE.ShaderMaterial({
    uniforms: {
      envMap: { value: envMap },
      bvh: { value: new MeshBVHUniformStruct() },
      projectionMatrixInv: { value: camera.projectionMatrixInverse },
      viewMatrixInv: { value: camera.matrixWorld },
      resolution: {
        value: new THREE.Vector2(window.innerWidth, window.innerHeight),
      },
      bounceCount: { value: clampedBounces },
      ior: { value: ior },
      color: { value: effectiveColor },
      fastChroma: { value: fastChroma },
      aberrationStrength: { value: aberrationStrength },
      modelMatrixInv: { value: new THREE.Matrix4() },
      envRotation: { value: new THREE.Matrix3() },
      brightness: { value: 1 },
    },
    defines: {
      CHROMATIC_ABERRATIONS: fastChroma ? 0 : 1,
      MAX_BOUNCES: MAX_SHADER_BOUNCES,
    },
    vertexShader: `
      varying vec3 vWorldPosition;
      varying vec3 vNormal;
      varying vec3 vLocalPosition;
      uniform mat4 viewMatrixInv;

      void main() {
        vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
        // normal in world space
        vNormal = normalize((viewMatrixInv * vec4(normalMatrix * normal, 0.0)).xyz);
        vLocalPosition = position;
        gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      #define RAY_OFFSET 0.001
      precision highp float;
      ${shaderStructs}
      ${shaderIntersectFunction}

      varying vec3 vWorldPosition;
      varying vec3 vNormal;
      varying vec3 vLocalPosition;

      uniform sampler2D envMap;
      uniform int bounceCount;
      uniform BVH bvh;
      uniform float ior;
      uniform vec3 color;
      uniform bool fastChroma;
      uniform mat4 projectionMatrixInv;
      uniform mat4 viewMatrixInv;
      uniform mat4 modelMatrix;
      uniform mat4 modelMatrixInv;
      uniform vec2 resolution;
      uniform float aberrationStrength;
      uniform mat3 envRotation;
      uniform float brightness;

      // Convert direction to equirectangular UV
      vec2 equirectUv(in vec3 dir) {
        float u = atan(dir.z, dir.x) / (2.0 * 3.141592653589793) + 0.5;
        float v = asin(clamp(dir.y, -1.0, 1.0)) / 3.141592653589793 + 0.5;
        return vec2(u, v);
      }

      vec4 envSample(sampler2D map, vec3 dir) {
        // Apply rotation to the direction before sampling
        vec3 rotatedDir = envRotation * normalize(dir);
        vec2 uv = equirectUv(rotatedDir);
        return texture(map, uv);
      }

      vec3 worldToLocalDir(vec3 dir) {
        return normalize((modelMatrixInv * vec4(dir, 0.0)).xyz);
      }

      vec3 localToWorldDir(vec3 dir) {
        return normalize((modelMatrix * vec4(dir, 0.0)).xyz);
      }

      vec3 totalInternalReflection(vec3 dirWorld, vec3 normalWorld, float iorVal) {
        float eta = 1.0 / iorVal;
        vec3 refractedWorld = refract(dirWorld, normalWorld, eta);
        vec3 initialWorldDir = refractedWorld;
        if (length(refractedWorld) == 0.0) {
          initialWorldDir = reflect(dirWorld, normalWorld);
        }

        vec3 rayDirection = worldToLocalDir(initialWorldDir);
        vec3 rayOrigin = vLocalPosition + rayDirection * RAY_OFFSET;

        for (int i = 0; i < MAX_BOUNCES; i++) {
          if (i >= bounceCount) {
            break;
          }
          uvec4 faceIdx = uvec4(0u);
          vec3 fNormal = vec3(0.0);
          vec3 bary = vec3(0.0);
          float side = 1.0;
          float dist = 0.0;
          bvhIntersectFirstHit(bvh, rayOrigin, rayDirection, faceIdx, fNormal, bary, side, dist);
          vec3 hitPos = rayOrigin + rayDirection * max(dist - RAY_OFFSET, 0.0);
          if (dist <= 0.0) {
            break;
          }
          vec3 refr = refract(rayDirection, fNormal, iorVal);
          if (length(refr) > 0.0) {
            rayDirection = refr;
            break;
          }
          rayDirection = reflect(rayDirection, fNormal);
          rayOrigin = hitPos + rayDirection * RAY_OFFSET;
        }

        return localToWorldDir(rayDirection);
      }

      void main() {
        vec3 rayDirection = normalize(vWorldPosition - cameraPosition);
        vec3 normalizedNormal = normalize(vNormal);

        // Calculate Fresnel effect for more realistic reflections
        float fresnel = pow(1.0 - abs(dot(-rayDirection, normalizedNormal)), 2.5);
        fresnel = clamp(fresnel * 0.5 + 0.5, 0.0, 1.0);

        vec3 colRGB;
        if (aberrationStrength > 0.0) {
          // do chromatic aberration
          vec3 dirG = totalInternalReflection(rayDirection, normalizedNormal, ior);
          vec3 dirR;
          vec3 dirB;
          float delta = aberrationStrength;
          if (fastChroma) {
            dirR = normalize(dirG + vec3(delta * 0.5));
            dirB = normalize(dirG - vec3(delta * 0.5));
          } else {
            dirR = totalInternalReflection(rayDirection, normalizedNormal, ior * (1.0 - delta));
            dirB = totalInternalReflection(rayDirection, normalizedNormal, ior * (1.0 + delta));
          }
          float r = envSample(envMap, dirR).r;
          float g = envSample(envMap, dirG).g;
          float b = envSample(envMap, dirB).b;
          colRGB = vec3(r, g, b);
        } else {
          vec3 dir0 = totalInternalReflection(rayDirection, normalizedNormal, ior);
          colRGB = envSample(envMap, dir0).rgb;
        }

        // Apply color tint and brightness with Fresnel enhancement
        vec3 finalColor = colRGB * color * brightness;
        finalColor = mix(finalColor, finalColor * 1.5, fresnel * 0.3);

        gl_FragColor = vec4(finalColor, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
    transparent: true,
    toneMapped: true,
  });

  // Provide BVH data to shader
  material.uniforms.bvh.value.updateFrom(bvh);

  material.onBeforeRender = function (
    renderer,
    _scene,
    currentCamera,
    _geometry,
    object
  ) {
    const size = renderer.getDrawingBufferSize(scratchResolution);
    material.uniforms.resolution.value.set(size.x, size.y);
    material.uniforms.viewMatrixInv.value.copy(currentCamera.matrixWorld);
    material.uniforms.projectionMatrixInv.value.copy(
      currentCamera.projectionMatrixInverse
    );
    material.uniforms.modelMatrixInv.value.copy(object.matrixWorld).invert();
  };

  return material;
}

const workingColor = new THREE.Color();

export const updateDiamondMaterialUniforms = (
  material: THREE.ShaderMaterial,
  hex: string | undefined,
  brightnessMultiplier: number,
  ior: number
) => {
  const colorUniform = material.uniforms.color?.value as
    | THREE.Color
    | undefined;
  if (colorUniform) {
    workingColor.set(hex || "#FFFFFF");
    const safeMultiplier = Number.isFinite(brightnessMultiplier)
      ? Math.max(0, brightnessMultiplier)
      : 1;
    if (safeMultiplier !== 1) {
      workingColor.multiplyScalar(safeMultiplier);
    }
    colorUniform.copy(workingColor);
    // No need for needsUpdate - uniform changes are applied automatically
  }
  if (typeof material.uniforms.ior?.value === "number") {
    material.uniforms.ior.value = Number.isFinite(ior) && ior > 0 ? ior : 2.4;
  }
};

/**
 * Creates a rotation matrix for HDR environment rotation
 * Used for both initial material setup and runtime updates
 */
export const createHdrRotationMatrix = (
  rotationX: number,
  rotationY: number,
  rotationZ: number
): THREE.Matrix3 => {
  const euler = new THREE.Euler(
    THREE.MathUtils.degToRad(rotationX),
    THREE.MathUtils.degToRad(rotationY),
    THREE.MathUtils.degToRad(rotationZ),
    "XYZ"
  );
  return new THREE.Matrix3().setFromMatrix4(
    new THREE.Matrix4().makeRotationFromEuler(euler)
  );
};

export const updateDiamondMaterialRotation = (
  material: THREE.ShaderMaterial,
  rotationX: number,
  rotationY: number,
  rotationZ: number
) => {
  if (material.uniforms.envRotation?.value) {
    material.uniforms.envRotation.value.copy(
      createHdrRotationMatrix(rotationX, rotationY, rotationZ)
    );
  }
};