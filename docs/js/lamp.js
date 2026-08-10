/**
 * The lamp: a three.js scene graph that reproduces MuJoCo's forward
 * kinematics exactly.
 *
 * scripts/export_web_assets.py bakes rest-pose world coordinates into the
 * meshes and extracts, for each joint, the screw axis it rotates about in
 * that same rest frame. Nesting one pivot per joint -- each holding a
 * constant local axis, each carrying the subtree below it -- evaluates the
 * space-form product of exponentials
 *
 *     T_k(q) = exp([S_1] q_1) ... exp([S_k] q_k) . T_k(0)
 *
 * which is the closed form of the real robot's kinematics. The exporter
 * proves the equivalence against MuJoCo over random poses before writing
 * these files, so what you drag around here is the actual LeLamp geometry
 * in the actual joint limits, not an approximation of it.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DEG = Math.PI / 180;
// world-space box the camera always keeps in frame, in metres. The lamp is
// ~0.44 m tall and reaches ~0.35 m across at full yaw.
const FRAME_H = 0.62;
const FRAME_W = 0.46;

export class Lamp {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: false, powerPreference: "high-performance",
    });
    this.renderer.setClearColor(0x0a0f15, 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    // without tone mapping the base's horizontal top face blows out under
    // the key light while the arm's vertical faces stay dark, which reads as
    // two different materials even though the MJCF gives them one
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 4 / 3, 0.05, 40);
    this.camera.up.set(0, 0, 1);            // MuJoCo is Z-up

    this.pivots = [];
    this.glow = [];
    this.q = [0, 0, 0, 0, 0];
    this.ready = false;
  }

  async load(rigUrl, glbUrl) {
    const rig = await (await fetch(rigUrl)).json();
    this.rig = rig;
    const gltf = await new GLTFLoader().loadAsync(glbUrl);

    // index the flat node list the exporter wrote
    const byName = new Map();
    gltf.scene.traverse((o) => { if (o.isMesh) byName.set(o.name, o); });

    // nested pivots, one per joint, in physical order from the base up
    let parent = new THREE.Group();
    this.root = parent;
    this.scene.add(parent);

    const attach = (segIdx, holder) => {
      for (const name of rig.segment_nodes[String(segIdx)] || []) {
        const mesh = byName.get(name);
        if (!mesh) continue;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        holder.add(mesh);
      }
    };

    attach(0, parent);                        // the base: never moves
    rig.joints.forEach((j, i) => {
      const p = new THREE.Vector3().fromArray(j.point);
      const pivot = new THREE.Group();
      pivot.position.copy(p);
      const inner = new THREE.Group();
      inner.position.copy(p).negate();        // net identity at the rest pose
      pivot.add(inner);
      parent.add(pivot);
      pivot.userData.axis = new THREE.Vector3().fromArray(j.axis).normalize();
      this.pivots.push(pivot);
      attach(i + 1, inner);
      parent = inner;
    });

    // the diffuser + shade glow with light01; keep their own materials so
    // the emissive tint does not leak into the rest of the lamp
    const warm = new THREE.Color(...rig.glow_warm);
    this.warm = warm;
    for (const name of rig.glow_nodes) {
      const mesh = byName.get(name);
      if (!mesh) continue;
      mesh.material = mesh.material.clone();
      mesh.material.emissive = warm.clone();
      mesh.material.emissiveIntensity = 0;
      mesh.userData.base = mesh.material.color.clone();
      mesh.castShadow = false;
      this.glow.push(mesh);
    }
    this.bulb = new THREE.PointLight(warm.getHex(), 0, 2.2, 2);
    this.glow[0]?.getWorldPosition(this.bulb.position);
    this.scene.add(this.bulb);

    this._lights();
    this._ground();
    this._camera(rig.camera);
    this.setPose([0, 0, 0, 0, 0], 0.45, [1, 0.85, 0.6]);
    this.ready = true;
    return rig;
  }

  _lights() {
    this.scene.add(new THREE.HemisphereLight(0x9fb6cc, 0x0a0f15, 0.9));
    // low and lateral: a steep key makes the base's top face the brightest
    // thing in frame, which is not how the MuJoCo renders read
    const key = new THREE.DirectionalLight(0xffffff, 0.95);
    key.position.set(-1.1, 0.85, 0.8);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    const s = 0.8;
    key.shadow.camera.left = -s; key.shadow.camera.right = s;
    key.shadow.camera.top = s; key.shadow.camera.bottom = -s;
    key.shadow.camera.near = 0.1; key.shadow.camera.far = 4;
    key.shadow.bias = -0.0015;
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x6f86a8, 0.45);
    rim.position.set(1.2, -0.9, 0.5);
    this.scene.add(rim);
  }

  _ground() {
    const g = new THREE.Mesh(
      new THREE.CircleGeometry(1.1, 64),
      new THREE.MeshStandardMaterial({
        color: 0x121a23, roughness: 0.95, metalness: 0.0,
      }),
    );
    g.receiveShadow = true;
    this.scene.add(g);
  }

  /** MuJoCo's azimuth/elevation/distance/lookat -> a three.js camera. */
  _camera(cam) {
    const az = cam.azimuth * DEG, el = cam.elevation * DEG;
    const fwd = new THREE.Vector3(
      Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el),
    );
    this.target = new THREE.Vector3().fromArray(cam.lookat);
    this.camera.position.copy(this.target).addScaledVector(fwd, -cam.distance);
    this.camera.lookAt(this.target);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.target.copy(this.target);
    this.controls.enablePan = false;
    this.controls.enableZoom = false;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.55;
    this.controls.minPolarAngle = 0.55;
    this.controls.maxPolarAngle = 1.75;
    // One finger must scroll the page, not orbit the lamp -- otherwise the
    // canvas becomes a scroll trap on a phone. Two fingers rotate.
    this.controls.touches = { ONE: null, TWO: THREE.TOUCH.DOLLY_ROTATE };
    this.controls.update();
  }

  /** q: 5 joint angles (rad), light: 0..1, rgb: 0..1 triple. */
  setPose(q, light = 0.5, rgb = [1, 0.85, 0.6]) {
    if (!this.pivots.length) return;
    for (let i = 0; i < this.pivots.length; i++) {
      const p = this.pivots[i];
      p.quaternion.setFromAxisAngle(p.userData.axis, q[i] ?? 0);
    }
    this.q = q;
    // Match LampRenderer: the shade's colour lerps toward the LED colour by
    // light01 rather than being blown out by emissive. A little emissive on
    // top is what makes it read as *lit* against a dark stage.
    // The rgb channel is the LED ring's colour and is genuinely (0,0,0) for
    // most of the corpus -- Cozmo's backpack lights are usually off. Every
    // renderer in the project tints toward a fixed warm instead, so do that,
    // and honour rgb only when the clip actually carries a colour.
    const sum = rgb[0] + rgb[1] + rgb[2];
    const led = sum > 0.06
      ? new THREE.Color(rgb[0], rgb[1], rgb[2]) : this.warm;
    const lit = Math.min(1, Math.max(0, light));
    for (const m of this.glow) {
      // MuJoCo lerps the geom's rgba, which are display values, so mix in
      // sRGB and convert back -- mixing in linear desaturates the warm tint
      const c = m.userData.base.clone().convertLinearToSRGB().lerp(led, lit);
      m.material.color.copy(c.convertSRGBToLinear());
      m.material.emissive.copy(led);
      m.material.emissiveIntensity = 0.5 * lit;
    }
    if (this.bulb) {
      this.bulb.color.copy(led);
      this.bulb.intensity = 0.35 * lit;
      this.glow[0]?.getWorldPosition(this.bulb.position);
    }
  }

  setViewport(rect, dpr) {
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    this.canvas.style.transform = `translate(${rect.left}px, ${rect.top}px)`;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    if (this._w !== w || this._h !== h || this._dpr !== dpr) {
      this.renderer.setPixelRatio(dpr);
      this.renderer.setSize(w, h, false);
      const aspect = w / h;
      this.camera.aspect = aspect;
      // Frame a fixed world box rather than a fixed field of view: the docks
      // range from 4:3 to square to narrow-on-mobile, and a constant vfov
      // crops the lamp in the tall ones.
      const dist = this.camera.position.distanceTo(this.target);
      const forH = 2 * Math.atan((FRAME_H / 2) / dist);
      const forW = 2 * Math.atan((FRAME_W / 2) / (dist * aspect));
      this.camera.fov = Math.max(forH, forW) * (180 / Math.PI);
      this.camera.updateProjectionMatrix();
      this._w = w; this._h = h; this._dpr = dpr;
    }
  }

  render() {
    this.controls?.update();
    this.renderer.render(this.scene, this.camera);
  }
}

/**
 * The runtime's idle behaviour, ported from runtime/motion/idle.py: a slow
 * sinusoidal breath on the arm and the LED. It is what the lamp does when
 * nothing else is driving it.
 */
export function breathe(t, pose) {
  const IDLE = pose ?? [0, 0.55, -1.05, -0.643, 0.2];
  const ph = (2 * Math.PI * t) / 4.0;          // BREATH_PERIOD_S = 4 s
  const s = Math.sin(ph);
  return {
    q: [IDLE[0], IDLE[1] + 0.028 * s, IDLE[2] - 0.02 * s, IDLE[3],
        IDLE[4] + 0.018 * s],
    light: 0.45 + 0.06 * s,
    rgb: [1.0, 0.85, 0.6],
  };
}
