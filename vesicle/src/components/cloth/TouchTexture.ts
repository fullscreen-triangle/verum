import { Texture } from "three";

interface Point {
  x: number;
  y: number;
  age: number;
  force: number;
}

interface TouchTextureOptions {
  size: number;
  radius: number;
  maxAge: number;
  debugCanvas?: boolean;
}

// util
function outSine(n: number) {
  return Math.sin((n * Math.PI) / 2);
}

export default class TouchTexture {
  options: TouchTextureOptions;
  texture: Texture | null;
  ctx: CanvasRenderingContext2D | null;
  trail: Point[];

  constructor({ size = 128, radius = 0.2, maxAge = 120, debugCanvas = false }) {
    this.options = {
      size,
      radius,
      maxAge,
      debugCanvas,
    };

    // Variables
    this.ctx = null;
    this.texture = null;
    this.trail = [];

    // Init Methods
    this.initCanvas();
  }

  initCanvas() {
    // create a 2D canvas to store the informations of the cursor
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = this.options.size;
    this.ctx = canvas.getContext("2d");

    // draw black background
    if (this.ctx) {
      this.ctx.fillStyle = "black";
      this.ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    // use the canvas as a texture
    this.texture = new Texture(canvas);
    this.texture.needsUpdate = true;

    // We style the canvas very roughly :in most of the cases,
    // it won't appear on the screen, it's just for debug
    canvas.id = "touchTexture";
    canvas.style.position = "fixed";
    canvas.style.bottom = "0";
    canvas.style.zIndex = "10000";

    // Show the debug canvas if needed
    if (this.options.debugCanvas) document.body.appendChild(canvas);
  }

  addPoint(pointPos: { x: number; y: number }) {
    let force = 0;
    const last = this.trail?.length ? this.trail[this.trail?.length - 1] : null;

    if (last) {
      // We calculate the force aka the distance between the new and old point
      // The more distance, the more force, the more the trail will be visible
      const dx = last.x - pointPos.x;
      const dy = last.y - pointPos.y;
      const dd = dx * dx + dy * dy;
      force = Math.min(dd * 10000, 1);
    }

    this.trail.push({ x: pointPos.x, y: pointPos.y, age: 0, force });
  }

  drawPoint(point: Point) {
    const pos = {
      x: point.x * this.options.size,
      y: (1 - point.y) * this.options.size,
    };

    // Initialize intensity to 1
    let intensity = 1;

    if (point.age < this.options.maxAge * 0.3) {
      intensity = outSine(point.age / (this.options.maxAge * 0.3));
    } else {
      intensity = outSine(
        1 -
          (point.age - this.options.maxAge * 0.3) / (this.options.maxAge * 0.7)
      );
    }

    // Multiply intensity by the point's force
    intensity *= point.force;

    // Calculate the radius of the touch based on the canvas size, radius option, and intensity
    const radius = this.options.size * this.options.radius * intensity;

    // Create a radial gradient for the touch
    const grd = this.ctx?.createRadialGradient(
      pos.x,
      pos.y,
      radius * 0.25,
      pos.x,
      pos.y,
      radius
    );

    // Add color stops to the gradient (white at the center, transparent at the edges)
    grd?.addColorStop(0, `rgba(255, 255, 255, 0.35)`);
    grd?.addColorStop(1, "rgba(0, 0, 0, 0.0)");

    // Begin a new path for drawing
    this.ctx?.beginPath();

    // Set the fill style to the gradient if both context and gradient are available
    if (this.ctx && grd) {
      this.ctx.fillStyle = grd;
    }

    // Draw a circle at the calculated position with the calculated radius
    this.ctx?.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

    // Fill the circle with the gradient
    this.ctx?.fill();
  }

  update() {
    // clearing the canvas on every frame
    this.clear();

    // age points
    this.trail.forEach((point, i) => {
      point.age++;
      // remove old
      if (point.age > this.options.maxAge) {
        this.trail.splice(i, 1);
      }
    });

    // draw white points
    this.trail.forEach((point) => {
      this.drawPoint(point);
    });

    if (this.texture) {
      this.texture.needsUpdate = true;
    }
  }

  clear() {
    // clear canvas
    if (this.ctx) {
      this.ctx.fillStyle = "black";
    }
    this.ctx?.fillRect(0, 0, this.options.size, this.options.size);
  }
}
