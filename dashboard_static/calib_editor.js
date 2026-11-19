// Field calibration editor used by the Calibration tab and by the
// per-match approval step. Draws a video frame (or image) with a margin
// around it so corners outside the frame can be placed, and lets the user
// drag the 4 field corners and the HIVE / FLOWER / exclusion boxes.
//
// Calibration format (same as field_corners.json):
//   corners_px      [[BL], [BR], [TR], [TL]]  (floor-level field corners)
//   hive_boxes_px   {red: [[x,y] x4], blue: [[x,y] x4]}
//   flower_boxes_px [[[x,y] x4], ...]
//   exclude_px      [[[x,y] x4], ...]

(function () {
  const CORNER_STYLE = [
    { label: "TL", color: "#e6b422" },
    { label: "TR", color: "#37b5e5" },
    { label: "BR", color: "#d95b43" },
    { label: "BL", color: "#4f80ff" },
  ];
  const BOX_STYLE = {
    hive_red: { stroke: "#ff3b3b", label: "red HIVE" },
    hive_blue: { stroke: "#3b8bff", label: "blue HIVE" },
    flower: { stroke: "#3ccf4e", label: "FLOWER" },
    exclude: { stroke: "#b0b0b0", label: "ignore", fill: "rgba(40,40,40,0.45)" },
  };

  // How far outside the video frame a field corner may be (fraction of the
  // larger frame side); same limit as automatic detection on the server.
  const MAX_OUTSIDE_FRAC = 0.5;

  function rectFromPoly(poly) {
    const xs = poly.map((p) => p[0]);
    const ys = poly.map((p) => p[1]);
    return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
  }

  function polyFromRect(r) {
    const x0 = Math.round(Math.min(r.x0, r.x1));
    const x1 = Math.round(Math.max(r.x0, r.x1));
    const y0 = Math.round(Math.min(r.y0, r.y1));
    const y1 = Math.round(Math.max(r.y0, r.y1));
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  }

  class CalibEditor {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.onChange = options.onChange || (() => {});
      this.marginFrac = options.marginFrac == null ? 0.15 : options.marginFrac;
      this.source = null;
      this.width = 0;
      this.height = 0;
      this.corners = [];        // [{x, y}] in TL, TR, BR, BL order (image px)
      this.boxes = [];          // [{kind, x0, y0, x1, y1}]
      this.mode = "edit";       // edit | hive_red | hive_blue | flower | exclude
      this.drag = null;
      this.selected = -1;
      canvas.addEventListener("mousedown", (e) => this._down(e));
      window.addEventListener("mousemove", (e) => this._move(e));
      window.addEventListener("mouseup", () => this._up());
      window.addEventListener("keydown", (e) => {
        if ((e.key === "Delete" || e.key === "Backspace") && this.selected >= 0 && document.activeElement === document.body) {
          this.boxes.splice(this.selected, 1);
          this.selected = -1;
          this.render();
          this.onChange();
        }
      });
    }

    // Border drawn around the video frame. It grows to show every field
    // corner (corners can be outside the view), up to MAX_OUTSIDE_FRAC.
    get margin() {
      const side = Math.max(this.width, this.height);
      let need = side * this.marginFrac;
      for (const c of this.corners) {
        need = Math.max(need, -c.x, -c.y, c.x - this.width, c.y - this.height);
      }
      // + a small strip so a corner on the limit is still drawn in full
      return Math.round(Math.min(need, side * MAX_OUTSIDE_FRAC) + 0.04 * side);
    }

    // Keep every corner inside the largest border, so it can always be seen
    // and dragged.
    _boundCorners() {
      if (!this.width) return;
      const lim = Math.max(this.width, this.height) * MAX_OUTSIDE_FRAC;
      for (const c of this.corners) {
        c.x = Math.max(-lim, Math.min(this.width + lim, c.x));
        c.y = Math.max(-lim, Math.min(this.height + lim, c.y));
      }
    }

    setSource(source, width, height) {
      const firstImage = !this.width;
      this.source = source;
      this.width = width;
      this.height = height;
      if (firstImage && !this.corners.length) {
        this.corners = this._defaultCorners();
      }
      this._boundCorners();
      this.render();
    }

    setMode(mode) {
      this.mode = mode;
      this.canvas.style.cursor = mode === "edit" ? "default" : "crosshair";
    }

    _defaultCorners() {
      const w = this.width;
      const h = this.height;
      return [
        { x: w * 0.2, y: h * 0.2 }, { x: w * 0.8, y: h * 0.2 },
        { x: w * 0.8, y: h * 0.8 }, { x: w * 0.2, y: h * 0.8 },
      ];
    }

    resetCorners() {
      this.corners = this._defaultCorners();
      this.render();
      this.onChange();
    }

    setCalibration(calib) {
      calib = calib || {};
      const c = calib.corners_px;
      if (Array.isArray(c) && c.length === 4) {
        const [bl, br, tr, tl] = c;
        this.corners = [tl, tr, br, bl].map((p) => ({ x: Number(p[0]), y: Number(p[1]) }));
      } else if (this.width) {
        this.corners = this._defaultCorners();
      }
      this.boxes = [];
      const hives = calib.hive_boxes_px || {};
      for (const color of ["red", "blue"]) {
        if (Array.isArray(hives[color])) {
          this.boxes.push({ kind: `hive_${color}`, ...rectFromPoly(hives[color]) });
        }
      }
      for (const poly of calib.flower_boxes_px || []) {
        this.boxes.push({ kind: "flower", ...rectFromPoly(poly) });
      }
      for (const poly of calib.exclude_px || []) {
        this.boxes.push({ kind: "exclude", ...rectFromPoly(poly) });
      }
      this.selected = -1;
      this._boundCorners();
      this.render();
    }

    getCalibration() {
      const out = {};
      if (this.corners.length === 4) {
        const r = (v) => Math.round(v * 100) / 100;
        const [tl, tr, br, bl] = this.corners;
        out.corners_px = [bl, br, tr, tl].map((p) => [r(p.x), r(p.y)]);
      }
      const hives = {};
      const flowers = [];
      const excludes = [];
      for (const b of this.boxes) {
        const poly = polyFromRect(b);
        if (b.kind === "hive_red") hives.red = poly;
        else if (b.kind === "hive_blue") hives.blue = poly;
        else if (b.kind === "flower") flowers.push(poly);
        else if (b.kind === "exclude") excludes.push(poly);
      }
      if (Object.keys(hives).length) out.hive_boxes_px = hives;
      if (flowers.length) out.flower_boxes_px = flowers;
      if (excludes.length) out.exclude_px = excludes;
      return out;
    }

    // canvas <-> image coordinates
    _toImage(event) {
      const rect = this.canvas.getBoundingClientRect();
      const sx = this.canvas.width / rect.width;
      const sy = this.canvas.height / rect.height;
      return {
        x: (event.clientX - rect.left) * sx - this.margin,
        y: (event.clientY - rect.top) * sy - this.margin,
      };
    }

    _hitRadius() {
      const rect = this.canvas.getBoundingClientRect();
      return 12 * (this.canvas.width / Math.max(rect.width, 1));
    }

    _down(event) {
      if (!this.width) return;
      const p = this._toImage(event);
      const hit = this._hitRadius();
      if (this.mode !== "edit") {
        if (this.mode === "hive_red" || this.mode === "hive_blue") {
          this.boxes = this.boxes.filter((b) => b.kind !== this.mode);
        }
        this.boxes.push({ kind: this.mode, x0: p.x, y0: p.y, x1: p.x, y1: p.y });
        this.selected = this.boxes.length - 1;
        this.drag = { type: "box-corner", index: this.selected, cx: "x1", cy: "y1" };
        return;
      }
      // corners first
      let best = -1;
      let bestD = Infinity;
      this.corners.forEach((c, i) => {
        const d = Math.hypot(c.x - p.x, c.y - p.y);
        if (d < bestD) { bestD = d; best = i; }
      });
      if (best >= 0 && bestD <= hit * 1.3) {
        this.drag = { type: "corner", index: best };
        return;
      }
      // box corners, then box bodies (move)
      for (let i = this.boxes.length - 1; i >= 0; i -= 1) {
        const b = this.boxes[i];
        for (const [cx, cy] of [["x0", "y0"], ["x1", "y0"], ["x1", "y1"], ["x0", "y1"]]) {
          if (Math.hypot(b[cx] - p.x, b[cy] - p.y) <= hit) {
            this.selected = i;
            this.drag = { type: "box-corner", index: i, cx, cy };
            this.render();
            return;
          }
        }
      }
      for (let i = this.boxes.length - 1; i >= 0; i -= 1) {
        const b = this.boxes[i];
        if (p.x >= Math.min(b.x0, b.x1) && p.x <= Math.max(b.x0, b.x1)
          && p.y >= Math.min(b.y0, b.y1) && p.y <= Math.max(b.y0, b.y1)) {
          this.selected = i;
          this.drag = { type: "box-move", index: i, px: p.x, py: p.y };
          this.render();
          return;
        }
      }
      this.selected = -1;
      this.render();
    }

    _move(event) {
      if (!this.drag) return;
      const p = this._toImage(event);
      const edge = Math.max(this.width, this.height) * MAX_OUTSIDE_FRAC;
      const lim = (v, size) => Math.max(-edge, Math.min(size + edge, v));
      if (this.drag.type === "corner") {
        const c = this.corners[this.drag.index];
        c.x = lim(p.x, this.width);
        c.y = lim(p.y, this.height);
      } else if (this.drag.type === "box-corner") {
        const b = this.boxes[this.drag.index];
        b[this.drag.cx] = Math.max(0, Math.min(this.width - 1, p.x));
        b[this.drag.cy] = Math.max(0, Math.min(this.height - 1, p.y));
      } else if (this.drag.type === "box-move") {
        const b = this.boxes[this.drag.index];
        const dx = p.x - this.drag.px;
        const dy = p.y - this.drag.py;
        b.x0 += dx; b.x1 += dx; b.y0 += dy; b.y1 += dy;
        this.drag.px = p.x;
        this.drag.py = p.y;
      }
      this.render();
    }

    _up() {
      if (!this.drag) return;
      const wasBox = this.drag.type !== "corner";
      if (wasBox && this.drag.index != null) {
        const b = this.boxes[this.drag.index];
        if (b && Math.abs(b.x1 - b.x0) < 3 && Math.abs(b.y1 - b.y0) < 3) {
          this.boxes.splice(this.drag.index, 1);   // a click, not a box
          this.selected = -1;
        }
      }
      this.drag = null;
      if (this.mode !== "edit") this.setMode("edit");
      this.render();
      this.onChange();
    }

    deleteSelected() {
      if (this.selected >= 0) {
        this.boxes.splice(this.selected, 1);
        this.selected = -1;
        this.render();
        this.onChange();
      }
    }

    render() {
      if (!this.width) return;
      const m = this.margin;
      const W = this.width + 2 * m;
      const H = this.height + 2 * m;
      if (this.canvas.width !== W || this.canvas.height !== H) {
        this.canvas.width = W;
        this.canvas.height = H;
      }
      const ctx = this.canvas.getContext("2d");
      ctx.fillStyle = "#262626";
      ctx.fillRect(0, 0, W, H);
      if (this.source) {
        try {
          ctx.drawImage(this.source, m, m, this.width, this.height);
        } catch (_err) {
          // source not ready yet
        }
      }
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1;
      ctx.strokeRect(m - 0.5, m - 0.5, this.width + 1, this.height + 1);
      const scale = Math.max(1, this.width / 640);
      // boxes
      this.boxes.forEach((b, i) => {
        const st = BOX_STYLE[b.kind] || BOX_STYLE.exclude;
        const x = Math.min(b.x0, b.x1) + m;
        const y = Math.min(b.y0, b.y1) + m;
        const w = Math.abs(b.x1 - b.x0);
        const h = Math.abs(b.y1 - b.y0);
        if (st.fill) {
          ctx.fillStyle = st.fill;
          ctx.fillRect(x, y, w, h);
        }
        ctx.strokeStyle = st.stroke;
        ctx.lineWidth = (i === this.selected ? 3 : 1.6) * scale;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = st.stroke;
        ctx.font = `${Math.round(11 * scale)}px sans-serif`;
        ctx.fillText(st.label, x + 3 * scale, y + 12 * scale);
        if (i === this.selected) {
          for (const [px, py] of [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) {
            ctx.fillRect(px - 3 * scale, py - 3 * scale, 6 * scale, 6 * scale);
          }
        }
      });
      // field outline
      if (this.corners.length === 4) {
        ctx.save();
        ctx.lineWidth = 2 * scale;
        ctx.strokeStyle = "rgba(60,255,90,0.95)";
        ctx.fillStyle = "rgba(60,255,90,0.10)";
        ctx.beginPath();
        this.corners.forEach((c, i) => {
          if (i === 0) ctx.moveTo(c.x + m, c.y + m);
          else ctx.lineTo(c.x + m, c.y + m);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        this.corners.forEach((c, i) => {
          const st = CORNER_STYLE[i];
          ctx.fillStyle = st.color;
          ctx.beginPath();
          ctx.arc(c.x + m, c.y + m, 7 * scale, 0, Math.PI * 2);
          ctx.fill();
          ctx.lineWidth = 2 * scale;
          ctx.strokeStyle = "#ffffff";
          ctx.stroke();
          ctx.fillStyle = "#ffffff";
          ctx.font = `bold ${Math.round(13 * scale)}px sans-serif`;
          ctx.fillText(st.label, c.x + m + 9 * scale, c.y + m - 9 * scale);
        });
        ctx.restore();
      }
    }
  }

  window.CalibEditor = CalibEditor;
})();
