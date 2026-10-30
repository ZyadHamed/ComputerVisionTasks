import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, Pipe, PipeTransform } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { TitleCasePipe } from '@angular/common';
import { SnakeService } from './snake.service';

// ── A tiny TitleCase pipe (Angular's is in CommonModule) ──────────────────────
// We import TitleCasePipe from @angular/common directly.

type ShapeType = 'rectangle' | 'square' | 'circle';

interface DrawnShape {
  type: ShapeType;
  // For rect / square: top-left in image coords + w / h
  x: number;
  y: number;
  w: number;
  h: number;
  // For circle only
  cx: number;
  cy: number;
  r: number;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule, TitleCasePipe],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnDestroy {
  @ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;

  // ── API (hardcoded, no longer shown in UI) ──────────────────────────────────
  readonly apiUrl = 'http://localhost:8000';

  // ── Snake params ─────────────────────────────────────────────────────────────
  alpha = 1;
  beta  = 1;
  gamma = 2;
  scales = '12,6,3';
  maxIterations = 200;
  numPoints = 200;

  // ── State ────────────────────────────────────────────────────────────────────
  statusError  = '';
  running      = false;
  resultUrl    = '';

  // ── Timer ─────────────────────────────────────────────────────────────────────
  elapsedMs    = 0;   // live counter while running
  elapsedFinal = 0;   // frozen value shown after completion
  private timerHandle: ReturnType<typeof setInterval> | null = null;
  private startTime   = 0;

  // ── Shape controls ────────────────────────────────────────────────────────────
  shapeType: ShapeType = 'rectangle';
  // circle fields
  shapeCx     = 0;
  shapeCy     = 0;
  shapeRadius = 50;
  // rect / square fields
  shapeX = 0;
  shapeY = 0;
  shapeW = 100;
  shapeH = 100;

  // ── Internals ────────────────────────────────────────────────────────────────
  file: File | null = null;
  private img: HTMLImageElement | null = null;
  private shape: DrawnShape | null = null;

  // drag state
  private dragging  = false;
  private dragStart = { x: 0, y: 0 };

  imgW = 0;
  imgH = 0;

  constructor(private snake: SnakeService) {}

  ngOnDestroy() {
    if (this.resultUrl) URL.revokeObjectURL(this.resultUrl);
    this.stopTimer();
  }

  // ── Timer helpers ─────────────────────────────────────────────────────────────
  private startTimer() {
    this.elapsedMs = 0;
    this.startTime = Date.now();
    this.timerHandle = setInterval(() => {
      this.elapsedMs = Date.now() - this.startTime;
    }, 100);
  }

  private stopTimer() {
    if (this.timerHandle !== null) {
      clearInterval(this.timerHandle);
      this.timerHandle = null;
    }
    this.elapsedFinal = Date.now() - this.startTime;
  }

  /** Format milliseconds as e.g. 1.4s or 0.8s */
  formatTime(ms: number): string {
    return (ms / 1000).toFixed(1) + 's';
  }

  // ── Getters ──────────────────────────────────────────────────────────────────
  get hasShape(): boolean {
    return this.shape !== null;
  }

  // ── Shape type ───────────────────────────────────────────────────────────────
  setShape(type: ShapeType) {
    this.shapeType = type;
    this.shape = null;
    this.redraw();
  }

  // ── File upload ──────────────────────────────────────────────────────────────
  onFile(event: Event) {
    const input = event.target as HTMLInputElement;
    const f = input.files?.[0];
    if (!f) return;

    this.file = f;
    const img = new Image();
    img.onload = () => {
      const canvas = this.canvasRef.nativeElement;
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      this.img  = img;
      this.imgW = img.naturalWidth;
      this.imgH = img.naturalHeight;
      this.shape = null;
      this.resultUrl = '';
      this.statusError = '';
      this.redraw();
    };
    img.src = URL.createObjectURL(f);
  }

  // ── Manual shape application (from sidebar fields) ───────────────────────────
  applyManualShape() {
    if (!this.img) return;
    if (this.shapeType === 'circle') {
      this.shape = { type: 'circle', cx: this.shapeCx, cy: this.shapeCy, r: this.shapeRadius, x: 0, y: 0, w: 0, h: 0 };
    } else {
      const side = this.shapeType === 'square' ? this.shapeW : undefined;
      this.shape = {
        type: this.shapeType,
        x: this.shapeX, y: this.shapeY,
        w: side ?? this.shapeW,
        h: side ?? this.shapeH,
        cx: 0, cy: 0, r: 0,
      };
    }
    this.redraw();
  }

  // ── Canvas mouse events (drag-to-draw) ───────────────────────────────────────
  onMouseDown(e: MouseEvent) {
    if (!this.img) return;
    const [ix, iy] = this.toImageCoords(e);
    this.dragging  = true;
    this.dragStart = { x: ix, y: iy };
    this.shape     = null;
  }

  onMouseMove(e: MouseEvent) {
    if (!this.dragging || !this.img) return;
    const [ix, iy] = this.toImageCoords(e);
    this.shape = this.buildShape(this.dragStart.x, this.dragStart.y, ix, iy);
    this.redraw();
  }

  onMouseUp(e: MouseEvent) {
    if (!this.dragging || !this.img) return;
    const [ix, iy] = this.toImageCoords(e);
    this.shape    = this.buildShape(this.dragStart.x, this.dragStart.y, ix, iy);
    this.dragging = false;
    this.syncSidebarFields();
    this.redraw();
  }

  onMouseLeave(e: MouseEvent) {
    if (this.dragging) {
      this.dragging = false;
      this.syncSidebarFields();
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────────────────
  private toImageCoords(e: MouseEvent): [number, number] {
    const canvas = this.canvasRef.nativeElement;
    const rect   = canvas.getBoundingClientRect();
    const ix = Math.round(((e.clientX - rect.left) * canvas.width)  / rect.width);
    const iy = Math.round(((e.clientY - rect.top)  * canvas.height) / rect.height);
    return [ix, iy];
  }

  private buildShape(x0: number, y0: number, x1: number, y1: number): DrawnShape {
    const dx = x1 - x0;
    const dy = y1 - y0;

    if (this.shapeType === 'circle') {
      const r  = Math.round(Math.sqrt(dx * dx + dy * dy));
      return { type: 'circle', cx: x0, cy: y0, r, x: 0, y: 0, w: 0, h: 0 };
    }

    const x = Math.min(x0, x1);
    const y = Math.min(y0, y1);
    let   w = Math.abs(dx);
    let   h = Math.abs(dy);

    if (this.shapeType === 'square') {
      const side = Math.max(w, h);
      w = side;
      h = side;
    }

    return { type: this.shapeType, x, y, w, h, cx: 0, cy: 0, r: 0 };
  }

  /** Sync sidebar numeric inputs from the drawn shape */
  private syncSidebarFields() {
    if (!this.shape) return;
    if (this.shape.type === 'circle') {
      this.shapeCx     = this.shape.cx;
      this.shapeCy     = this.shape.cy;
      this.shapeRadius = this.shape.r;
    } else {
      this.shapeX = this.shape.x;
      this.shapeY = this.shape.y;
      this.shapeW = this.shape.w;
      this.shapeH = this.shape.h;
    }
  }

  // ── Convert shape → contour points (sampled polygon) ─────────────────────────
  private shapeToContour(s: DrawnShape, numPoints = 80): number[][] {
    const pts: number[][] = [];
    if (s.type === 'circle') {
      for (let i = 0; i < numPoints; i++) {
        const a = (2 * Math.PI * i) / numPoints;
        pts.push([Math.round(s.cx + s.r * Math.cos(a)), Math.round(s.cy + s.r * Math.sin(a))]);
      }
    } else {
      // Rectangle / square perimeter
      const { x, y, w, h } = s;
      const perimeter = 2 * (w + h);
      const step = perimeter / numPoints;
      for (let i = 0; i < numPoints; i++) {
        const d = i * step;
        let px: number, py: number;
        if (d < w)                    { px = x + d;     py = y;       }
        else if (d < w + h)           { px = x + w;     py = y + (d - w); }
        else if (d < 2 * w + h)      { px = x + w - (d - w - h); py = y + h; }
        else                          { px = x;          py = y + h - (d - 2 * w - h); }
        pts.push([Math.round(px), Math.round(py)]);
      }
    }
    return pts;
  }

  // ── Redraw canvas ─────────────────────────────────────────────────────────────
  private redraw() {
    if (!this.img) return;
    const canvas = this.canvasRef.nativeElement;
    const ctx    = canvas.getContext('2d')!;
    ctx.drawImage(this.img, 0, 0);

    if (!this.shape) return;
    const s = this.shape;

    ctx.save();
    ctx.strokeStyle = '#4f6ef7';
    ctx.lineWidth   = Math.max(2, canvas.width / 400);
    ctx.setLineDash([8, 4]);
    ctx.shadowColor  = 'rgba(79,110,247,0.6)';
    ctx.shadowBlur   = 10;

    ctx.beginPath();
    if (s.type === 'circle') {
      ctx.arc(s.cx, s.cy, s.r, 0, Math.PI * 2);
    } else {
      ctx.rect(s.x, s.y, s.w, s.h);
    }
    ctx.stroke();

    // Corner / center handles
    ctx.setLineDash([]);
    ctx.shadowBlur = 0;
    ctx.fillStyle  = '#4f6ef7';
    const hr = Math.max(5, canvas.width / 200);

    if (s.type === 'circle') {
      [[s.cx, s.cy], [s.cx + s.r, s.cy]].forEach(([hx, hy]) => {
        ctx.beginPath();
        ctx.arc(hx, hy, hr, 0, Math.PI * 2);
        ctx.fill();
      });
    } else {
      [[s.x, s.y], [s.x + s.w, s.y], [s.x + s.w, s.y + s.h], [s.x, s.y + s.h]].forEach(([hx, hy]) => {
        ctx.beginPath();
        ctx.arc(hx, hy, hr, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    ctx.restore();
  }

  // ── Run ───────────────────────────────────────────────────────────────────────
  run() {
    if (!this.file)  { this.statusError = 'Upload an image first.'; return; }
    if (!this.shape) { this.statusError = 'Draw a contour on the canvas first.'; return; }

    this.running     = true;
    this.statusError = '';
    this.elapsedFinal = 0;
    this.startTimer();

    const contour = this.shapeToContour(this.shape, this.numPoints);

    this.snake
      .runOverlay(this.apiUrl, this.file, contour, {
        alpha: this.alpha,
        beta:  this.beta,
        gamma: this.gamma,
        scales: this.scales,
        maxIterations: this.maxIterations,
        numPoints: this.numPoints,
      })
      .subscribe({
        next: (blob) => {
          this.stopTimer();
          if (this.resultUrl) URL.revokeObjectURL(this.resultUrl);
          this.resultUrl = URL.createObjectURL(blob);
          this.running   = false;
        },
        error: async (err) => {
          let message = 'Request failed — is the backend running?';
          if (err.error instanceof Blob) {
            try {
              const body = JSON.parse(await err.error.text());
              message = JSON.stringify(body.detail ?? body);
            } catch { /* keep default */ }
          }
          this.stopTimer();
          this.statusError = message;
          this.running     = false;
        },
      });
  }

  // ── Reset ─────────────────────────────────────────────────────────────────────
  reset() {
    this.shape       = null;
    this.statusError = '';
    if (this.resultUrl) { URL.revokeObjectURL(this.resultUrl); this.resultUrl = ''; }
    this.redraw();
  }

  // ── Download result ───────────────────────────────────────────────────────────
  download() {
    if (!this.resultUrl) return;
    const a = document.createElement('a');
    a.href     = this.resultUrl;
    a.download = 'snake_result.png';
    a.click();
  }
}
