import {
  AfterViewInit,
  Component,
  ElementRef,
  OnDestroy,
  ViewChild,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import {
  CirclesResponse,
  DetectedCircle,
  DetectedEllipse,
  DetectedLine,
  EllipsesResponse,
  FormOptions,
  ImageProcessingService,
  LinesResponse,
} from './image-processing.service';
import { SnakeFrame, SnakeMeasurements, SnakeService } from './snake.service';

type PortId = 'input' | 'output1' | 'output2';
type OutputId = 'output1' | 'output2';
type Algorithm = 'canny' | 'lines' | 'circles' | 'ellipses' | 'snake';
type ShapeType = 'rectangle' | 'square' | 'circle';

interface DrawnShape {
  type: ShapeType;
  x: number;
  y: number;
  w: number;
  h: number;
  cx: number;
  cy: number;
  r: number;
}

type Analysis =
  | ({ kind: 'lines' } & LinesResponse)
  | ({ kind: 'circles' } & CirclesResponse)
  | ({ kind: 'ellipses' } & EllipsesResponse)
  | {
      kind: 'snake';
      initialContour: number[][] | null;
      currentContour: number[][] | null;
      frame: SnakeFrame | null;
      measurements: SnakeMeasurements | null;
    };

interface WorkspacePort {
  blob: Blob | null;
  url: string | null;
  image: HTMLImageElement | null;
  analysis: Analysis | null;
  algorithm: string;
  source: PortId | null;
  durationMs: number | null;
  showOverlay: boolean;
  highlightedIndex: number | null;
}

function emptyPort(): WorkspacePort {
  return {
    blob: null,
    url: null,
    image: null,
    analysis: null,
    algorithm: '',
    source: null,
    durationMs: null,
    showOverlay: true,
    highlightedIndex: null,
  };
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule, DecimalPipe],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements AfterViewInit, OnDestroy {
  @ViewChild('inputCanvas') inputCanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('output1Canvas') output1CanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('output2Canvas') output2CanvasRef!: ElementRef<HTMLCanvasElement>;

  readonly ports: Record<PortId, WorkspacePort> = {
    input: emptyPort(),
    output1: emptyPort(),
    output2: emptyPort(),
  };
  readonly directions = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘'];
  readonly algorithms: { value: Algorithm; label: string; description: string }[] = [
    { value: 'canny', label: 'Canny edges', description: 'Edge map' },
    { value: 'lines', label: 'Hough lines', description: 'ρ / θ' },
    { value: 'circles', label: 'Hough circles', description: 'Center / radius' },
    { value: 'ellipses', label: 'Hough ellipses', description: 'Axes / rotation' },
    { value: 'snake', label: 'Active contour', description: 'Snake animation' },
  ];

  sourcePort: PortId = 'input';
  destinationPort: OutputId = 'output1';
  selectedOutput: OutputId = 'output1';
  algorithm: Algorithm = 'lines';
  imageName = '';
  running = false;
  loadingLatest = true;
  statusError = '';
  statusMessage = '';
  elapsedMs = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private dragStart = { x: 0, y: 0 };
  private dragging = false;
  private readonly drawnShapes: Record<PortId, DrawnShape | null> = {
    input: null,
    output1: null,
    output2: null,
  };

  // Shared Canny parameters (used as preprocessing for the Hough transforms).
  sigma = 1.4;
  lowThreshold = 20;
  highThreshold = 50;
  cannyKernelSize: number | null = null;
  cannyManual = false;
  cannyPropagate = true;
  applyCanny = true;

  // Hough parameters.
  threshold = 100;
  useVectorized = true;
  minRadius = 10;
  maxRadius = 70;
  radiusStep = 1;
  radii = '';
  maxCircles = 2;
  minCenterDistance = 80;
  radiusDistance = 5;
  batchSize = 1024;
  pairTolerance = 2;
  centerTolerance = 4;
  maxEllipses = 2;
  maxEdgePoints = 20000;

  // Preserve the existing Snake controls and drag-to-draw experience.
  alpha = 1;
  beta = 1;
  gamma = 2;
  scales = '12,6,3';
  maxIterations = 200;
  numPoints = 200;
  shapeType: ShapeType = 'rectangle';
  shapeX = 0;
  shapeY = 0;
  shapeW = 100;
  shapeH = 100;
  shapeCx = 0;
  shapeCy = 0;
  shapeRadius = 50;

  constructor(
    private readonly processing: ImageProcessingService,
    private readonly snake: SnakeService,
  ) {}

  ngAfterViewInit(): void {
    void this.restoreLatest();
  }

  ngOnDestroy(): void {
    this.stopTimer();
    for (const id of ['input', 'output1', 'output2'] as PortId[]) {
      if (this.ports[id].url) URL.revokeObjectURL(this.ports[id].url!);
    }
  }

  get selectedPort(): WorkspacePort {
    return this.ports[this.selectedOutput];
  }

  get selectedLines(): DetectedLine[] {
    const analysis = this.selectedPort.analysis;
    return analysis?.kind === 'lines' ? analysis.lines : [];
  }

  get selectedCircles(): DetectedCircle[] {
    const analysis = this.selectedPort.analysis;
    return analysis?.kind === 'circles' ? analysis.circles : [];
  }

  get selectedEllipses(): DetectedEllipse[] {
    const analysis = this.selectedPort.analysis;
    return analysis?.kind === 'ellipses' ? analysis.ellipses : [];
  }

  get selectedSnake(): Extract<Analysis, { kind: 'snake' }> | null {
    const analysis = this.selectedPort.analysis;
    return analysis?.kind === 'snake' ? analysis : null;
  }

  get chainCodeText(): string {
    const codes = this.selectedSnake?.frame?.chain_code ?? [];
    return codes.map((code) => `${code}${this.directions[code] ?? '?'}`).join(' ');
  }

  get hasDrawnShape(): boolean {
    return this.drawnShapes[this.sourcePort] !== null;
  }

  get sourceAvailable(): boolean {
    return this.ports[this.sourcePort].blob !== null;
  }

  get canRun(): boolean {
    return !this.running && this.sourceAvailable
      && this.sourcePort !== this.destinationPort
      && (this.algorithm !== 'snake' || this.hasDrawnShape);
  }

  get isHough(): boolean {
    return this.algorithm === 'lines' || this.algorithm === 'circles' || this.algorithm === 'ellipses';
  }

  get selectedCount(): number | null {
    const analysis = this.selectedPort.analysis;
    if (!analysis) return null;
    if (analysis.kind === 'snake') return null;
    return analysis.count;
  }

  get portLabels(): Record<PortId, string> {
    return { input: 'Input', output1: 'Output 1', output2: 'Output 2' };
  }

  portDescription(id: PortId): string {
    const port = this.ports[id];
    if (!port.image) return 'No image';
    return `${port.image.naturalWidth} × ${port.image.naturalHeight} px`;
  }

  portSummary(id: OutputId): string {
    const port = this.ports[id];
    if (!port.blob) return 'Ready for an operation';
    const a = port.analysis;
    if (a?.kind === 'snake') return a.measurements
      ? (a.measurements.converged ? 'Snake converged' : 'Iteration limit reached')
      : 'Snake contour';
    if (a) return `${a.count} detected ${a.kind}`;
    return port.algorithm || 'Copied image';
  }

  formatTime(ms: number | null): string {
    return ms === null ? '—' : `${(ms / 1000).toFixed(2)} s`;
  }

  formatNumber(value: number, digits = 1): string {
    return Number(value).toFixed(digits);
  }

  private async restoreLatest(): Promise<void> {
    try {
      const saved = await this.processing.getLatestImage();
      if (saved) {
        await this.setPort('input', saved, null, 'Restored latest image', null, null);
        this.imageName = 'Latest saved image';
        this.statusMessage = 'Restored the last uploaded image from the backend.';
      }
    } catch (error) {
      this.statusError = `Could not restore the saved image: ${await this.errorMessage(error)}`;
    } finally {
      this.loadingLatest = false;
    }
  }

  async onFile(event: Event): Promise<void> {
    const element = event.target as HTMLInputElement;
    const file = element.files?.[0];
    if (!file || this.running) return;
    element.value = ''; // Allow selecting the same filename again.
    this.statusError = '';
    this.statusMessage = '';
    try {
      await this.processing.uploadLatestImage(file);
      await this.setPort('input', file, null, 'Original image', null, null);
      this.drawnShapes.input = null;
      this.imageName = file.name;
      this.sourcePort = 'input';
      this.statusMessage = 'Original saved. Both outputs remain available for comparison.';
    } catch (error) {
      this.statusError = `Upload failed: ${await this.errorMessage(error)}`;
    }
  }

  onSourceChange(): void {
    this.syncShapeFields();
    this.redrawAll();
  }

  setAlgorithm(value: Algorithm): void {
    this.algorithm = value;
    this.redrawAll();
  }

  setShapeType(type: ShapeType): void {
    this.shapeType = type;
    this.drawnShapes[this.sourcePort] = null;
    this.redrawAll();
  }

  clearDrawnShape(): void {
    this.drawnShapes[this.sourcePort] = null;
    this.redrawPort(this.sourcePort);
  }

  applyManualShape(): void {
    if (!this.ports[this.sourcePort].image) return;
    if (this.shapeType === 'circle') {
      this.drawnShapes[this.sourcePort] = {
        type: 'circle', x: 0, y: 0, w: 0, h: 0,
        cx: this.shapeCx, cy: this.shapeCy, r: Math.max(0, this.shapeRadius),
      };
    } else {
      const width = Math.max(0, this.shapeW);
      const height = this.shapeType === 'square' ? width : Math.max(0, this.shapeH);
      this.drawnShapes[this.sourcePort] = {
        type: this.shapeType, x: this.shapeX, y: this.shapeY,
        w: width, h: height, cx: 0, cy: 0, r: 0,
      };
    }
    this.redrawAll();
  }

  private syncShapeFields(): void {
    const shape = this.drawnShapes[this.sourcePort];
    if (!shape) return;
    this.shapeType = shape.type;
    if (shape.type === 'circle') {
      this.shapeCx = shape.cx;
      this.shapeCy = shape.cy;
      this.shapeRadius = shape.r;
    } else {
      this.shapeX = shape.x;
      this.shapeY = shape.y;
      this.shapeW = shape.w;
      this.shapeH = shape.h;
    }
  }

  onCanvasDown(id: PortId, event: MouseEvent): void {
    if (this.running || this.algorithm !== 'snake' || this.sourcePort !== id || !this.ports[id].image) return;
    this.dragStart = this.imageCoords(id, event);
    this.drawnShapes[id] = null;
    this.dragging = true;
  }

  onCanvasMove(id: PortId, event: MouseEvent): void {
    if (!this.dragging || this.sourcePort !== id) return;
    const cursor = this.imageCoords(id, event);
    this.drawnShapes[id] = this.shapeFromDrag(this.dragStart, cursor);
    this.redrawPort(id);
  }

  onCanvasUp(id: PortId, event: MouseEvent): void {
    if (!this.dragging || this.sourcePort !== id) return;
    const cursor = this.imageCoords(id, event);
    this.drawnShapes[id] = this.shapeFromDrag(this.dragStart, cursor);
    this.dragging = false;
    this.syncShapeFields();
    this.redrawPort(id);
  }

  onCanvasLeave(id: PortId): void {
    if (!this.dragging || this.sourcePort !== id) return;
    this.dragging = false;
    this.syncShapeFields();
  }

  private imageCoords(id: PortId, event: MouseEvent): { x: number; y: number } {
    const canvas = this.canvasFor(id)!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(canvas.width - 1, Math.round((event.clientX - rect.left) * canvas.width / rect.width))),
      y: Math.max(0, Math.min(canvas.height - 1, Math.round((event.clientY - rect.top) * canvas.height / rect.height))),
    };
  }

  private shapeFromDrag(p0: { x: number; y: number }, p1: { x: number; y: number }): DrawnShape {
    const dx = p1.x - p0.x;
    const dy = p1.y - p0.y;
    if (this.shapeType === 'circle') {
      return {
        type: 'circle', cx: p0.x, cy: p0.y, r: Math.round(Math.hypot(dx, dy)),
        x: 0, y: 0, w: 0, h: 0,
      };
    }
    const width = this.shapeType === 'square' ? Math.max(Math.abs(dx), Math.abs(dy)) : Math.abs(dx);
    const height = this.shapeType === 'square' ? width : Math.abs(dy);
    return {
      type: this.shapeType, x: Math.min(p0.x, p1.x), y: Math.min(p0.y, p1.y),
      w: width, h: height, cx: 0, cy: 0, r: 0,
    };
  }

  private shapeToContour(shape: DrawnShape): number[][] {
    const n = Math.max(10, Math.round(this.numPoints));
    const points: number[][] = [];
    if (shape.type === 'circle') {
      if (shape.r <= 0) throw new Error('Draw a circle with a radius greater than zero.');
      for (let i = 0; i < n; i++) {
        const angle = i * 2 * Math.PI / n;
        points.push([
          Math.round(shape.cx + shape.r * Math.cos(angle)),
          Math.round(shape.cy + shape.r * Math.sin(angle)),
        ]);
      }
      return points;
    }
    const { x, y, w, h } = shape;
    if (w <= 0 || h <= 0) throw new Error('Draw a contour with nonzero width and height.');
    const perimeter = 2 * (w + h);
    for (let i = 0; i < n; i++) {
      const distance = i * perimeter / n;
      let px: number;
      let py: number;
      if (distance < w) {
        px = x + distance; py = y;
      } else if (distance < w + h) {
        px = x + w; py = y + distance - w;
      } else if (distance < 2 * w + h) {
        px = x + w - (distance - w - h); py = y + h;
      } else {
        px = x; py = y + h - (distance - 2 * w - h);
      }
      points.push([Math.round(px), Math.round(py)]);
    }
    return points;
  }

  private commonHoughParams(): FormOptions {
    return {
      threshold: this.threshold,
      use_vectorized: this.useVectorized,
      apply_canny: this.applyCanny,
      sigma: this.sigma,
      low_threshold: this.lowThreshold,
      high_threshold: this.highThreshold,
    };
  }

  async run(): Promise<void> {
    if (!this.canRun) {
      this.statusError = !this.sourceAvailable
        ? 'Choose a source port containing an image.'
        : this.sourcePort === this.destinationPort
          ? 'Source and destination must be different ports.'
          : 'Draw a contour on the selected source before running Snake.';
      return;
    }

    const source = this.sourcePort;
    const destination = this.destinationPort;
    const algorithm = this.algorithm;
    const blob = this.ports[source].blob!;
    // Keep the live Snake iteration inspector focused on the destination.
    this.selectedOutput = destination;
    this.running = true;
    this.statusError = '';
    this.statusMessage = '';
    this.elapsedMs = 0;
    const started = performance.now();
    this.startTimer(started);

    try {
      if (algorithm === 'canny') {
        const result = await this.processing.runCanny(blob, {
          sigma: this.sigma,
          low_threshold: this.lowThreshold,
          high_threshold: this.highThreshold,
          kernel_size: this.cannyKernelSize,
          manual: this.cannyManual,
          propagate: this.cannyPropagate,
        });
        await this.setPort(destination, result, null, 'Canny edges', source, null);
      } else if (algorithm === 'lines') {
        const result = await this.processing.detectLines(blob, this.commonHoughParams());
        await this.setPort(destination, blob, { kind: 'lines', ...result }, 'Hough lines', source, null);
      } else if (algorithm === 'circles') {
        const result = await this.processing.detectCircles(blob, {
          ...this.commonHoughParams(),
          min_radius: this.minRadius,
          max_radius: this.maxRadius,
          radius_step: this.radiusStep,
          radii: this.radii.trim() || null,
          max_circles: this.maxCircles,
          min_center_distance: this.minCenterDistance,
          radius_distance: this.radiusDistance,
          batch_size: this.batchSize,
        });
        await this.setPort(destination, blob, { kind: 'circles', ...result }, 'Hough circles', source, null);
      } else if (algorithm === 'ellipses') {
        const result = await this.processing.detectEllipses(blob, {
          ...this.commonHoughParams(),
          pair_tolerance: this.pairTolerance,
          center_tolerance: this.centerTolerance,
          max_ellipses: this.maxEllipses,
          max_edge_points: this.maxEdgePoints,
        });
        await this.setPort(destination, blob, { kind: 'ellipses', ...result }, 'Hough ellipses', source, null);
      } else {
        await this.runSnake(source, destination, blob);
      }

      const elapsed = performance.now() - started;
      this.ports[destination].durationMs = elapsed;
      this.selectedOutput = destination;
      this.statusMessage = `${this.ports[destination].algorithm} completed in ${this.formatTime(elapsed)}.`;
    } catch (error) {
      this.statusError = await this.errorMessage(error);
    } finally {
      this.elapsedMs = Math.round(performance.now() - started);
      this.running = false;
      this.stopTimer();
    }
  }

  private async runSnake(source: PortId, destination: OutputId, blob: Blob): Promise<void> {
    const shape = this.drawnShapes[source];
    if (!shape) throw new Error('Draw a contour on the selected source.');
    const contour = this.shapeToContour(shape);
    const analysis: Extract<Analysis, { kind: 'snake' }> = {
      kind: 'snake', initialContour: null, currentContour: null,
      frame: null, measurements: null,
    };
    await this.setPort(destination, blob, analysis, 'Active contour', source, null);
    const measurements = await this.snake.runIterations(
      this.processing.apiUrl,
      blob,
      contour,
      {
        alpha: this.alpha, beta: this.beta, gamma: this.gamma,
        scales: this.scales, maxIterations: this.maxIterations, numPoints: this.numPoints,
      },
      async (frame: SnakeFrame) => {
        analysis.frame = frame;
        analysis.currentContour = frame.contour;
        if (frame.iteration === 0) analysis.initialContour = frame.contour;
        this.redrawPort(destination);
        // Retain the animated iteration preview from the original interface.
        await new Promise<void>((resolve) => setTimeout(resolve, 35));
      },
    );
    analysis.measurements = measurements;
    this.redrawPort(destination);
  }

  async copyToDestination(): Promise<void> {
    if (this.running || !this.sourceAvailable || this.sourcePort === this.destinationPort) {
      this.statusError = 'Select a filled source and a different destination.';
      return;
    }
    try {
      const source = this.ports[this.sourcePort];
      const analysis = source.analysis
        ? JSON.parse(JSON.stringify(source.analysis)) as Analysis : null;
      await this.setPort(
        this.destinationPort, source.blob!, analysis,
        source.algorithm || 'Copied image', this.sourcePort, source.durationMs,
      );
      this.selectedOutput = this.destinationPort;
      this.statusError = '';
      this.statusMessage = 'Copied image and annotations. Raw pixels remain unchanged.';
    } catch (error) {
      this.statusError = await this.errorMessage(error);
    }
  }

  swapOutputs(): void {
    if (this.running) return;
    const one = this.ports.output1;
    this.ports.output1 = this.ports.output2;
    this.ports.output2 = one;
    const firstShape = this.drawnShapes.output1;
    this.drawnShapes.output1 = this.drawnShapes.output2;
    this.drawnShapes.output2 = firstShape;
    this.syncShapeFields();
    // Each canvas stays in its named position; only data and metadata swap.
    this.redrawAll();
    this.statusError = '';
    this.statusMessage = 'Output 1 and Output 2 swapped.';
  }

  clearOutput(id: OutputId): void {
    if (this.running) return;
    if (this.ports[id].url) URL.revokeObjectURL(this.ports[id].url!);
    this.ports[id] = emptyPort();
    this.drawnShapes[id] = null;
    this.redrawPort(id);
    this.statusMessage = `${this.portLabels[id]} cleared.`;
    this.statusError = '';
  }

  selectShape(id: OutputId, index: number): void {
    const port = this.ports[id];
    this.selectedOutput = id;
    port.highlightedIndex = port.highlightedIndex === index ? null : index;
    port.showOverlay = true;
    this.redrawPort(id);
  }

  toggleOverlay(id: OutputId): void {
    this.ports[id].showOverlay = !this.ports[id].showOverlay;
    this.redrawPort(id);
  }

  async downloadPort(id: PortId, annotated = false): Promise<void> {
    const port = this.ports[id];
    if (!port.blob) return;
    try {
      if (!annotated) {
        const ext = port.blob.type === 'image/jpeg' ? 'jpg'
          : port.blob.type === 'image/webp' ? 'webp' : 'png';
        this.downloadBlob(port.blob, `${id}-raw.${ext}`);
        return;
      }
      const canvas = document.createElement('canvas');
      this.paintPort(id, canvas, true);
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((result) => result ? resolve(result) : reject(new Error('Could not encode PNG.')), 'image/png');
      });
      this.downloadBlob(blob, `${id}-overlay.png`);
    } catch (error) {
      this.statusError = await this.errorMessage(error);
    }
  }

  exportJson(id: OutputId): void {
    const port = this.ports[id];
    if (!port.analysis) return;
    const payload = {
      algorithm: port.algorithm,
      source: port.source,
      duration_ms: port.durationMs,
      image_width: port.image?.naturalWidth,
      image_height: port.image?.naturalHeight,
      results: port.analysis,
    };
    this.downloadBlob(new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    }), `${id}-detections.json`);
  }

  private downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  private async setPort(
    id: PortId,
    blob: Blob,
    analysis: Analysis | null,
    algorithm: string,
    source: PortId | null,
    durationMs: number | null,
  ): Promise<void> {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    try {
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error('Could not decode the returned image.'));
        image.src = url;
      });
    } catch (error) {
      URL.revokeObjectURL(url);
      throw error;
    }

    const previousUrl = this.ports[id].url;
    this.ports[id] = {
      blob, url, image, analysis, algorithm, source, durationMs,
      showOverlay: true, highlightedIndex: null,
    };
    this.drawnShapes[id] = null;
    if (previousUrl) URL.revokeObjectURL(previousUrl);
    this.redrawPort(id);
  }

  private canvasFor(id: PortId): HTMLCanvasElement | null {
    if (id === 'input') return this.inputCanvasRef?.nativeElement ?? null;
    if (id === 'output1') return this.output1CanvasRef?.nativeElement ?? null;
    return this.output2CanvasRef?.nativeElement ?? null;
  }

  private redrawAll(): void {
    this.redrawPort('input');
    this.redrawPort('output1');
    this.redrawPort('output2');
  }

  redrawPort(id: PortId): void {
    const canvas = this.canvasFor(id);
    if (canvas) this.paintPort(id, canvas, false);
  }

  private paintPort(id: PortId, canvas: HTMLCanvasElement, forceOverlay: boolean): void {
    const port = this.ports[id];
    const image = port.image;
    if (!image) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(image, 0, 0);

    if ((port.showOverlay || forceOverlay) && port.analysis) {
      this.drawAnalysis(ctx, port.analysis, port.highlightedIndex);
    }
    // A blue, dashed initial contour is shown only for Snake on its chosen source.
    if (this.algorithm === 'snake' && id === this.sourcePort && this.drawnShapes[id]) {
      this.drawManualShape(ctx, this.drawnShapes[id]!);
    }
  }

  private drawAnalysis(ctx: CanvasRenderingContext2D, analysis: Analysis, highlighted: number | null): void {
    const diagonal = Math.hypot(ctx.canvas.width, ctx.canvas.height);
    if (analysis.kind === 'snake') {
      if (analysis.initialContour) this.drawContour(ctx, analysis.initialContour, '#507cfb', 1.5);
      if (analysis.currentContour) this.drawContour(ctx, analysis.currentContour, '#ff4969', 2.25, true);
      return;
    }
    const setStroke = (index: number) => {
      ctx.strokeStyle = index === highlighted ? '#12b8cb' : '#fa4257';
      ctx.lineWidth = index === highlighted ? 3.5 : 2;
    };
    ctx.save();
    if (analysis.kind === 'lines') {
      analysis.lines.forEach((line, index) => {
        setStroke(index);
        const cos = Math.cos(line.theta);
        const sin = Math.sin(line.theta);
        const x0 = cos * line.rho;
        const y0 = sin * line.rho;
        ctx.beginPath();
        ctx.moveTo(x0 - diagonal * sin, y0 + diagonal * cos);
        ctx.lineTo(x0 + diagonal * sin, y0 - diagonal * cos);
        ctx.stroke();
      });
    } else if (analysis.kind === 'circles') {
      analysis.circles.forEach((circle, index) => {
        setStroke(index);
        ctx.beginPath();
        ctx.arc(circle.center_x, circle.center_y, Math.max(0, circle.radius), 0, Math.PI * 2);
        ctx.stroke();
      });
    } else if (analysis.kind === 'ellipses') {
      analysis.ellipses.forEach((ellipse, index) => {
        setStroke(index);
        ctx.beginPath();
        ctx.ellipse(ellipse.center_x, ellipse.center_y, Math.max(0.001, ellipse.a),
          Math.max(0.001, ellipse.b), ellipse.alpha, 0, Math.PI * 2);
        ctx.stroke();
      });
    }
    ctx.restore();
  }

  private drawContour(
    ctx: CanvasRenderingContext2D, points: number[][], color: string,
    width: number, showPoints = false,
  ): void {
    if (points.length < 2) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (const [x, y] of points.slice(1)) ctx.lineTo(x, y);
    ctx.closePath();
    ctx.stroke();
    if (showPoints) {
      ctx.fillStyle = '#f8c94c';
      for (let i = 0; i < points.length; i += Math.max(1, Math.round(points.length / 100))) {
        ctx.beginPath();
        ctx.arc(points[i][0], points[i][1], Math.max(1.5, ctx.canvas.width / 1000), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  private drawManualShape(ctx: CanvasRenderingContext2D, shape: DrawnShape): void {
    ctx.save();
    ctx.setLineDash([8, 4]);
    ctx.lineWidth = Math.max(2, ctx.canvas.width / 400);
    ctx.strokeStyle = '#506af3';
    ctx.beginPath();
    if (shape.type === 'circle') ctx.arc(shape.cx, shape.cy, Math.max(0, shape.r), 0, 2 * Math.PI);
    else ctx.rect(shape.x, shape.y, shape.w, shape.h);
    ctx.stroke();
    ctx.restore();
  }

  private startTimer(start: number): void {
    this.stopTimer();
    this.timer = setInterval(() => this.elapsedMs = Math.round(performance.now() - start), 100);
  }

  private stopTimer(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  private async errorMessage(error: unknown): Promise<string> {
    if (error instanceof HttpErrorResponse) {
      let data = error.error;
      if (data instanceof Blob) {
        try { data = JSON.parse(await data.text()); } catch { data = null; }
      }
      if (typeof data?.detail === 'string') return data.detail;
      if (Array.isArray(data?.detail)) return JSON.stringify(data.detail);
      return error.status === 0
        ? 'Cannot reach the backend. Make sure FastAPI is running at http://localhost:8000.'
        : `Backend request failed (HTTP ${error.status}).`;
    }
    return error instanceof Error ? error.message : String(error);
  }
}
