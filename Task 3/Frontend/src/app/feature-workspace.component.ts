import { AfterViewInit, Component, ElementRef, OnDestroy, ViewChild } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import {
  CornerDetector, DescriptorMethod, FeatureLabService, FeatureMode, FeatureSet,
  MatchMetric, MatchOptions, MatchPair, MatchResponse, CornerOptions, SiftOptions,
} from './feature-lab.service';

type Side = 'a' | 'b';
type WorkspaceView = 'keypoints' | 'matching';
type PreviewMode = 'points' | 'server' | 'raw';
type CompareMode = 'interactive' | 'server';

interface FeatureViewport {
  blob: Blob | null;
  url: string | null;
  image: HTMLImageElement | null;
  previewError: string | null;
  name: string;
  features: FeatureSet | null;
  preview: PreviewMode;
  overlayBlob: Blob | null;
  overlayUrl: string | null;
  overlayImage: HTMLImageElement | null;
}

const emptyViewport = (): FeatureViewport => ({
  blob: null, url: null, image: null, previewError: null, name: '', features: null,
  preview: 'points', overlayBlob: null, overlayUrl: null, overlayImage: null,
});

@Component({
  selector: 'app-feature-workspace',
  standalone: true,
  imports: [FormsModule, DecimalPipe],
  templateUrl: './feature-workspace.component.html',
  styleUrl: './feature-workspace.component.css',
})
export class FeatureWorkspaceComponent implements AfterViewInit, OnDestroy {
  @ViewChild('keypointCanvas') private keypointCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('imageCanvasA') private imageCanvasA!: ElementRef<HTMLCanvasElement>;
  @ViewChild('imageCanvasB') private imageCanvasB!: ElementRef<HTMLCanvasElement>;
  @ViewChild('comparisonCanvas') private comparisonCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('descriptorCanvasA') private descriptorCanvasA!: ElementRef<HTMLCanvasElement>;
  @ViewChild('descriptorCanvasB') private descriptorCanvasB!: ElementRef<HTMLCanvasElement>;

  readonly slots: Record<Side, FeatureViewport> = { a: emptyViewport(), b: emptyViewport() };
  activeView: WorkspaceView = 'keypoints';
  draggingSide: Side | null = null;
  reuseExtractedFeatures = true;
  scaleCircleWithSigma = true;
  readonly featureMethods: { value: FeatureMode; label: string; description: string }[] = [
    { value: 'corners', label: 'Corner detector', description: 'Harris / λ-minus + SIFT / MOPS' },
    { value: 'sift', label: 'Full SIFT', description: 'DoG detection + SIFT descriptor' },
  ];

  featureMode: FeatureMode = 'corners';
  cornerMethod: CornerDetector = 'harris';
  descriptorMethod: DescriptorMethod = 'mops';
  cornerThreshold = 0.01;
  windowSize = 5;
  cornerMargin: number | null = null;
  cornerOctaves = 3;


  siftContrast = 0.03;
  siftEdge = 10;
  siftOctaves = 3;
  siftIntervals = 3;
  siftSigma = 1.6;
  siftAssumedBlur = 0.5;
  siftPeakRatio = 0.8;
  siftDescriptorThreshold = 0.2;

  metric: MatchMetric = 'ssd';
  ratio = 0.75;
  crossCheck = false;
  matchThreshold: number | null = null;
  matchResult: MatchResponse | null = null;
  selectedMatchIndex: number | null = null;
  drawLines = true;
  drawCircles = true;
  showOnlySelected = false;
  drawOrientations = false;
  circleRadius = 5;
  comparisonMode: CompareMode = 'interactive';
  nativeZoom = false;
  serverMatchImage: HTMLImageElement | null = null;
  serverMatchBlob: Blob | null = null;
  private serverMatchUrl: string | null = null;
  // Only the *browser canvas preview* is scaled for memory efficiency.
  // Original files and all descriptors are sent to the backend unchanged.
  private readonly previewPixelBudget = 12_000_000;
  private readonly previewMaxDimension = 8192;
  private comparisonScale = 1;
  private comparisonRenderVersion = 0;
  matchPage = 0;
  readonly matchPageSize = 100;
  private entryCacheResult: MatchResponse | null = null;
  private entryCache: { match: MatchPair; index: number }[] = [];

  isBusy = false;
  busyLabel = '';
  restoring = true;
  status = '';
  error = '';
  private destroyed = false;

  constructor(private readonly lab: FeatureLabService) {}

  ngAfterViewInit(): void {
    this.renderAll();
    void this.restoreLatest();
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    this.disposeViewport('a');
    this.disposeViewport('b');
    this.clearServerMatch();
  }

  setView(view: WorkspaceView): void {
    this.activeView = view;
    // Keep both canvases mounted while switching between keypoints and matching.
    requestAnimationFrame(() => this.renderAll());
  }

  get canExtract(): boolean {
    return !this.isBusy && !!this.slots.a.blob && !!this.slots.b.blob;
  }

  get canMatch(): boolean {
    return !this.isBusy && !!this.slots.a.features && !!this.slots.b.features &&
      !!this.slots.a.blob && !!this.slots.b.blob;
  }

  get effectiveDescriptor(): string {
    return this.featureMode === 'sift' ? 'SIFT · 128D'
      : this.descriptorMethod === 'sift' ? 'SIFT · 128D' : 'MOPS · 64D';
  }

  get detectorLabel(): string {
    return this.featureMode === 'sift' ? 'SIFT / DoG'
      : this.cornerMethod === 'harris' ? 'Harris' : 'Lambda Minus';
  }

  get matches(): MatchPair[] { return this.matchResult?.matches ?? []; }


  colorForMatch(index: number): string { return this.matchColor(index); }

  renderComparisonView(): void { this.renderComparison(); }

  invalidateOverlaysAndRedraw(): void {
    this.clearOverlay('a');
    this.clearOverlay('b');
    this.comparisonMode = 'interactive';
    this.renderAll();
  }

  get selectedMatch(): MatchPair | null {
    return this.selectedMatchIndex === null ? null : this.matches[this.selectedMatchIndex] ?? null;
  }

  get visibleMatchEntries(): { match: MatchPair; index: number }[] {
    if (this.entryCacheResult !== this.matchResult) {
      this.entryCacheResult = this.matchResult;
      this.entryCache = this.matches.map((match, index) => ({ match, index }));
    }
    if (this.showOnlySelected && this.selectedMatchIndex !== null) {
      const chosen = this.entryCache[this.selectedMatchIndex];
      return chosen ? [chosen] : [];
    }
    return this.entryCache;
  }

  get matchPageCount(): number {
    return Math.max(1, Math.ceil(this.matches.length / this.matchPageSize));
  }

  get pagedMatches(): { match: MatchPair; index: number }[] {
    const start = this.matchPage * this.matchPageSize;
    return this.matches.slice(start, start + this.matchPageSize)
      .map((match, index) => ({ match, index: start + index }));
  }

  changeMatchPage(delta: number): void {
    this.matchPage = Math.min(this.matchPageCount - 1, Math.max(0, this.matchPage + delta));
  }

  get matchDescriptorA(): number[] {
    const selected = this.selectedMatch;
    return selected ? this.slots.a.features?.descriptors[selected.index1] ?? [] : [];
  }

  get matchDescriptorB(): number[] {
    const selected = this.selectedMatch;
    return selected ? this.slots.b.features?.descriptors[selected.index2] ?? [] : [];
  }

  private async restoreLatest(): Promise<void> {
    try {
      const latest = await this.lab.getLatestImage();
      if (latest && !this.slots.a.blob && !this.destroyed) {
        await this.setImage('a', latest, 'Latest saved image');
        this.status = 'Restored the saved original into Image A. Add Image B to start matching.';
      }
    } catch (error) {
      // A fresh backend may not yet contain an uploaded image. This is not a failure.
      if (!(error instanceof HttpErrorResponse && error.status === 400)) {
        this.error = `Could not restore the latest image: ${await this.errorMessage(error)}`;
      }
    } finally {
      this.restoring = false;
    }
  }

  async useLatest(): Promise<void> {
    if (this.isBusy) return;
    this.error = '';
    try {
      const latest = await this.lab.getLatestImage();
      if (!latest) throw new Error('There is no image saved in the backend yet.');
      await this.setImage('a', latest, 'Latest saved image');
      this.status = 'Loaded the latest backend image into Image A.';
    } catch (error) {
      this.error = await this.errorMessage(error);
    }
  }

  async saveAsLatest(): Promise<void> {
    if (!this.slots.a.blob || this.isBusy) return;
    this.error = '';
    try {
      const blob = this.slots.a.blob;
      const file = new File([blob], this.slots.a.name || 'feature-image.png',
        { type: blob.type || 'image/png' });
      await this.lab.uploadLatestImage(file);
      this.status = 'Saved Image A as the latest backend image.';
    } catch (error) {
      this.error = await this.errorMessage(error);
    }
  }

  async onFile(side: Side, event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (file) await this.loadImageFile(side, file);
  }

  onDragOver(side: Side, event: DragEvent): void {
    event.preventDefault();
    if (!this.isBusy) this.draggingSide = side;
  }

  onDragLeave(side: Side, event: DragEvent): void {
    event.preventDefault();
    if (this.draggingSide === side) this.draggingSide = null;
  }

  async onDrop(side: Side, event: DragEvent): Promise<void> {
    event.preventDefault();
    this.draggingSide = null;
    const file = event.dataTransfer?.files?.[0];
    if (file) await this.loadImageFile(side, file);
  }

  private async loadImageFile(side: Side, file: File): Promise<void> {
    if (this.isBusy) return;
    if (!file.type.startsWith('image/')) {
      this.error = 'Please choose an image file (PNG, JPG, or another supported image format).';
      return;
    }
    this.error = '';
    this.status = '';
    try {
      await this.setImage(side, file, file.name);
      this.status = `Loaded ${file.name} into Image ${side.toUpperCase()}.`;
    } catch (error) {
      this.error = await this.errorMessage(error);
    }
  }

  private async loadBrowserImage(blob: Blob): Promise<{ image: HTMLImageElement; url: string }> {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    try {
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error('The selected image cannot be decoded.'));
        image.src = url;
      });
      return { image, url };
    } catch (error) {
      URL.revokeObjectURL(url);
      throw error;
    }
  }

  private async setImage(side: Side, blob: Blob, name: string): Promise<void> {
    let image: HTMLImageElement | null = null;
    let url: string | null = null;
    let previewError: string | null = null;
    try {
      ({ image, url } = await this.loadBrowserImage(blob));
    } catch {
      // A browser preview failure (e.g. very large dimensions) must not
      // prevent sending the full original file to the processing backend.
      previewError = 'The browser could not display a preview. The original file can still be processed.';
    }
    if (this.destroyed) {
      if (url) URL.revokeObjectURL(url);
      return;
    }
    this.disposeViewport(side);
    this.slots[side] = { ...emptyViewport(), blob, name, image, url, previewError };
    this.clearMatch();
    this.renderAll();
  }

  private disposeViewport(side: Side): void {
    const slot = this.slots[side];
    if (slot.url) URL.revokeObjectURL(slot.url);
    if (slot.overlayUrl) URL.revokeObjectURL(slot.overlayUrl);
  }

  private clearOverlay(side: Side): void {
    const slot = this.slots[side];
    if (slot.overlayUrl) URL.revokeObjectURL(slot.overlayUrl);
    slot.overlayUrl = null;
    slot.overlayImage = null;
    slot.overlayBlob = null;
    if (slot.preview === 'server') slot.preview = 'points';
  }

  clearSide(side: Side): void {
    if (this.isBusy) return;
    this.disposeViewport(side);
    this.slots[side] = emptyViewport();
    this.clearMatch();
    this.renderAll();
  }

  swapImages(): void {
    if (this.isBusy) return;
    const first = this.slots.a;
    this.slots.a = this.slots.b;
    this.slots.b = first;
    this.clearMatch();
    this.renderAll();
    this.status = 'Swapped the image viewports. Run matching again to update correspondence direction.';
  }

  onFeaturesChanged(): void {
    for (const side of ['a', 'b'] as Side[]) {
      this.slots[side].features = null;
      this.clearOverlay(side);
    }
    this.clearMatch();
    this.renderAll();
  }

  onMatchOptionsChanged(): void {
    this.clearMatch();
    this.renderAll();
  }

  onDrawingChanged(): void {
    // Backend visualizations are baked PNGs. Switch back to editable lines/points.
    this.comparisonMode = 'interactive';
    this.renderAll();
  }

private cornerOptions(): CornerOptions {
  if (this.windowSize % 2 !== 1)
    throw new Error('NMS window size must be an odd number.');

  return {
    method: this.cornerMethod,
    descriptor_method: this.descriptorMethod,
    threshold: this.cornerThreshold,
    window_size: this.windowSize,
    margin: this.cornerMargin,
    num_octaves: this.cornerOctaves,
  };
}

  private siftOptions(): SiftOptions {
    return {
      max_keypoints: null,
      contrast_threshold: this.siftContrast,
      edge_threshold: this.siftEdge,
      num_octaves: this.siftOctaves,
      num_intervals: this.siftIntervals,
      sigma: this.siftSigma,
      assumed_blur: this.siftAssumedBlur,
      num_cells: 4,
      num_bins: 8,
      peak_ratio: this.siftPeakRatio,
      descriptor_threshold: this.siftDescriptorThreshold,
    };
  }

  private matchOptions(): MatchOptions {
    return {
      // Existing matcher uses method to validate descriptor dimensions, not detector type.
      method: this.featureMode === 'sift' || this.descriptorMethod === 'sift' ? 'sift' : 'harris',
      metric: this.metric,
      ratio: this.ratio,
      threshold: this.matchThreshold,
      cross_check: this.crossCheck,
      patch_size: 11,
      max_matches: null,
    };
  }

  private extract(image: Blob): Promise<FeatureSet> {
    return this.featureMode === 'sift'
      ? this.lab.extractSift(image, this.siftOptions())
      : this.lab.extractCorners(image, this.cornerOptions());
  }

  async extractOne(side: Side): Promise<void> {
    const slot = this.slots[side];
    if (this.isBusy || !slot.blob) return;
    this.isBusy = true;
    this.busyLabel = `Extracting ${side.toUpperCase()}`;
    this.error = '';
    this.status = '';
    try {
      slot.features = await this.extract(slot.blob);
      this.clearOverlay(side);
      slot.preview = 'points';
      this.clearMatch();
      this.renderAll();
      this.status = `Image ${side.toUpperCase()}: ${slot.features.num_keypoints} descriptors extracted.`;
    } catch (error) {
      this.error = await this.errorMessage(error);
    } finally {
      this.isBusy = false;
      this.busyLabel = '';
    }
  }

  async extractAndMatch(): Promise<void> {
    if (!this.canExtract) {
      this.error = 'Upload both Image A and Image B first.';
      return;
    }
    this.isBusy = true;
    this.busyLabel = 'Extracting features';
    this.error = '';
    this.status = '';
    try {
      const a = this.slots.a;
      const b = this.slots.b;
      // Each request uploads its own pixels; no dependence on the global latest-image slot.
      const [fa, fb] = await Promise.all([
        this.reuseExtractedFeatures && a.features ? Promise.resolve(a.features) : this.extract(a.blob!),
        this.reuseExtractedFeatures && b.features ? Promise.resolve(b.features) : this.extract(b.blob!),
      ]);
      a.features = fa;
      b.features = fb;
      this.clearOverlay('a');
      this.clearOverlay('b');
      a.preview = 'points';
      b.preview = 'points';
      this.clearMatch();
      this.renderAll();
      this.busyLabel = 'Matching descriptors';
      await this.performMatch();
    } catch (error) {
      this.error = await this.errorMessage(error);
    } finally {
      this.isBusy = false;
      this.busyLabel = '';
    }
  }

  async matchOnly(): Promise<void> {
    if (!this.canMatch) {
      this.error = 'Extract keypoints for both images before matching.';
      return;
    }
    this.isBusy = true;
    this.busyLabel = 'Matching descriptors';
    this.error = '';
    this.status = '';
    try {
      await this.performMatch();
    } catch (error) {
      this.error = await this.errorMessage(error);
    } finally {
      this.isBusy = false;
      this.busyLabel = '';
    }
  }

  private async performMatch(): Promise<void> {
    const a = this.slots.a;
    const b = this.slots.b;
    this.clearMatch();
    const result = await this.lab.match(a.blob!, b.blob!, a.features!, b.features!, this.matchOptions());
    this.matchResult = result;
    this.matchPage = 0;
    this.comparisonMode = 'interactive';
    this.renderAll();
    this.status = `Matching complete: ${result.num_matches} correspondences using ${result.metric.toUpperCase()}.`;
  }

  async requestServerOverlay(side: Side): Promise<void> {
    const slot = this.slots[side];
    if (this.isBusy || !slot.blob) return;
    if (slot.overlayImage) {
      slot.preview = 'server';
      this.renderSide(side);
      return;
    }
    this.isBusy = true;
    this.busyLabel = `Rendering ${side.toUpperCase()} overlay`;
    this.error = '';
    try {
      const blob = this.featureMode === 'sift'
        ? await this.lab.siftOverlay(slot.blob, this.siftOptions(), this.scaleCircleWithSigma ? null : this.circleRadius, this.drawOrientations)
        : await this.lab.cornerOverlay(slot.blob, this.cornerOptions(), this.circleRadius);
      const loaded = await this.loadBrowserImage(blob);
      this.clearOverlay(side);
      slot.overlayBlob = blob;
      slot.overlayImage = loaded.image;
      slot.overlayUrl = loaded.url;
      slot.preview = 'server';
      this.renderSide(side);
      this.status = `Image ${side.toUpperCase()}: rendered from the backend visualization endpoint.`;
    } catch (error) {
      this.error = await this.errorMessage(error);
    } finally {
      this.isBusy = false;
      this.busyLabel = '';
    }
  }

  setPreview(side: Side, mode: PreviewMode): void {
    if (mode === 'server') { void this.requestServerOverlay(side); return; }
    this.slots[side].preview = mode;
    this.renderSide(side);
  }

  async requestServerMatch(): Promise<void> {
    if (!this.canMatch || !this.matchResult) return;
    this.isBusy = true;
    this.busyLabel = 'Rendering backend match image';
    this.error = '';
    try {
      const response = await this.lab.visualizeMatches(
        this.slots.a.blob!, this.slots.b.blob!,
        this.slots.a.features!, this.slots.b.features!,
        this.matchOptions(), this.circleRadius, this.drawLines,
      );
      if (!response.body) throw new Error('The backend did not return a PNG.');
      const loaded = await this.loadBrowserImage(response.body);
      this.clearServerMatch();
      this.serverMatchBlob = response.body;
      this.serverMatchImage = loaded.image;
      this.serverMatchUrl = loaded.url;
      this.comparisonMode = 'server';
      this.renderComparison();
      this.status = `Backend visualization generated (${response.headers.get('X-Match-Count') ?? this.matches.length} matches).`;
    } catch (error) {
      this.error = await this.errorMessage(error);
    } finally {
      this.isBusy = false;
      this.busyLabel = '';
    }
  }

  private clearServerMatch(): void {
    if (this.serverMatchUrl) URL.revokeObjectURL(this.serverMatchUrl);
    this.serverMatchUrl = null;
    this.serverMatchImage = null;
    this.serverMatchBlob = null;
  }

  private clearMatch(): void {
    this.matchResult = null;
    this.entryCacheResult = null;
    this.entryCache = [];
    this.matchPage = 0;
    this.selectedMatchIndex = null;
    this.comparisonMode = 'interactive';
    this.clearServerMatch();
  }

  selectMatch(index: number): void {
    this.selectedMatchIndex = this.selectedMatchIndex === index ? null : index;
    if (this.selectedMatchIndex !== null) this.matchPage = Math.floor(index / this.matchPageSize);
    this.comparisonMode = 'interactive';
    this.renderAll();
    // The descriptor canvases are created by an @if block after Angular's next render.
    if (this.selectedMatchIndex !== null) setTimeout(() => this.renderDescriptors(), 0);
  }

  onComparisonClick(event: MouseEvent): void {
    if (!this.matchResult || !this.slots.a.image || !this.slots.b.image || this.comparisonMode === 'server') return;
    const canvas = this.comparisonCanvas?.nativeElement;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * canvas.width / rect.width / this.comparisonScale;
    const y = (event.clientY - rect.top) * canvas.height / rect.height / this.comparisonScale;
    const w1 = this.slots.a.image.naturalWidth;
    let bestIndex: number | null = null;
    let bestDistance = 18 * canvas.width / Math.max(rect.width, 1) / this.comparisonScale;
    for (const { match, index } of this.visibleMatchEntries) {
      const d1 = Math.hypot(x - match.point1[0], y - match.point1[1]);
      const d2 = Math.hypot(x - (match.point2[0] + w1), y - match.point2[1]);
      const d = Math.min(d1, d2);
      if (d < bestDistance) { bestIndex = index; bestDistance = d; }
    }
    if (bestIndex !== null) this.selectMatch(bestIndex);
  }

  private canvasFor(side: Side): HTMLCanvasElement | null {
    return side === 'a' ? this.imageCanvasA?.nativeElement ?? null : this.imageCanvasB?.nativeElement ?? null;
  }

  private renderAll(): void {
    this.renderSide('a');
    this.renderSide('b');
    this.renderComparison();
    this.renderDescriptors();
  }

  private renderSide(side: Side): void {
    const canvas = this.canvasFor(side);
    if (canvas) this.drawImagePort(side, canvas);
    if (side === 'a' && this.keypointCanvas?.nativeElement) {
      this.drawImagePort('a', this.keypointCanvas.nativeElement);
    }
  }

  private fitPreviewCanvas(canvas: HTMLCanvasElement, width: number, height: number): number {
    const scale = Math.min(1,
      Math.sqrt(this.previewPixelBudget / Math.max(1, width * height)),
      this.previewMaxDimension / Math.max(1, width, height));
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    return scale;
  }

  private drawImagePort(side: Side, canvas: HTMLCanvasElement): void {
    const slot = this.slots[side];
    const image = slot.image;
    if (!image) {
      canvas.width = 1;
      canvas.height = 1;
      return;
    }
    const scale = this.fitPreviewCanvas(canvas, image.naturalWidth, image.naturalHeight);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.save();
    ctx.scale(scale, scale);
    const serverImage = slot.preview === 'server' ? slot.overlayImage : null;
    ctx.drawImage(serverImage ?? image, 0, 0, image.naturalWidth, image.naturalHeight);
    if (slot.preview !== 'points' || !slot.features) {
      ctx.restore();
      return;
    }

    const keypoints = slot.features.keypoints;
    const selected = this.selectedMatch;
    const selectedIndex = side === 'a' ? selected?.index1 : selected?.index2;
    ctx.save();
    ctx.strokeStyle = '#fa4257';
    ctx.fillStyle = '#fa4257';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    let inPath = 0;
    for (let i = 0; i < keypoints.length; i++) {
      const [x, y] = keypoints[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const kp = slot.features.keypoint_details?.[i];
      const radius = this.scaleCircleWithSigma && this.featureMode === 'sift' &&
        kp?.sigma !== undefined && Number.isFinite(kp.sigma)
        ? Math.max(2, kp.sigma * 2) : Math.max(2, this.circleRadius);
      ctx.moveTo(x + radius, y);
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      if (this.drawOrientations && kp?.angle !== undefined) {
        const rad = kp.angle * Math.PI / 180;
        ctx.moveTo(x, y);
        ctx.lineTo(x + radius * Math.cos(rad), y + radius * Math.sin(rad));
      }
      // Stroke the path periodically to avoid an unbounded Canvas path.
      if (++inPath === 2048) {
        ctx.stroke();
        ctx.beginPath();
        inPath = 0;
      }
    }
    if (inPath) ctx.stroke();
    if (selectedIndex !== undefined && selectedIndex !== null && keypoints[selectedIndex]) {
      const [x, y] = keypoints[selectedIndex];
      ctx.strokeStyle = '#18c8d1';
      ctx.lineWidth = 3.4;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, this.circleRadius) + 5, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
    ctx.restore();
  }

  private matchColor(index: number): string {
    return `hsl(${Math.round(index * 137.508) % 360}, 78%, 45%)`;
  }

  private renderComparison(): void {
    const renderVersion = ++this.comparisonRenderVersion;
    const canvas = this.comparisonCanvas?.nativeElement;
    if (!canvas) return;
    const first = this.slots.a.image;
    const second = this.slots.b.image;
    const w1 = first?.naturalWidth ?? 0;
    const w2 = second?.naturalWidth ?? 0;
    const h1 = first?.naturalHeight ?? 0;
    const h2 = second?.naturalHeight ?? 0;
    if (!w1 && !w2) {
      canvas.width = 1; canvas.height = 1;
      return;
    }
    const actualWidth = Math.max(1, w1 + w2);
    const actualHeight = Math.max(1, h1, h2);
    this.comparisonScale = this.fitPreviewCanvas(canvas, actualWidth, actualHeight);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(this.comparisonScale, this.comparisonScale);
    ctx.fillStyle = '#f6f8ff';
    ctx.fillRect(0, 0, actualWidth, actualHeight);

    if (this.comparisonMode === 'server' && this.serverMatchImage) {
      ctx.drawImage(this.serverMatchImage, 0, 0, actualWidth, actualHeight);
      return;
    }
    if (first) ctx.drawImage(first, 0, 0);
    if (second) ctx.drawImage(second, w1, 0);
    if (!this.matchResult) return;

    const entries = this.visibleMatchEntries;
    // Draw in animation-frame batches: all matches remain present, but massive
    // results do not freeze the browser's main thread for one long frame.
    const batchSize = 1200;
    const paintBatch = (kind: 'lines' | 'circles', from: number): void => {
      if (this.destroyed || renderVersion !== this.comparisonRenderVersion) return;
      const until = Math.min(entries.length, from + batchSize);
      for (let i = from; i < until; i++) {
        const { match, index } = entries[i];
        ctx.save();
        ctx.strokeStyle = this.selectedMatchIndex === index ? '#0fbbc7' : this.matchColor(index);
        ctx.lineWidth = this.selectedMatchIndex === index ? 3.7 : (kind === 'lines' ? 1.8 : 2);
        ctx.globalAlpha = this.selectedMatchIndex === null || this.selectedMatchIndex === index
          ? (kind === 'lines' ? 0.78 : 1) : (kind === 'lines' ? 0.13 : 0.18);
        if (kind === 'lines') {
          ctx.beginPath();
          ctx.moveTo(match.point1[0], match.point1[1]);
          ctx.lineTo(match.point2[0] + w1, match.point2[1]);
          ctx.stroke();
        } else {
          const radius = Math.max(1, this.circleRadius) + (this.selectedMatchIndex === index ? 4 : 0);
          for (const [x, y] of [match.point1, [match.point2[0] + w1, match.point2[1]]]) {
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
        ctx.restore();
      }
      if (until < entries.length) {
        requestAnimationFrame(() => paintBatch(kind, until));
      } else if (kind === 'lines' && this.drawCircles) {
        requestAnimationFrame(() => paintBatch('circles', 0));
      }
    };
    if (this.drawLines) paintBatch('lines', 0);
    else if (this.drawCircles) paintBatch('circles', 0);
  }

  private renderDescriptors(): void {
    this.drawDescriptor(this.descriptorCanvasA?.nativeElement, this.matchDescriptorA);
    this.drawDescriptor(this.descriptorCanvasB?.nativeElement, this.matchDescriptorB);
  }

  private drawDescriptor(canvas: HTMLCanvasElement | undefined, values: number[]): void {
    if (!canvas) return;
    canvas.width = 360;
    canvas.height = 100;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#f7f9ff';
    ctx.fillRect(0, 0, 360, 100);
    if (!values.length) return;
    if (values.length === 64) {
      // MOPS: visualize normalized 8x8 intensity patch.
      for (let i = 0; i < 64; i++) {
        const g = Math.max(0, Math.min(255, Math.round(128 + values[i] * 38)));
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(15 + (i % 8) * 11, 5 + Math.floor(i / 8) * 11, 11, 11);
      }
      ctx.strokeStyle = '#8896ad';
      ctx.strokeRect(15, 5, 88, 88);
      ctx.fillStyle = '#71809a';
      ctx.font = '12px sans-serif';
      ctx.fillText('MOPS · 8×8 normalized intensity', 123, 52);
    } else {
      // SIFT: 128 histogram bins, grouped into 16 cells (8 orientation bins each).
      const max = Math.max(...values, 0.0001);
      for (let i = 0; i < values.length; i++) {
        const x = i * 360 / values.length;
        const height = Math.min(84, Math.max(0, values[i] / max * 79));
        ctx.fillStyle = `hsl(${Math.floor(i / 8) * 23}, 76%, 48%)`;
        ctx.fillRect(x + 0.4, 91 - height, Math.max(1, 360 / values.length - 0.7), height);
      }
      ctx.strokeStyle = '#dce3f1';
      ctx.beginPath(); ctx.moveTo(0, 92); ctx.lineTo(360, 92); ctx.stroke();
    }
  }

  async downloadViewport(side: Side): Promise<void> {
    const canvas = this.canvasFor(side);
    if (!canvas || !this.slots[side].image) return;
    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(value => value ? resolve(value) : reject(new Error('PNG encoding failed.')), 'image/png');
    });
    this.downloadBlob(blob, `image-${side}-${this.slots[side].preview}.png`);
  }

  async downloadKeypointPreview(): Promise<void> {
    const canvas = this.keypointCanvas?.nativeElement;
    if (!canvas || !this.slots.a.image) return;
    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(value => value ? resolve(value) : reject(new Error('PNG encoding failed.')), 'image/png');
    });
    this.downloadBlob(blob, `keypoints-${this.slots.a.preview}.png`);
  }

  async downloadComparison(): Promise<void> {
    const canvas = this.comparisonCanvas?.nativeElement;
    if (!canvas || !this.slots.a.image || !this.slots.b.image) return;
    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(value => value ? resolve(value) : reject(new Error('PNG encoding failed.')), 'image/png');
    });
    this.downloadBlob(blob, 'feature-correspondences.png');
  }

  downloadFeatures(side: Side): void {
    const features = this.slots[side].features;
    if (!features) return;
    this.downloadJson(features, `image-${side}-features.json`);
  }

  downloadMatches(): void {
    if (this.matchResult) this.downloadJson(this.matchResult, 'feature-matches.json');
  }

  downloadRaw(side: Side): void {
    const blob = this.slots[side].blob;
    if (blob) this.downloadBlob(blob, `image-${side}-original.${blob.type === 'image/jpeg' ? 'jpg' : 'png'}`);
  }

  private downloadJson(value: unknown, name: string): void {
    this.downloadBlob(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }), name);
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

  private async errorMessage(error: unknown): Promise<string> {
    if (error instanceof HttpErrorResponse) {
      let payload = error.error;
      if (payload instanceof Blob) {
        try { payload = JSON.parse(await payload.text()); } catch { payload = null; }
      }
      if (typeof payload?.detail === 'string') return payload.detail;
      if (Array.isArray(payload?.detail)) {
        return payload.detail.map((issue: any) => {
          const location = Array.isArray(issue.loc) ? issue.loc.join('.') : 'request';
          return `${location}: ${issue.msg ?? 'Invalid value'}`;
        }).join('; ');
      }
      return error.status === 0
        ? 'FastAPI is unreachable. Check that your backend is running on the configured API URL.'
        : `Request failed with HTTP ${error.status}.`;
    }
    return error instanceof Error ? error.message : String(error);
  }
}
