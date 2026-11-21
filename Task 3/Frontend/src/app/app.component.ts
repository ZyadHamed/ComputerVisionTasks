import { Component, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { SiftService, SiftParams, SiftResult, MatchParams } from './sift.service';

type Tab = 'keypoints' | 'matching';

interface ImageSlot {
  file: File | null;
  blob: Blob | null;
  url: string | null;
  name: string;
}

function emptySlot(): ImageSlot {
  return { file: null, blob: null, url: null, name: '' };
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnDestroy {
  // ─── Tab state ───────────────────────────────────────────────────────────────
  activeTab: Tab = 'keypoints';

  // ─── Shared SIFT detection parameters ────────────────────────────────────────
  maxKeypoints = 500;
  contrastThreshold = 0.03;
  edgeThreshold = 10.0;
  numOctaves = 3;
  numIntervals = 3;
  sigma = 1.6;
  assumedBlur = 0.5;

  // ─── Visualization options (Tab 1) ───────────────────────────────────────────
  circleRadius: number | null = null;
  drawOrientation = true;

  // ─── Matching parameters (Tab 2) ─────────────────────────────────────────────
  matchMetric: 'ssd' | 'ncc' = 'ssd';
  ratioThreshold = 0.75;
  crossCheck = false;
  maxMatches: number | null = null;
  useLatestForImage1 = false;
  useExtractedSiftForImage1 = false;

  // ─── Collapsible sections ─────────────────────────────────────────────────────
  detectionOpen = true;
  vizOpen = true;
  matchDetectionOpen = true;
  matchParamsOpen = true;

  // ─── Tab 1 state ─────────────────────────────────────────────────────────────
  kpImageSlot: ImageSlot = emptySlot();
  kpOutputUrl: string | null = null;
  kpKeypointCount: number | null = null;
  kpTimeMs: number | null = null;
  kpSiftResult: SiftResult | null = null;
  kpLoading = false;
  kpError = '';
  kpDragOver = false;

  // ─── Tab 2 state ─────────────────────────────────────────────────────────────
  matchSlot1: ImageSlot = emptySlot();
  matchSlot2: ImageSlot = emptySlot();
  matchLatestImageUrl: string | null = null;
  matchOutputUrl: string | null = null;
  matchCount: number | null = null;
  matchTimeMs: number | null = null;
  matchDescMs: number | null = null;
  matchFeatures1: SiftResult | null = null;
  matchFeatures2: SiftResult | null = null;
  matchLoading = false;
  matchError = '';
  matchDrag1 = false;
  matchDrag2 = false;

  constructor(private readonly sift: SiftService) {}

  ngOnDestroy(): void {
    this.revokeUrl(this.kpImageSlot);
    this.revokeUrl(this.matchSlot1);
    this.revokeUrl(this.matchSlot2);
    this.clearLatestImagePreview();
    if (this.kpOutputUrl) URL.revokeObjectURL(this.kpOutputUrl);
    if (this.matchOutputUrl) URL.revokeObjectURL(this.matchOutputUrl);
  }

  // ─── Tab switching ────────────────────────────────────────────────────────────
  setTab(tab: Tab): void {
    this.activeTab = tab;
  }

  // ─── File upload helpers ──────────────────────────────────────────────────────
  private assignSlot(slot: ImageSlot, file: File): void {
    if (slot.url) URL.revokeObjectURL(slot.url);
    slot.file = file;
    slot.blob = file;
    slot.url = URL.createObjectURL(file);
    slot.name = file.name;
  }

  private revokeUrl(slot: ImageSlot): void {
    if (slot.url) {
      URL.revokeObjectURL(slot.url);
      slot.url = null;
    }
  }

  // ─── Tab 1: file upload ───────────────────────────────────────────────────────
  onKpFileInput(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) {
      this.assignSlot(this.kpImageSlot, file);
      this.kpSiftResult = null;
      this.useExtractedSiftForImage1 = false;
      this.clearKpOutput();
      this.kpError = '';
    }
    input.value = '';
  }

  onKpDragOver(event: DragEvent): void {
    event.preventDefault();
    this.kpDragOver = true;
  }

  onKpDragLeave(): void {
    this.kpDragOver = false;
  }

  onKpDrop(event: DragEvent): void {
    event.preventDefault();
    this.kpDragOver = false;
    const file = event.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) {
      this.assignSlot(this.kpImageSlot, file);
      this.kpSiftResult = null;
      this.useExtractedSiftForImage1 = false;
      this.clearKpOutput();
      this.kpError = '';
    }
  }

  private clearKpOutput(): void {
    if (this.kpOutputUrl) URL.revokeObjectURL(this.kpOutputUrl);
    this.kpOutputUrl = null;
    this.kpKeypointCount = null;
    this.kpTimeMs = null;
  }

  // ─── Tab 1: detect keypoints (JSON) ──────────────────────────────────────────
  async detectKeypoints(): Promise<void> {
    if (this.kpLoading) return;
    this.kpLoading = true;
    this.kpError = '';
    this.clearKpOutput();

    try {
      const params = this.buildSiftParams();
      const result = await this.sift.extractSift(this.kpImageSlot.blob, params);
      this.kpSiftResult = result;
      this.kpKeypointCount = result.num_keypoints;
      this.kpTimeMs = result.execution_time_ms;
      // Draw keypoints on a canvas overlay
      this.drawKeypointsOnCanvas(result);
    } catch (err) {
      this.kpError = await this.extractError(err);
    } finally {
      this.kpLoading = false;
    }
  }

  // ─── Tab 1: visualize (PNG) ───────────────────────────────────────────────────
  async visualizeKeypoints(): Promise<void> {
    if (this.kpLoading) return;
    this.kpLoading = true;
    this.kpError = '';
    this.clearKpOutput();

    try {
      const params = this.buildSiftParams();
      const result = await this.sift.visualizeSift(this.kpImageSlot.blob, {
        ...params,
        circle_radius: this.circleRadius,
        draw_orientation: this.drawOrientation,
      });
      this.kpSiftResult = null; // PNG mode — no raw JSON result
      this.useExtractedSiftForImage1 = false;
      this.kpOutputUrl = URL.createObjectURL(result.blob);
      this.kpKeypointCount = result.keypointCount;
      this.kpTimeMs = result.timeMs;
    } catch (err) {
      this.kpError = await this.extractError(err);
    } finally {
      this.kpLoading = false;
    }
  }

  private buildSiftParams(): SiftParams {
    return {
      max_keypoints: this.maxKeypoints,
      contrast_threshold: this.contrastThreshold,
      edge_threshold: this.edgeThreshold,
      num_octaves: this.numOctaves,
      num_intervals: this.numIntervals,
      sigma: this.sigma,
      assumed_blur: this.assumedBlur,
    };
  }

  private drawKeypointsOnCanvas(result: SiftResult): void {
    const img = new Image();
    const src = this.kpImageSlot.url;
    if (!src) return;

    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);

      ctx.save();
      for (const kp of result.keypoint_details) {
        const r = Math.max(2, kp.sigma * 2);
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(65,102,229,0.9)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        if (this.drawOrientation) {
          const rad = (kp.angle * Math.PI) / 180;
          ctx.beginPath();
          ctx.moveTo(kp.x, kp.y);
          ctx.lineTo(kp.x + r * Math.cos(rad), kp.y + r * Math.sin(rad));
          ctx.strokeStyle = 'rgba(255,80,80,0.9)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
      ctx.restore();

      canvas.toBlob((blob) => {
        if (blob) {
          if (this.kpOutputUrl) URL.revokeObjectURL(this.kpOutputUrl);
          this.kpOutputUrl = URL.createObjectURL(blob);
        }
      }, 'image/png');
    };
    img.src = src;
  }

  // ─── Tab 2: file upload (slot 1 & 2) ─────────────────────────────────────────
  onMatch1FileInput(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) {
      this.assignSlot(this.matchSlot1, file);
      this.useLatestForImage1 = false;
      this.useExtractedSiftForImage1 = false;
      this.matchFeatures1 = null;
      this.clearMatchOutput();
      this.matchError = '';
    }
    input.value = '';
  }

  onMatch2FileInput(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) {
      this.assignSlot(this.matchSlot2, file);
      this.matchFeatures2 = null;
      this.clearMatchOutput();
      this.matchError = '';
    }
    input.value = '';
  }

  onMatch1DragOver(event: DragEvent): void { event.preventDefault(); this.matchDrag1 = true; }
  onMatch1DragLeave(): void { this.matchDrag1 = false; }
  onMatch1Drop(event: DragEvent): void {
    event.preventDefault();
    this.matchDrag1 = false;
    const file = event.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) {
      this.assignSlot(this.matchSlot1, file);
      this.useLatestForImage1 = false;
      this.useExtractedSiftForImage1 = false;
      this.matchFeatures1 = null;
      this.clearMatchOutput();
      this.matchError = '';
    }
  }

  onMatch2DragOver(event: DragEvent): void { event.preventDefault(); this.matchDrag2 = true; }
  onMatch2DragLeave(): void { this.matchDrag2 = false; }
  onMatch2Drop(event: DragEvent): void {
    event.preventDefault();
    this.matchDrag2 = false;
    const file = event.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) {
      this.assignSlot(this.matchSlot2, file);
      this.matchFeatures2 = null;
      this.clearMatchOutput();
      this.matchError = '';
    }
  }

  private clearMatchOutput(): void {
    if (this.matchOutputUrl) URL.revokeObjectURL(this.matchOutputUrl);
    this.matchOutputUrl = null;
    this.matchCount = null;
    this.matchTimeMs = null;
    this.matchDescMs = null;
  }

  clearLatestImagePreview(): void {
    if (this.matchLatestImageUrl) URL.revokeObjectURL(this.matchLatestImageUrl);
    this.matchLatestImageUrl = null;
  }

  // ─── Tab 2: extract & match ───────────────────────────────────────────────────
  get canMatch(): boolean {
    const hasImg2 = this.matchSlot2.blob !== null;
    const hasImg1 = this.useExtractedSiftForImage1
      ? this.kpSiftResult !== null && this.kpImageSlot.blob !== null
      : this.useLatestForImage1 || this.matchSlot1.blob !== null;
    return hasImg1 && hasImg2 && !this.matchLoading;
  }

  async extractAndMatch(): Promise<void> {
    if (!this.canMatch) return;
    this.matchLoading = true;
    this.matchError = '';
    this.clearMatchOutput();

    try {
      const siftParams = this.buildSiftParams();
      let img1Blob: Blob;
      let features1: SiftResult;
      if (this.useExtractedSiftForImage1) {
        if (!this.kpImageSlot.blob || !this.kpSiftResult) {
          throw new Error('Detect keypoints first to use the extracted SIFT descriptors.');
        }
        img1Blob = this.kpImageSlot.blob;
        features1 = this.kpSiftResult;
      } else {
        let selectedImage = this.matchSlot1.blob;
        if (this.useLatestForImage1) {
          this.clearLatestImagePreview();
          selectedImage = await this.sift.getLatestImage();
          this.matchLatestImageUrl = URL.createObjectURL(selectedImage);
        }
        if (!selectedImage) {
          throw new Error('Select an image 1 or use the latest stored image.');
        }
        img1Blob = selectedImage;
        features1 = await this.sift.extractSift(img1Blob, siftParams);
      }
      this.matchFeatures1 = features1;

      // Extract features for image 2
      const img2Blob = this.matchSlot2.blob;
      if (!img2Blob) {
        throw new Error('Select an image 2 to match against.');
      }
      const features2 = await this.sift.extractSift(img2Blob, siftParams);
      this.matchFeatures2 = features2;

      // Visualize matches
      const matchParams: MatchParams = {
        method: 'sift',
        metric: this.matchMetric,
        ratio: this.ratioThreshold,
        cross_check: this.crossCheck,
        max_matches: this.maxMatches,
      };

      const result = await this.sift.visualizeMatches(
        img1Blob,
        img2Blob,
        features1,
        features2,
        matchParams,
      );

      this.matchOutputUrl = URL.createObjectURL(result.blob);
      this.matchCount = result.matchCount;
      this.matchTimeMs = result.matchTimeMs;
      this.matchDescMs = result.descTimeMs;
    } catch (err) {
      this.matchError = await this.extractError(err);
    } finally {
      this.matchLoading = false;
    }
  }

  // ─── Utilities ────────────────────────────────────────────────────────────────
  private async extractError(err: unknown): Promise<string> {
    if (typeof err === 'object' && err !== null) {
      const e = err as Record<string, unknown>;
      const payload = e['error'];
      if (payload instanceof Blob) {
        const text = await payload.text();
        if (text) {
          try {
            const body = JSON.parse(text) as Record<string, unknown>;
            if (typeof body['detail'] === 'string') return body['detail'];
          } catch {
            return text;
          }
        }
      } else if (payload && typeof payload === 'object') {
        const detail = (payload as Record<string, unknown>)['detail'];
        if (typeof detail === 'string') return detail;
      }
      if (typeof e['message'] === 'string') return e['message'];
      if (typeof e['status'] === 'number') return `HTTP ${e['status']} error`;
    }
    if (err instanceof Error) return err.message;
    return 'An unexpected error occurred.';
  }

  formatMs(ms: number | null): string {
    if (ms === null) return '—';
    return ms < 1000 ? `${ms.toFixed(1)} ms` : `${(ms / 1000).toFixed(2)} s`;
  }
}
