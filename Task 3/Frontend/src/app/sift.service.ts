import { Injectable } from '@angular/core';
import { HttpClient, HttpResponse } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

export interface KeypointDetail {
  x: number;
  y: number;
  sigma: number;
  angle: number;
  response: number;
  octave: number;
  scale: number;
}

export interface SiftResult {
  keypoints: number[][];
  descriptors: number[][];
  keypoint_details: KeypointDetail[];
  num_keypoints: number;
  execution_time_ms: number;
}

export interface MatchEntry {
  index1: number;
  index2: number;
  point1: number[];
  point2: number[];
  score: number;
}

export interface MatchResult {
  method: string;
  metric: string;
  num_keypoints1: number;
  num_keypoints2: number;
  num_matches: number;
  matching_time_ms: number;
  descriptor_time_ms: number;
  matches: MatchEntry[];
}

export interface SiftParams {
  max_keypoints?: number;
  contrast_threshold?: number;
  edge_threshold?: number;
  num_octaves?: number;
  num_intervals?: number;
  sigma?: number;
  assumed_blur?: number;
  save_as_latest?: boolean;
}

export interface VisualizeParams extends SiftParams {
  circle_radius?: number | null;
  draw_orientation?: boolean;
}

export interface MatchParams {
  method?: 'sift' | 'harris';
  metric?: 'ssd' | 'ncc';
  ratio?: number;
  cross_check?: boolean;
  max_matches?: number | null;
}

@Injectable({ providedIn: 'root' })
export class SiftService {
  readonly apiUrl = 'http://localhost:8000';

  constructor(private readonly http: HttpClient) {}

  async uploadImage(file: File): Promise<void> {
    const form = new FormData();
    form.append('file', file);
    await firstValueFrom(
      this.http.post(`${this.apiUrl}/images/upload`, form)
    );
  }

  async getLatestImage(): Promise<Blob> {
    return firstValueFrom(
      this.http.get(`${this.apiUrl}/images/latest`, { responseType: 'blob' })
    );
  }

  async extractSift(file: Blob | null, params: SiftParams): Promise<SiftResult> {
    const form = this.buildSiftForm(file, params);
    return firstValueFrom(
      this.http.post<SiftResult>(`${this.apiUrl}/features/sift`, form)
    );
  }

  async visualizeSift(
    file: Blob | null,
    params: VisualizeParams,
  ): Promise<{ blob: Blob; keypointCount: number; timeMs: number }> {
    const form = this.buildSiftForm(file, params);
    if (params.circle_radius != null) form.append('circle_radius', String(params.circle_radius));
    form.append('draw_orientation', String(params.draw_orientation ?? true));

    const response = await firstValueFrom(
      this.http.post(`${this.apiUrl}/features/sift/visualize`, form, {
        observe: 'response',
        responseType: 'blob',
      })
    );
    const blob = response.body!;
    const keypointCount = parseInt(response.headers.get('X-Keypoint-Count') ?? '0', 10);
    const timeMs = parseFloat(response.headers.get('X-Execution-Time-Ms') ?? '0');
    return { blob, keypointCount, timeMs };
  }

  async matchFeatures(
    image1: Blob | null,
    image2: Blob,
    features1: SiftResult,
    features2: SiftResult,
    params: MatchParams,
  ): Promise<MatchResult> {
    const form = this.buildMatchForm(image1, image2, features1, features2, params);
    return firstValueFrom(
      this.http.post<MatchResult>(`${this.apiUrl}/features/match`, form)
    );
  }

  async visualizeMatches(
    image1: Blob | null,
    image2: Blob,
    features1: SiftResult,
    features2: SiftResult,
    params: MatchParams,
  ): Promise<{ blob: Blob; matchCount: number; matchTimeMs: number; descTimeMs: number }> {
    const form = this.buildMatchForm(image1, image2, features1, features2, params);
    const response = await firstValueFrom(
      this.http.post(`${this.apiUrl}/features/match/visualize`, form, {
        observe: 'response',
        responseType: 'blob',
      })
    );
    const blob = response.body!;
    const matchCount = parseInt(response.headers.get('X-Match-Count') ?? '0', 10);
    const matchTimeMs = parseFloat(response.headers.get('X-Matching-Time-Ms') ?? '0');
    const descTimeMs = parseFloat(response.headers.get('X-Descriptor-Time-Ms') ?? '0');
    return { blob, matchCount, matchTimeMs, descTimeMs };
  }

  // --- Private helpers ---

  private buildSiftForm(file: Blob | null, params: SiftParams): FormData {
    const form = new FormData();
    if (file) form.append('file', file, 'image.png');
    if (params.max_keypoints != null) form.append('max_keypoints', String(params.max_keypoints));
    if (params.contrast_threshold != null) form.append('contrast_threshold', String(params.contrast_threshold));
    if (params.edge_threshold != null) form.append('edge_threshold', String(params.edge_threshold));
    if (params.num_octaves != null) form.append('num_octaves', String(params.num_octaves));
    if (params.num_intervals != null) form.append('num_intervals', String(params.num_intervals));
    if (params.sigma != null) form.append('sigma', String(params.sigma));
    if (params.assumed_blur != null) form.append('assumed_blur', String(params.assumed_blur));
    if (params.save_as_latest != null) form.append('save_as_latest', String(params.save_as_latest));
    return form;
  }

  private buildMatchForm(
    image1: Blob | null,
    image2: Blob,
    features1: SiftResult,
    features2: SiftResult,
    params: MatchParams,
  ): FormData {
    const form = new FormData();
    if (image1) form.append('image1', image1, 'image1.png');
    form.append('image2', image2, 'image2.png');
    form.append('features1', new Blob([JSON.stringify({
      keypoints: features1.keypoints,
      descriptors: features1.descriptors,
    })], { type: 'application/json' }), 'features1.json');
    form.append('features2', new Blob([JSON.stringify({
      keypoints: features2.keypoints,
      descriptors: features2.descriptors,
    })], { type: 'application/json' }), 'features2.json');
    if (params.method) form.append('method', params.method);
    if (params.metric) form.append('metric', params.metric);
    if (params.ratio != null) form.append('ratio', String(params.ratio));
    if (params.cross_check != null) form.append('cross_check', String(params.cross_check));
    if (params.max_matches != null) form.append('max_matches', String(params.max_matches));
    return form;
  }
}
