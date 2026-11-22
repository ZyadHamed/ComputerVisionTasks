import { Injectable } from '@angular/core';
import { HttpClient, HttpResponse } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

export type CornerDetector = 'harris' | 'lamda_minus';
export type FeatureMode = 'corners' | 'sift';
export type DescriptorMethod = 'sift' | 'mops';
export type MatchMetric = 'ssd' | 'ncc';

export interface FeatureDetail {
  x: number;
  y: number;
  sigma?: number;
  angle?: number;
  response?: number;
  octave?: number;
  scale?: number;
}

export interface FeatureSet {
  keypoints: number[][];
  descriptors: number[][];
  keypoint_details: FeatureDetail[];
  num_keypoints: number;
  execution_time_ms: number;
  detector_method?: string;
  descriptor_method?: string;
}

export interface MatchPair {
  index1: number;
  index2: number;
  point1: [number, number];
  point2: [number, number];
  score: number;
}

export interface MatchResponse {
  method: string;
  metric: MatchMetric;
  num_keypoints1: number;
  num_keypoints2: number;
  dropped_near_border1: number;
  dropped_near_border2: number;
  num_matches: number;
  descriptor_time_ms: number;
  matching_time_ms: number;
  matches: MatchPair[];
}

export interface CornerOptions {
  method: CornerDetector;
  descriptor_method: DescriptorMethod;
  threshold: number;
  window_size: number;
  margin: number | null;
  max_keypoints: number | null;
}

export interface SiftOptions {
  max_keypoints: number | null;
  contrast_threshold: number;
  edge_threshold: number;
  num_octaves: number;
  num_intervals: number;
  sigma: number;
  assumed_blur: number;
  num_cells: number;
  num_bins: number;
  peak_ratio: number;
  descriptor_threshold: number;
}

export interface MatchOptions {
  method: 'harris' | 'sift';
  metric: MatchMetric;
  ratio: number;
  threshold: number | null;
  cross_check: boolean;
  patch_size: number;
  max_matches: number | null;
}

type FormFields = Record<string, string | number | boolean | null | undefined>;

@Injectable({ providedIn: 'root' })
export class FeatureLabService {
  // Change this value if FastAPI is running on another host/port.
  readonly apiUrl = 'http://localhost:8000';

  constructor(private readonly http: HttpClient) {}

  getLatestImage(): Promise<Blob> {
    return firstValueFrom(this.http.get(`${this.apiUrl}/images/latest`, { responseType: 'blob' }));
  }

  async uploadLatestImage(image: Blob): Promise<void> {
    const body = new FormData();
    body.append('file', image, 'latest-image.png');
    await firstValueFrom(this.http.post(`${this.apiUrl}/images/upload`, body));
  }

  private form(fields: FormFields): FormData {
    const form = new FormData();
    for (const [key, value] of Object.entries(fields)) {
      if (value !== null && value !== undefined && value !== '') {
        form.append(key, String(value));
      }
    }
    return form;
  }

  private withImage(image: Blob, fields: FormFields): FormData {
    const form = this.form(fields);
    form.append('file', image, 'feature-source.png');
    return form;
  }

  private matchForm(
    image1: Blob, image2: Blob, features1: FeatureSet, features2: FeatureSet,
    params: MatchOptions, extra: FormFields = {},
  ): FormData {
    const form = this.form({ ...params, ...extra });
    form.append('image1', image1, 'image-a.png');
    form.append('image2', image2, 'image-b.png');
    // One descriptor per keypoint; include only the fields required by the matching API.
    // JSON is uploaded as a FILE part, never a regular text field.
    // This matches FastAPI UploadFile parameters and bypasses the multipart
    // parser's 1 MiB per-plain-text-field limit for large descriptor arrays.
    form.append('features1', new Blob([
      JSON.stringify({ keypoints: features1.keypoints, descriptors: features1.descriptors })
    ], { type: 'application/json' }), 'features1.json');
    form.append('features2', new Blob([
      JSON.stringify({ keypoints: features2.keypoints, descriptors: features2.descriptors })
    ], { type: 'application/json' }), 'features2.json');
    return form;
  }

  extractCorners(image: Blob, options: CornerOptions): Promise<FeatureSet> {
    const body = this.withImage(image, { ...options, save_as_latest: false });
    return firstValueFrom(this.http.post<FeatureSet>(`${this.apiUrl}/corners`, body));
  }

  extractSift(image: Blob, options: SiftOptions): Promise<FeatureSet> {
    const body = this.withImage(image, { ...options, save_as_latest: false });
    return firstValueFrom(this.http.post<FeatureSet>(`${this.apiUrl}/features/sift`, body));
  }

  cornerOverlay(image: Blob, options: CornerOptions, circleRadius: number): Promise<Blob> {
    const body = this.withImage(image, {
      method: options.method,
      threshold: options.threshold,
      window_size: options.window_size,
      margin: options.margin ?? (options.descriptor_method === 'mops' ? 30 : 10),
      max_keypoints: options.max_keypoints,
      circle_radius: circleRadius,
      // Supported by the optional no-save backend patch in the README.
      save_as_latest: false,
    });
    return firstValueFrom(this.http.post(`${this.apiUrl}/corners/overlay`, body, { responseType: 'blob' }));
  }

  siftOverlay(image: Blob, options: SiftOptions, radius: number | null, drawOrientation: boolean): Promise<Blob> {
    const body = this.withImage(image, {
      ...options, circle_radius: radius, draw_orientation: drawOrientation, save_as_latest: false,
    });
    return firstValueFrom(this.http.post(`${this.apiUrl}/features/sift/visualize`, body, { responseType: 'blob' }));
  }

  match(image1: Blob, image2: Blob, features1: FeatureSet, features2: FeatureSet,
        options: MatchOptions): Promise<MatchResponse> {
    return firstValueFrom(this.http.post<MatchResponse>(
      `${this.apiUrl}/features/match`,
      this.matchForm(image1, image2, features1, features2, options),
    ));
  }

  visualizeMatches(
    image1: Blob, image2: Blob, features1: FeatureSet, features2: FeatureSet,
    options: MatchOptions, circleRadius: number, drawLines: boolean,
  ): Promise<HttpResponse<Blob>> {
    return firstValueFrom(this.http.post(`${this.apiUrl}/features/match/visualize`,
      this.matchForm(image1, image2, features1, features2, options, {
        circle_radius: circleRadius, draw_lines: drawLines,
      }), { responseType: 'blob', observe: 'response' }));
  }
}
