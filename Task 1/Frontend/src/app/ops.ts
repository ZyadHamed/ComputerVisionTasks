export type PortId = 'input' | 'output1' | 'output2';
export const PORTS: { id: PortId; title: string }[] = [
  { id: 'input', title: 'Input' }, { id: 'output1', title: 'Output 1' }, { id: 'output2', title: 'Output 2' }
];
export interface Param {
  key: string; label: string; type: 'range' | 'bool' | 'select';
  min?: number; max?: number; step?: number; def: any; options?: string[];
  showIf?: (v: any) => boolean;
}
export interface Op {
  id: string; label: string; endpoint: string; params: Param[];
  kind: 'image' | 'json' | 'copy'; dual?: boolean; note?: string;
}
const blur: Param[] = [
  { key: 'apply_blur', label: 'Apply blur', type: 'bool', def: false },
  { key: 'sigma', label: 'Sigma', type: 'range', min: 0.1, max: 10, step: 0.1, def: 1, showIf: v => v.apply_blur },
  { key: 'manual', label: 'Manual implementation', type: 'bool', def: true },
];
export const OPS: Op[] = [
  { id: 'copy', label: 'Copy / Move (no operation)', endpoint: '', params: [], kind: 'copy' },
  { id: 'grayscale', label: 'Grayscale', endpoint: '/image/grayscale', kind: 'image',
    params: [{ key: 'mode', label: 'Mode', type: 'select', options: ['weighted', 'average'], def: 'weighted' }] },
  { id: 'normalize', label: 'Normalize', endpoint: '/image/normalize', kind: 'image', params: [] },
  { id: 'noise', label: 'Add Noise', endpoint: '/image/noise', kind: 'image', params: [
    { key: 'noise_type', label: 'Noise type', type: 'select', options: ['Gaussian', 'Uniform', 'Salt and Pepper'], def: 'Gaussian' },
    { key: 'mean', label: 'Mean', type: 'range', min: -100, max: 100, step: 1, def: 0, showIf: v => v.noise_type === 'Gaussian' },
    { key: 'sigma', label: 'Sigma', type: 'range', min: 1, max: 100, step: 1, def: 25, showIf: v => v.noise_type === 'Gaussian' },
    { key: 'low', label: 'Low', type: 'range', min: -255, max: 0, step: 1, def: -50, showIf: v => v.noise_type === 'Uniform' },
    { key: 'high', label: 'High', type: 'range', min: 1, max: 255, step: 1, def: 50, showIf: v => v.noise_type === 'Uniform' },
    { key: 'p', label: 'Probability (p)', type: 'range', min: 0, max: 1, step: 0.01, def: 0.05, showIf: v => v.noise_type === 'Salt and Pepper' },
  ] },
  { id: 'lowpass', label: 'Low-pass Filter (spatial)', endpoint: '/filters/low-pass', kind: 'image', params: [
    { key: 'filter_type', label: 'Filter type', type: 'select', options: ['Average', 'Gaussian', 'Median'], def: 'Average' },
    { key: 'kernel_size', label: 'Kernel size (odd)', type: 'range', min: 3, max: 31, step: 2, def: 3 },
    { key: 'sigma', label: 'Sigma', type: 'range', min: 0.1, max: 10, step: 0.1, def: 1, showIf: v => v.filter_type === 'Gaussian' },
    { key: 'manual', label: 'Manual implementation', type: 'bool', def: false },
  ] },
  { id: 'highpass', label: 'High-pass Filter (frequency)', endpoint: '/frequency-filter/high-pass', kind: 'image',
    params: [{ key: 'radius', label: 'Radius', type: 'range', min: 1, max: 300, step: 1, def: 30 }] },
  { id: 'lowpass_freq', label: 'Low-pass Filter (frequency)', endpoint: '/frequency-filter/low-pass', kind: 'image',
    params: [{ key: 'radius', label: 'Radius', type: 'range', min: 1, max: 300, step: 1, def: 30 }] },
  { id: 'sobel', label: 'Sobel Edge Detection', endpoint: '/edge-detection/sobel', kind: 'image', params: blur },
  { id: 'prewitt', label: 'Prewitt Edge Detection', endpoint: '/edge-detection/prewitt', kind: 'image', params: blur },
  { id: 'roberts', label: 'Roberts Edge Detection', endpoint: '/edge-detection/roberts', kind: 'image', params: blur },
  { id: 'canny', label: 'Canny Edge Detection', endpoint: '/edge-detection/canny', kind: 'image', params: [
    { key: 'threshold_mode', label: 'Threshold mode', type: 'select', options: ['manual', 'image_percentile', 'gradient_percentile'], def: 'manual' },
    { key: 'threshold1', label: 'Threshold 1', type: 'range', min: 0, max: 255, step: 1, def: 100, showIf: v => v.threshold_mode === 'manual' },
    { key: 'threshold2', label: 'Threshold 2', type: 'range', min: 0, max: 255, step: 1, def: 200, showIf: v => v.threshold_mode === 'manual' },
    ...blur] },
  { id: 'hybrid', label: 'Hybrid Image', endpoint: '/image/hybrid', kind: 'image', dual: true,
    note: 'Source = image 1 (low-pass). Both images must have the same dimensions. High-pass coefficient = 1 − low-pass.',
    params: [
      { key: 'radius', label: 'Radius', type: 'range', min: 1, max: 300, step: 1, def: 30 },
      { key: 'low_pass_coefficient', label: 'Low-pass coefficient', type: 'range', min: 0, max: 1, step: 0.05, def: 0.8 },
    ] },
  { id: 'hist_eq', label: 'Histogram Equalize', endpoint: '/histogram/equalize', kind: 'image',
    params: [{ key: 'grayscale', label: 'Grayscale', type: 'bool', def: false }] },
  { id: 'hist_graph', label: 'Histogram Graph (PNG)', endpoint: '/histogram/graph', kind: 'image', params: [
    { key: 'graph_type', label: 'Graph type', type: 'select', options: ['histogram', 'pdf', 'cdf', 'equalized'], def: 'histogram' },
    { key: 'grayscale', label: 'Grayscale', type: 'bool', def: false }] },
  { id: 'hist_data', label: 'Histogram Data (JSON)', endpoint: '/histogram/data', kind: 'json',
    params: [{ key: 'grayscale', label: 'Grayscale', type: 'bool', def: false }] },
];
