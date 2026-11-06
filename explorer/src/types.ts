export type Feature = {
  key: string;
  label: string;
  unit: string;
  group: string;
  decimals: number;
  min: number;
  max: number;
};
export type Preset = {
  id: string;
  name: string;
  description: string;
  values: number[];
};
export type Metric = { name: string; mae: number; rmse: number; r2: number };
export type Release = {
  features: Feature[];
  presets: Preset[];
  metrics: Metric[];
  scatter: { actual: number; predicted: number }[];
  parity: { samples: number; maxAbsoluteError: number; tolerance: number };
  hashes: Record<string, string>;
};
