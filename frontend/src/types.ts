export type AnnotationKind =
  | "subtask"
  | "recovery"
  | "observed_event"
  | "expected_outcome"
  | "verification"
  | "progress"
  | "episode_evaluation";

export type Annotation = {
  id: string;
  dataset_id: string;
  dataset_version: string;
  episode_index: number;
  kind: AnnotationKind;
  scope: "point" | "segment" | "frame_value" | "episode";
  start_frame: number | null;
  end_frame_exclusive: number | null;
  timestamp_s?: number | null;
  end_timestamp_s?: number | null;
  parent_id: string | null;
  camera_key: string | null;
  label: string | null;
  value_float?: number | null;
  score?: number | null;
  success?: boolean | null;
  description?: string | null;
  attributes: Record<string, unknown>;
};

export type Link = {
  source_id: string;
  relation: "verifies_expected" | "supported_by";
  target_id: string;
};

export type Episode = { episode_index: number; length: number; tasks: string[] };
export type DatasetInfo = {
  dataset_id: string;
  dataset_version: string;
  format_version: string;
  fps: number;
  features: Record<string, { dtype: string; shape?: number[]; names?: string[] }>;
  camera_keys: string[];
  total_episodes: number;
  total_frames: number;
  project: { dataset_id: string; dataset_version: string; annotation_revision: number };
};
