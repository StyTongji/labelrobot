import type { Annotation, DatasetInfo, Episode, Link } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? response.statusText);
  }
  return response.json();
}

export const api = {
  dataset: () => request<DatasetInfo>("/api/dataset"),
  episodes: () => request<Episode[]>("/api/episodes?limit=1000"),
  index: (episode: number) => request<{ frame_index: number[]; timestamp_s: number[] }>(`/api/episodes/${episode}/index`),
  rows: (episode: number, frame: number, columns: string[]) =>
    request<Record<string, unknown>[]>(`/api/episodes/${episode}/rows?start=${frame}&stop=${frame + 1}&columns=${encodeURIComponent(columns.join(","))}`),
  annotations: (episode: number) => request<{ revision: number; annotations: Annotation[]; links: Link[] }>(`/api/episodes/${episode}/annotations`),
  tools: () => request<Record<string, string[]>>("/api/tools"),
  command: (body: object) => request<{ revision: number }>("/api/annotation-commands", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }),
  snapshot: () => request<{ snapshot_id: string }>("/api/snapshots", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
  }),
};
