import { useState } from "react";
import type { Annotation } from "../types";

type Props = {
  frame: number;
  length: number;
  annotations: Annotation[];
  onSeek: (frame: number) => void;
  onEdit: (annotation: Annotation) => void;
  onChange: (annotation: Annotation) => void;
};

const colors: Record<string, string> = {
  subtask: "#4f8cff", recovery: "#8b5cf6", observed_event: "#f97316",
  expected_outcome: "#eab308", verification: "#22c55e", progress: "#06b6d4",
};

export function Timeline({ frame, length, annotations, onSeek, onEdit, onChange }: Props) {
  const [zoom, setZoom] = useState(1);
  const timed = annotations.filter((annotation) => annotation.start_frame !== null);
  function beginDrag(event: React.PointerEvent, annotation: Annotation, mode: "move" | "start" | "end") {
    if (annotation.scope !== "segment") return;
    event.preventDefault(); event.stopPropagation();
    const lane = (event.currentTarget as HTMLElement).closest(".track-lane") as HTMLElement;
    const originX = event.clientX;
    const originStart = annotation.start_frame!;
    const originEnd = annotation.end_frame_exclusive!;
    let latest = annotation;
    function move(pointer: PointerEvent) {
      const delta = Math.round((pointer.clientX - originX) / lane.clientWidth * length);
      let start = originStart; let end = originEnd;
      if (mode === "move") { start = Math.max(0, Math.min(length - (originEnd - originStart), originStart + delta)); end = start + originEnd - originStart; }
      if (mode === "start") start = Math.max(0, Math.min(originEnd - 1, originStart + delta));
      if (mode === "end") end = Math.max(originStart + 1, Math.min(length, originEnd + delta));
      latest = { ...annotation, start_frame: start, end_frame_exclusive: end };
      const clip = lane.querySelector(`[data-id="${annotation.id}"]`) as HTMLElement | null;
      if (clip) { clip.style.left = `${start / length * 100}%`; clip.style.width = `${(end - start) / length * 100}%`; }
    }
    function finish() { window.removeEventListener("pointermove", move); if (latest !== annotation) onChange(latest); }
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", finish, { once: true });
  }
  return <section className="timeline-card">
    <header><strong>Timeline</strong><label>Zoom <input type="range" min={1} max={20} value={zoom} onChange={(event) => setZoom(Number(event.target.value))} /></label><span>frame {frame} / {Math.max(0, length - 1)}</span></header>
    <input className="scrubber" type="range" min={0} max={Math.max(0, length - 1)} value={frame}
      onChange={(event) => onSeek(Number(event.target.value))} />
    <div className="tracks-scroll"><div className="tracks" style={{ minWidth: `${zoom * 100}%` }}>
      <div className="ruler"><span>0</span><span>{Math.floor(length / 4)}</span><span>{Math.floor(length / 2)}</span><span>{Math.floor(length * 3 / 4)}</span><span>{Math.max(0, length - 1)}</span></div>
      {["subtask", "recovery", "expected_outcome", "observed_event", "verification", "progress"].map((kind) => {
        const items = timed.filter((annotation) => annotation.kind === kind);
        return <div className="track" key={kind}>
          <span className="track-label">{kind}</span>
          <div className="track-lane" onDoubleClick={(event) => {
            const bounds = event.currentTarget.getBoundingClientRect();
            onSeek(Math.round((event.clientX - bounds.left) / bounds.width * Math.max(0, length - 1)));
          }}>
            {items.map((annotation) => {
              const start = annotation.start_frame ?? 0;
              const end = annotation.end_frame_exclusive ?? start + 1;
              return <button key={annotation.id} data-id={annotation.id} className="clip" title={`${annotation.label ?? kind} [${start}, ${end})`}
                onDoubleClick={() => onEdit(annotation)} onPointerDown={(event) => beginDrag(event, annotation, "move")} style={{
                  left: `${start / Math.max(1, length) * 100}%`,
                  width: `${Math.max(0.6, (end - start) / Math.max(1, length) * 100)}%`,
                  background: colors[kind],
                }}><i className="resize start" onPointerDown={(event) => beginDrag(event, annotation, "start")} />{annotation.label ?? "•"}<i className="resize end" onPointerDown={(event) => beginDrag(event, annotation, "end")} /></button>;
            })}
            <i className="playhead" style={{ left: `${frame / Math.max(1, length) * 100}%` }} />
          </div>
        </div>;
      })}
    </div></div>
  </section>;
}
