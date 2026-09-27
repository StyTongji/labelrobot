import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { AnnotationForm } from "./components/AnnotationForm";
import { Timeline } from "./components/Timeline";
import type { Annotation, DatasetInfo, Episode, Link } from "./types";
import "./style.css";

type Operation = { action: "upsert"; annotation: Annotation; links?: Link[] } | { action: "delete"; id: string };
type History = { forward: Operation[]; inverse: Operation[] };

export default function App() {
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [frame, setFrame] = useState(0);
  const [timestamps, setTimestamps] = useState<number[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [links, setLinks] = useState<Link[]>([]);
  const [revision, setRevision] = useState(0);
  const [tools, setTools] = useState<Record<string, string[]>>({});
  const [editing, setEditing] = useState<Annotation | null>(null);
  const [row, setRow] = useState<Record<string, unknown>>({});
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [history, setHistory] = useState<History[]>([]);
  const [redo, setRedo] = useState<History[]>([]);
  const [notice, setNotice] = useState("Loading…");

  useEffect(() => { Promise.all([api.dataset(), api.episodes(), api.tools()]).then(([info, list, vocabulary]) => {
    setDataset(info); setEpisodes(list); setTools(vocabulary); setEpisode(list[0] ?? null); setNotice("");
  }).catch((error) => setNotice(error.message)); }, []);

  useEffect(() => {
    if (!episode) return;
    setFrame(0); setEditing(null); setHistory([]); setRedo([]);
    Promise.all([api.index(episode.episode_index), api.annotations(episode.episode_index)]).then(([index, saved]) => {
      setTimestamps(index.timestamp_s); setAnnotations(saved.annotations); setLinks(saved.links); setRevision(saved.revision);
    }).catch((error) => setNotice(error.message));
  }, [episode]);

  const numericColumns = useMemo(() => dataset ? Object.entries(dataset.features)
    .filter(([, feature]) => !["video", "image"].includes(feature.dtype)).map(([key]) => key) : [], [dataset]);
  useEffect(() => {
    if (!episode) return;
    api.rows(episode.episode_index, frame, ["frame_index", "timestamp", ...numericColumns]).then((rows) => setRow(rows[0] ?? {})).catch((error) => setNotice(error.message));
  }, [episode, frame, numericColumns]);

  useEffect(() => {
    if (!playing || !episode || !dataset) return;
    let cancelled = false;
    let timer = 0;
    const started = performance.now();
    const next = frame + 1 >= episode.length ? 0 : frame + 1;
    Promise.all(dataset.camera_keys.map((camera) => new Promise<void>((resolve) => {
      const image = new Image();
      image.onload = image.onerror = () => resolve();
      image.src = frameUrl(episode.episode_index, next, camera);
    }))).then(() => {
      if (cancelled) return;
      const delay = Math.max(0, 1000 / dataset.fps / speed - (performance.now() - started));
      timer = window.setTimeout(() => setFrame(next), delay);
    });
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [playing, episode, dataset, speed, frame]);

  async function run(operations: Operation[], inverse: Operation[], record = true): Promise<boolean> {
    try {
      const result = await api.command({ command_id: crypto.randomUUID(), expected_revision: revision, operations });
      setRevision(result.revision);
      const saved = await api.annotations(episode!.episode_index);
      setAnnotations(saved.annotations); setLinks(saved.links);
      if (record) { setHistory((items) => [...items.slice(-99), { forward: operations, inverse }]); setRedo([]); }
      setEditing(null); setNotice("Saved");
      return true;
    } catch (error) { setNotice(error instanceof Error ? error.message : String(error)); return false; }
  }

  function save(annotation: Annotation, newLinks: Link[]) {
    const previous = annotations.find((item) => item.id === annotation.id);
    const previousLinks = links.filter((link) => link.source_id === annotation.id);
    const forward: Operation[] = [{ action: "upsert", annotation, ...(annotation.kind === "verification" ? { links: newLinks } : {}) }];
    const inverse: Operation[] = previous ? [{ action: "upsert", annotation: previous, ...(previous.kind === "verification" ? { links: previousLinks } : {}) }] : [{ action: "delete", id: annotation.id }];
    void run(forward, inverse);
  }

  function remove(annotation: Annotation) {
    const annotationLinks = links.filter((link) => link.source_id === annotation.id);
    void run([{ action: "delete", id: annotation.id }], [{ action: "upsert", annotation, ...(annotationLinks.length ? { links: annotationLinks } : {}) }]);
  }

  async function undo() {
    const item = history.at(-1); if (!item) return;
    if (await run(item.inverse, item.forward, false)) { setHistory((items) => items.slice(0, -1)); setRedo((items) => [...items, item]); }
  }
  async function redoAction() {
    const item = redo.at(-1); if (!item) return;
    if (await run(item.forward, item.inverse, false)) { setRedo((items) => items.slice(0, -1)); setHistory((items) => [...items, item]); }
  }

  async function snapshot() {
    try { const result = await api.snapshot(); setNotice(`Snapshot ${result.snapshot_id} created`); }
    catch (error) { setNotice(error instanceof Error ? error.message : String(error)); }
  }

  function seekTimestamp(timestamp: number) {
    if (!timestamps.length || !Number.isFinite(timestamp)) return;
    let low = 0; let high = timestamps.length - 1;
    while (low < high) { const middle = Math.floor((low + high) / 2); if (timestamps[middle] < timestamp) low = middle + 1; else high = middle; }
    if (low > 0 && timestamp - timestamps[low - 1] <= timestamps[low] - timestamp) low -= 1;
    setFrame(low);
  }

  if (!dataset || !episode) return <main className="loading">{notice || "No episodes"}</main>;
  return <main>
    <nav className="topbar"><div><b>LabelRobot</b><small>{dataset.format_version} · {dataset.total_frames.toLocaleString()} frames</small></div><div className="nav-actions"><button disabled={!history.length} onClick={undo}>Undo</button><button disabled={!redo.length} onClick={redoAction}>Redo</button><button onClick={snapshot}>Create snapshot</button></div></nav>
    <div className="workspace">
      <aside className="episodes"><h2>Episodes</h2>{episodes.map((item) => <button className={item.episode_index === episode.episode_index ? "selected" : ""} key={item.episode_index} onClick={() => setEpisode(item)}><b>#{item.episode_index}</b><span>{item.length} frames</span><small>{item.tasks.join(", ") || "No instruction"}</small></button>)}</aside>
      <section className="content">
        <div className="viewer-toolbar"><button onClick={() => setFrame(Math.max(0, frame - 1))}>◀ Frame</button><button className="primary" onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button><button onClick={() => setFrame(Math.min(episode.length - 1, frame + 1))}>Frame ▶</button><label>Speed<select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value={0.25}>0.25×</option><option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option></select></label><label>Timestamp <input type="number" step="any" value={timestamps[frame] ?? 0} onChange={(event) => seekTimestamp(Number(event.target.value))} /></label></div>
        <div className="viewer-grid">{dataset.camera_keys.length ? dataset.camera_keys.map((camera) => <figure key={camera}><img src={frameUrl(episode.episode_index, frame, camera)} /><figcaption>{camera}</figcaption></figure>) : <div className="empty-camera">Dataset has no camera feature</div>}</div>
        <Timeline frame={frame} length={episode.length} annotations={annotations} onSeek={setFrame} onEdit={setEditing} onChange={(annotation) => save(annotation, links.filter((link) => link.source_id === annotation.id))} />
        <section className="annotation-tree"><h2>Hierarchy and events</h2>{annotations.map((annotation) => <button key={annotation.id} onClick={() => setEditing(annotation)} style={{ paddingLeft: `${12 + depth(annotation, annotations) * 18}px` }}><span>{annotation.kind}</span><b>{annotation.label ?? annotation.description ?? annotation.id}</b><small>{annotation.start_frame === null ? "episode" : annotation.end_frame_exclusive ? `[${annotation.start_frame}, ${annotation.end_frame_exclusive})` : `@ ${annotation.start_frame}`}</small></button>)}</section>
      </section>
      <aside className="inspector"><AnnotationForm episode={episode.episode_index} frame={frame} length={episode.length} project={dataset.project} annotations={annotations} links={links} tools={tools} editing={editing} onSave={save} onDelete={remove} onCancel={() => setEditing(null)} /><section className="row-data"><h2>Frame data</h2><pre>{JSON.stringify(describeRow(row, dataset), null, 2)}</pre></section></aside>
    </div>
    <footer className={notice.includes("Saved") || notice.includes("Snapshot") ? "ok" : ""}>{notice || `revision ${revision}`}</footer>
  </main>;
}

function frameUrl(episode: number, frame: number, camera: string) {
  return `/api/episodes/${episode}/frames/${frame}/${encodeURIComponent(camera)}`;
}

function depth(annotation: Annotation, all: Annotation[]) {
  let result = 0; let parent = annotation.parent_id;
  while (parent && result < 20) { result += 1; parent = all.find((item) => item.id === parent)?.parent_id ?? null; }
  return result;
}

function describeRow(row: Record<string, unknown>, dataset: DatasetInfo) {
  return Object.fromEntries(Object.entries(row).map(([key, value]) => {
    const names = dataset.features[key]?.names;
    return [key, names?.length ? { names, values: value } : value];
  }));
}
