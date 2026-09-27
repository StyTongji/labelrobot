import { useEffect, useMemo, useState } from "react";
import type { Annotation, AnnotationKind, Link } from "../types";

type Props = {
  episode: number;
  frame: number;
  length: number;
  project: { dataset_id: string; dataset_version: string };
  annotations: Annotation[];
  links: Link[];
  tools: Record<string, string[]>;
  editing: Annotation | null;
  onSave: (annotation: Annotation, links: Link[]) => void;
  onDelete: (annotation: Annotation) => void;
  onCancel: () => void;
};

const scope: Record<AnnotationKind, Annotation["scope"]> = {
  subtask: "segment", recovery: "segment", observed_event: "point", expected_outcome: "point",
  verification: "point", progress: "frame_value", episode_evaluation: "episode",
};

export function AnnotationForm(props: Props) {
  const [kind, setKind] = useState<AnnotationKind>("subtask");
  const [label, setLabel] = useState("");
  const [start, setStart] = useState(props.frame);
  const [end, setEnd] = useState(Math.min(props.length, props.frame + 1));
  const [parent, setParent] = useState("");
  const [description, setDescription] = useState("");
  const [value, setValue] = useState("0");
  const [score, setScore] = useState("3");
  const [success, setSuccess] = useState("unknown");
  const [expected, setExpected] = useState("");
  const [observed, setObserved] = useState<string[]>([]);
  const [result, setResult] = useState("uncertain");

  useEffect(() => {
    const item = props.editing;
    setKind(item?.kind ?? "subtask"); setLabel(item?.label ?? "");
    setStart(item?.start_frame ?? props.frame); setEnd(item?.end_frame_exclusive ?? Math.min(props.length, props.frame + 1));
    setParent(item?.parent_id ?? ""); setDescription(item?.description ?? "");
    setValue(String(item?.value_float ?? 0)); setScore(String(item?.score ?? 3));
    setSuccess(item?.success === null || item?.success === undefined ? "unknown" : String(item.success));
    setResult(String(item?.attributes.result ?? "uncertain"));
    const itemLinks = props.links.filter((link) => link.source_id === item?.id);
    setExpected(itemLinks.find((link) => link.relation === "verifies_expected")?.target_id ?? "");
    setObserved(itemLinks.filter((link) => link.relation === "supported_by").map((link) => link.target_id));
    if (item?.kind === "expected_outcome") setEnd(Number(item.attributes.window_end_frame_exclusive ?? Math.min(props.length, (item.start_frame ?? props.frame) + 1)));
  }, [props.editing, props.frame, props.length, props.links]);

  const parents = useMemo(() => props.annotations.filter((item) => ["subtask", "recovery"].includes(item.kind) && item.id !== props.editing?.id), [props.annotations, props.editing]);
  const expectations = props.annotations.filter((item) => item.kind === "expected_outcome");
  const events = props.annotations.filter((item) => item.kind === "observed_event");
  const labels = props.tools[kind] ?? [];

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const isEpisode = kind === "episode_evaluation";
    const isSegment = ["subtask", "recovery"].includes(kind) || (kind === "observed_event" && end > start + 1);
    const attributes: Record<string, unknown> = {};
    if (kind === "expected_outcome") Object.assign(attributes, { provenance: "retrospective", window_start_frame: start, window_end_frame_exclusive: end });
    if (kind === "verification") Object.assign(attributes, { result });
    const annotation: Annotation = {
      id: props.editing?.id ?? crypto.randomUUID(), dataset_id: props.project.dataset_id,
      dataset_version: props.project.dataset_version, episode_index: props.episode, kind,
      scope: isEpisode ? "episode" : isSegment ? "segment" : scope[kind],
      start_frame: isEpisode ? null : start, end_frame_exclusive: isSegment ? end : null,
      parent_id: parent || null, camera_key: null, label: label || null,
      value_float: kind === "progress" ? Number(value) : null,
      score: isEpisode ? Number(score) : null,
      success: isEpisode ? success === "unknown" ? null : success === "true" : null,
      description: description || null, attributes,
    };
    const links: Link[] = kind === "verification" ? [
      ...(expected ? [{ source_id: annotation.id, relation: "verifies_expected" as const, target_id: expected }] : []),
      ...observed.map((target_id) => ({ source_id: annotation.id, relation: "supported_by" as const, target_id })),
    ] : [];
    props.onSave(annotation, links);
  }

  return <form className="annotation-form" onSubmit={submit}>
    <header><strong>{props.editing ? "Edit annotation" : "New annotation"}</strong></header>
    <label>Type<select value={kind} onChange={(event) => setKind(event.target.value as AnnotationKind)}>
      {Object.keys(scope).map((value) => <option key={value}>{value}</option>)}
    </select></label>
    {!(["progress", "verification", "episode_evaluation"].includes(kind)) && <label>Label
      {labels.length ? <select value={label} onChange={(event) => setLabel(event.target.value)}><option value="">Select…</option>{labels.map((item) => <option key={item}>{item}</option>)}</select>
        : <input value={label} onChange={(event) => setLabel(event.target.value)} />}
    </label>}
    {kind !== "episode_evaluation" && <label>Frame<input type="number" min={0} max={props.length - 1} value={start} onChange={(event) => setStart(Number(event.target.value))} /></label>}
    {["subtask", "recovery", "expected_outcome", "observed_event"].includes(kind) && <label>End (exclusive)<input type="number" min={start + 1} max={props.length} value={end} onChange={(event) => setEnd(Number(event.target.value))} /></label>}
    {kind !== "episode_evaluation" && <label>Parent subtask<select value={parent} onChange={(event) => setParent(event.target.value)}><option value="">None</option>{parents.map((item) => <option value={item.id} key={item.id}>{item.label} [{item.start_frame}, {item.end_frame_exclusive})</option>)}</select></label>}
    {kind === "progress" && <label>Progress<input type="number" min={0} max={1} step={0.01} value={value} onChange={(event) => setValue(event.target.value)} /></label>}
    {kind === "episode_evaluation" && <><label>Score<input type="number" min={0} max={5} value={score} onChange={(event) => setScore(event.target.value)} /></label><label>Success<select value={success} onChange={(event) => setSuccess(event.target.value)}><option>unknown</option><option>true</option><option>false</option></select></label></>}
    {kind === "verification" && <><label>Expected<select value={expected} required onChange={(event) => setExpected(event.target.value)}><option value="">Select…</option>{expectations.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><label>Observed events<select multiple value={observed} onChange={(event) => setObserved([...event.target.selectedOptions].map((option) => option.value))}>{events.map((item) => <option key={item.id} value={item.id}>{item.label} @ {item.start_frame}</option>)}</select></label><label>Result<select value={result} onChange={(event) => setResult(event.target.value)}><option>met</option><option>not_met</option><option>uncertain</option></select></label></>}
    <label>Note<textarea value={description} onChange={(event) => setDescription(event.target.value)} /></label>
    <div className="form-actions"><button type="submit">Save</button>{props.editing && <button type="button" className="danger" onClick={() => props.onDelete(props.editing!)}>Delete</button>}<button type="button" onClick={props.onCancel}>Clear</button></div>
  </form>;
}
