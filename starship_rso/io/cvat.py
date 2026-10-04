"""Import/export point tracks in CVAT "for video 1.1" XML.

Workflow: run the pipeline -> export confirmed tracks as pseudo-labels ->
correct them in CVAT (point tracks, one label per category) -> import the
corrected XML as training/evaluation labels.

Each annotated point carries the track id, frame, position and these optional
attributes: ``visibility`` (clear|faint|ambiguous), ``identity`` (free text,
e.g. a NORAD id, with ``identity_source`` describing the evidence). Box tracks
are accepted on import and reduced to their centres.

Frame numbers in CVAT are relative to the uploaded clip; ``frame_offset``
converts them to indices of the source video.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LABELS = [
    "unknown",
    "near_field_particle",
    "payload_candidate",
    "vehicle_feature",
    "background_feature",
    "artifact",
]


@dataclass
class LabeledPoint:
    frame: int
    x: float
    y: float
    outside: bool = False
    occluded: bool = False
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class LabeledTrack:
    track_id: int
    label: str
    points: list[LabeledPoint] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)

    def visible(self) -> list[LabeledPoint]:
        return [p for p in self.points if not p.outside]


def export_tracks(
    tracks: list[LabeledTrack],
    path: str | Path,
    width: int,
    height: int,
    n_frames: int,
    labels: list[str] | None = None,
    frame_offset: int = 0,
) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "size").text = str(n_frames)
    ET.SubElement(task, "mode").text = "interpolation"
    osz = ET.SubElement(task, "original_size")
    ET.SubElement(osz, "width").text = str(width)
    ET.SubElement(osz, "height").text = str(height)
    labs = ET.SubElement(task, "labels")
    for name in labels or DEFAULT_LABELS:
        lab = ET.SubElement(labs, "label")
        ET.SubElement(lab, "name").text = name
        ET.SubElement(lab, "type").text = "points"
        attrs = ET.SubElement(lab, "attributes")
        for an, values in (("visibility", "clear\nfaint\nambiguous"), ("identity", ""), ("identity_source", "")):
            a = ET.SubElement(attrs, "attribute")
            ET.SubElement(a, "name").text = an
            ET.SubElement(a, "mutable").text = "False" if an != "visibility" else "True"
            ET.SubElement(a, "input_type").text = "select" if an == "visibility" else "text"
            ET.SubElement(a, "default_value").text = "clear" if an == "visibility" else ""
            ET.SubElement(a, "values").text = values
    for tr in tracks:
        # CVAT rejects shapes outside the task: keep only frames in [frame_offset, frame_offset + n_frames)
        pts = sorted((p for p in tr.points if 0 <= p.frame - frame_offset < n_frames), key=lambda p: p.frame)
        if not pts:
            continue
        te = ET.SubElement(root, "track", id=str(tr.track_id), label=tr.label, source="auto")
        for i, p in enumerate(pts):
            pe = ET.SubElement(
                te,
                "points",
                frame=str(p.frame - frame_offset),
                keyframe="1",
                outside="1" if p.outside else "0",
                occluded="1" if p.occluded else "0",
                points=f"{p.x:.2f},{p.y:.2f}",
                z_order="0",
            )
            attrs = {**tr.attributes, **p.attributes}
            attrs.setdefault("visibility", "clear")
            for k, v in attrs.items():
                ET.SubElement(pe, "attribute", name=k).text = str(v)
        # close the track one frame after its last point, as CVAT expects (only if that frame exists)
        if pts and not pts[-1].outside and pts[-1].frame + 1 - frame_offset < n_frames:
            last = pts[-1]
            ET.SubElement(
                te,
                "points",
                frame=str(last.frame + 1 - frame_offset),
                keyframe="1",
                outside="1",
                occluded="0",
                points=f"{last.x:.2f},{last.y:.2f}",
                z_order="0",
            )
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def import_tracks(path: str | Path, frame_offset: int = 0) -> tuple[list[LabeledTrack], dict]:
    """Return (tracks, meta) where meta has width/height/size if present."""
    root = ET.parse(path).getroot()
    meta: dict = {}
    # Element truthiness depends on children, so test against None explicitly.
    w = root.find("./meta/task/original_size/width")
    if w is None:
        w = root.find("./meta/job/original_size/width")
    h = root.find("./meta/task/original_size/height")
    if h is None:
        h = root.find("./meta/job/original_size/height")
    if w is not None and h is not None:
        meta["width"], meta["height"] = int(w.text), int(h.text)
    tracks: list[LabeledTrack] = []
    for te in root.findall("track"):
        tr = LabeledTrack(track_id=int(te.get("id", len(tracks))), label=te.get("label", "unknown"))
        for shape in list(te):
            if shape.tag not in ("points", "box"):
                continue
            frame = int(shape.get("frame")) + frame_offset
            outside = shape.get("outside", "0") == "1"
            occluded = shape.get("occluded", "0") == "1"
            if shape.tag == "points":
                first = shape.get("points").split(";")[0]
                x, y = (float(v) for v in first.split(","))
            else:
                x = 0.5 * (float(shape.get("xtl")) + float(shape.get("xbr")))
                y = 0.5 * (float(shape.get("ytl")) + float(shape.get("ybr")))
            attrs = {a.get("name"): (a.text or "") for a in shape.findall("attribute")}
            tr.points.append(LabeledPoint(frame, x, y, outside, occluded, attrs))
        tr.points.sort(key=lambda p: p.frame)
        for k in ("identity", "identity_source"):
            vals = {p.attributes.get(k) for p in tr.points if p.attributes.get(k)}
            if len(vals) == 1:
                tr.attributes[k] = vals.pop()
        tracks.append(tr)
    return tracks, meta


def keyframes_only_warning(tracks: list[LabeledTrack]) -> list[int]:
    """Track ids whose visible points skip frames (CVAT interpolation must be reviewed)."""
    bad = []
    for tr in tracks:
        f = [p.frame for p in tr.visible()]
        if len(f) > 1 and any(b - a > 1 for a, b in zip(f, f[1:])):
            bad.append(tr.track_id)
    return bad


def interpolate_track(tr: LabeledTrack) -> LabeledTrack:
    """Linearly fill frames between consecutive visible keyframes (review the result)."""
    out = LabeledTrack(tr.track_id, tr.label, [], dict(tr.attributes))
    vis = sorted(tr.points, key=lambda p: p.frame)
    for a, b in zip(vis, vis[1:]):
        out.points.append(a)
        if a.outside or b.outside:
            continue
        for f in range(a.frame + 1, b.frame):
            w = (f - a.frame) / (b.frame - a.frame)
            out.points.append(LabeledPoint(f, a.x + w * (b.x - a.x), a.y + w * (b.y - a.y), False, False, {}))
    if vis:
        out.points.append(vis[-1])
    return out
