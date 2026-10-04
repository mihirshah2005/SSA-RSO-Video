import json

import numpy as np
import pytest

from starship_rso.config import load_config


def test_heatmap_model_shapes_decode_and_learns():
    torch = pytest.importorskip("torch")
    from starship_rso.train.model import TemporalUNet, build_input, decode, focal_loss, input_channels

    torch.manual_seed(0)
    m = TemporalUNet(frames=3, base=8, depth=3)
    x = torch.rand(2, 3, 50, 70) * 255
    inp = build_input(x[:, -1:], [x[:, 1:2], x[:, 0:1]])
    assert inp.shape[1] == input_channels(3) == 5
    hl, of = m(inp)
    assert hl.shape == (2, 1, 50, 70) and of.shape == (2, 2, 50, 70)
    # decode finds a planted peak with its sub-pixel offset
    logits = torch.full((1, 1, 20, 20), -8.0)
    logits[0, 0, 7, 11] = 4.0
    off = torch.zeros(1, 2, 20, 20)
    off[0, 0, 7, 11], off[0, 1, 7, 11] = 0.25, -0.3
    d = decode(logits, off, 0.5)[0]
    assert d.shape[0] == 1 and abs(float(d[0, 0]) - 11.25) < 1e-6 and abs(float(d[0, 1]) - 6.7) < 1e-6
    # a few optimisation steps reduce the focal loss on a fixed toy target
    tgt = torch.zeros(1, 1, 32, 32)
    tgt[0, 0, 16, 16] = 1.0
    w = torch.ones_like(tgt)
    xin = torch.rand(1, 5, 32, 32)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    l0 = None
    for _ in range(30):
        loss = focal_loss(m(xin)[0], tgt, w)
        l0 = l0 if l0 is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < l0


def test_cli_parser_and_feasibility(capsys):
    from starship_rso.cli import build_parser, main

    p = build_parser()
    a = p.parse_args(["run", "--video", "x.mp4", "-c", "configs/default.yaml", "--no-display"])
    assert a.cmd == "run" and a.no_display
    main(["feasibility"])
    assert "pointing error" in capsys.readouterr().out


@pytest.mark.slow
def test_end_to_end_synthetic_identification(tmp_path):
    """Render a short clip, run the causal pipeline, check detection and identity against truth."""
    from starship_rso.cli import main
    from starship_rso.pipeline.runner import run
    from starship_rso.sim.scene import SimSpec, simulate

    spec = SimSpec(width=640, height=360, duration_s=6.0, n_passes=1, n_releases=1, n_particles=4, seed=3,
                   pass_range_km=[6.0, 9.0])
    sim = simulate(tmp_path / "sim", spec)
    cfg = load_config(["configs/default.yaml", str(sim / "mission.yaml")], ["realtime.enabled=false"])
    out = run(cfg, str(sim / "video.mp4"), str(tmp_path / "run"), display=False, mode="synthetic")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["frames_processed"] == int(spec.duration_s * spec.fps)
    main(["evaluate", "--run", str(out), "--truth", str(sim / "truth.json"), "--skip", "30"])
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["detection_by_kind"]["catalog"]["recall"] > 0.9
    ident = metrics["identity"]
    assert ident["accepted_wrong"] == 0, ident
    assert ident["accepted_correct"] >= 1, ident
    # one release, one release event (fragments of the resolved payload must not add more)
    assert len(summary["release_events"]) <= spec.n_releases, summary["release_events"]
    # nothing is reported while the vehicle mask is still being learned
    frames = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert frames[0]["warm"] and not any(f["dets"] for f in frames if f["warm"])
    assert not frames[-1]["warm"]
