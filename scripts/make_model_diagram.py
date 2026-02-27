from graphviz import Digraph
from pathlib import Path

def make_model_diagram(out_path: str = "docs/model_diagram.png") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g = Digraph("model", format="png")
    g.attr(rankdir="LR", splines="true", nodesep="0.35", ranksep="0.5")
    g.attr("node", shape="box", style="rounded", fontsize="11")

    # Observed + deterministic
    g.node("y", "Observed\n$y_{i,m,t}$", shape="box", style="rounded,filled", fillcolor="#f2f2f2")

    g.node("mu", "Linear predictor\n$\\mu_{i,m,t}$", shape="box")

    # Fixed effects
    g.node("alpha", "Global intercept\n$\\alpha$", shape="ellipse")
    g.node("beta", "Global time effect\n$\\beta$", shape="ellipse")
    g.node("time", "Time indicator\n$time_t\\in\\{0,1\\}$", shape="box", style="rounded")

    # Subject random effects
    g.node("b0i", "Subject RE intercept\n$b_{0i}$", shape="ellipse")
    g.node("b1i", "Subject RE slope\n$b_{1i}$", shape="ellipse")
    g.node("sigma_id", "Subject RE scales\n$\\sigma_{id}$", shape="ellipse")
    g.node("corr_id", "Subject RE corr\n$\\rho_{id}$", shape="ellipse")
    g.node("L_id", "Cholesky\n$L_{id}$", shape="box")

    # Measure random effects
    g.node("u0m", "Measure RE intercept\n$u_{0m}$", shape="ellipse")
    g.node("u1m", "Measure RE slope\n$u_{1m}$", shape="ellipse")
    g.node("sd_u0", "Measure RE sd\n$sd_{u0}$", shape="ellipse")
    g.node("sd_u1", "Measure RE sd\n$sd_{u1}$", shape="ellipse")

    # Noise
    g.node("sigma", "Residual sd\n$\\sigma$", shape="ellipse")

    # Deterministic measure-specific slope
    g.node("beta_m", "Measure-specific time effect\n$\\beta_m = \\beta + u_{1m}$", shape="box")

    # Edges: priors -> parameters -> mu -> y
    g.edge("alpha", "mu")
    g.edge("beta", "mu")
    g.edge("time", "mu")

    g.edge("b0i", "mu")
    g.edge("b1i", "mu")

    g.edge("u0m", "mu")
    g.edge("u1m", "mu")

    g.edge("mu", "y")
    g.edge("sigma", "y")

    # Subject covariance construction
    g.edge("sigma_id", "L_id")
    g.edge("corr_id", "L_id")
    g.edge("L_id", "b0i")
    g.edge("L_id", "b1i")

    # Measure RE priors
    g.edge("sd_u0", "u0m")
    g.edge("sd_u1", "u1m")

    # Derived per-measure effect
    g.edge("beta", "beta_m")
    g.edge("u1m", "beta_m")

    # Rendering
    # Graphviz writes without the suffix; so pass path without ".png"
    render_path = out_path.with_suffix("")
    g.render(str(render_path), cleanup=True)

    # Graphviz will output render_path + ".png"
    produced = render_path.with_suffix(".png")
    if produced != out_path:
        produced.replace(out_path)

if __name__ == "__main__":
    make_model_diagram("docs/model_diagram.png")
    print("Wrote docs/model_diagram.png")